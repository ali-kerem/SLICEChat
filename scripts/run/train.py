from pathlib import Path
from dotenv import load_dotenv

# Load Hugging Face settings before libraries read them at import time.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import os
import yaml
import json
import argparse
import random

import numpy as np
import torch
from torch.backends import cudnn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy

from src.data_utils.preprocessors import PreprocessorFactory
from src.mllms import MLLMFactory
from src.modules import MLLMLightningModule, MLLMDataModule
from src.utils.checkpoint import get_checkpoint_config
from src.utils.distributed import get_ranks, get_num_nodes, calculate_accumulation_steps
from src.utils.lora import apply_lora


def seed_everything(seed: int, deterministic: bool) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if deterministic:
        # Trade throughput for reproducibility. Note: fused Mamba/flash-attention
        # kernels may not support deterministic algorithms and could error here.
        cudnn.benchmark = False
        cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.benchmark = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MLLM")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--run-name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory")
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # If it we are continuing from a checkpoint, we can infer llm_name_or_path, modality, encoder_type, projector_type 
    # and encoder_kwargs from the ckpt_path
    if config["model"].get("ckpt_path", None) is not None:
        ckpt_config = get_checkpoint_config(config["model"]["ckpt_path"])
        config["model"]["llm_name_or_path"] = ckpt_config["model"]["llm_name_or_path"]
        config["model"]["modality"] = ckpt_config["model"]["modality"]
        config["model"]["mllm_args"]["encoder_type"] = ckpt_config["model"]["mllm_args"]["encoder_type"]
        config["model"]["mllm_args"]["projector_type"] = ckpt_config["model"]["mllm_args"]["projector_type"]
        config["model"]["mllm_args"]["encoder_kwargs"] = ckpt_config["model"]["mllm_args"]["encoder_kwargs"]
        if "qwen2_5_vl_mrope" in ckpt_config["model"]["mllm_args"]:
            config["model"]["mllm_args"]["qwen2_5_vl_mrope"] = ckpt_config["model"]["mllm_args"]["qwen2_5_vl_mrope"]
        config["data"]["preprocessor_args"]["use_image_start_end_tokens"] = ckpt_config["data"]["preprocessor_args"]["use_image_start_end_tokens"]
        config["data"]["preprocessor_args"]["remove_system_prompt"] = ckpt_config["data"]["preprocessor_args"]["remove_system_prompt"]
        config["data"]["dataset_args"]["system_prompt"] = ckpt_config["data"]["dataset_args"]["system_prompt"]
        config["data"]["dataset_args"]["token_compression_rate"] = ckpt_config["data"]["dataset_args"]["token_compression_rate"]

    if config["train_args"]["resume_from_checkpoint"]:
        assert config["model"].get("ckpt_path", None) is not None, "ckpt_path is required for resume_from_checkpoint"

        ckpt_config = get_checkpoint_config(config["model"]["ckpt_path"])

        ckpt_config["model"]["ckpt_path"] = config["model"]["ckpt_path"]
        ckpt_config["train_args"]["resume_from_checkpoint"] = True
        config = ckpt_config

        config["log_dir"] = os.path.dirname(os.path.dirname(config["model"]["ckpt_path"]))
    else:
        config["log_dir"] = args.log_dir
    
    if config.get("run_name", None) is None:
        config['run_name'] = args.run_name

    # Older checkpoints saved this argument for the removed text-guidance encoders.
    config["data"]["dataset_args"].pop("return_text", None)

    config["data"]["dataset_args"]["token_compression_rate"] = float(config["data"]["dataset_args"]["token_compression_rate"])

    # Only print configuration from the main process (rank 0)
    rank, local_rank = get_ranks()
    if rank == 0 and local_rank == 0:
        print("Configuration:", flush=True)
        print(json.dumps(config, indent=2), flush=True)

    return config


def main():
    args = parse_args()

    os.makedirs(args["log_dir"], exist_ok=True)

    if not args["train_args"]["resume_from_checkpoint"]:
        with open(os.path.join(args["log_dir"], "args.yaml"), "w") as f:
            yaml.dump(args, f)

    if args["train_args"]["use_tensor_cores"]:
        torch.set_float32_matmul_precision("medium")

    seed_everything(
        seed=args["data"]["dataloader_args"]["seed"],
        deterministic=args["train_args"]["deterministic"],
    )

    if args["wandb"]["enabled"]:
        wandb_logger = WandbLogger(
            project=args["wandb"]["project_name"],
            name=args["run_name"]
        )
        wandb_logger.log_hyperparams(args)
    else:
        wandb_logger = None

    preprocessor = PreprocessorFactory.create(
        preprocessor_type=args["model"]["modality"],
        tokenizer_name_or_path=args["model"]["llm_name_or_path"],
        **args["data"]["preprocessor_args"]
    )

    ckpt_path = args["model"].get("ckpt_path")
    resume = args["train_args"]["resume_from_checkpoint"]
    placeholder_tokens = preprocessor.get_placeholder_token_name_to_ids()

    if ckpt_path is not None and not resume:
        # Load model weights; start a new optimizer/scheduler.
        print(f"Loading model weights from {ckpt_path}")
        model = MLLMFactory.from_pretrained(
            ckpt_path,
            placeholder_tokens=placeholder_tokens,
            config=args,
        )
    else:
        # Fresh training loads upstream weights.
        # Full resume builds from configs; DeepSpeed restores weights later.
        model = MLLMFactory.create(
            mllm_type=args["model"]["modality"],
            llm_name_or_path=args["model"]["llm_name_or_path"],
            placeholder_tokens=placeholder_tokens,
            init_from_config=resume,
            **args["model"]["mllm_args"],
        )

        if args["train_args"].get("use_lora", False) and not model.freeze_llm:
            apply_lora(model, args)

    resume_ckpt_path = ckpt_path if resume else None
    if resume:
        print(f"Will resume training from checkpoint: {resume_ckpt_path}")

    model.train()

    datamodule = MLLMDataModule(
        modality=args["model"]["modality"],
        preprocessor=preprocessor,
        dataloader_args=args["data"]["dataloader_args"],
        dataset_args=args["data"]["dataset_args"],
    )

    lightning_module = MLLMLightningModule(
        model=model,
        optimizer_args=args["optimizer"],
        cpu_adam=args["train_args"]["deepspeed_args"]["offload_optimizer"]
    )

    save_dir = os.path.join(args["log_dir"], "checkpoints")

    os.makedirs(save_dir, exist_ok=True)

    # Save last checkpoint based on global_step. The checkpoint is saved every every_n_train_steps * global_batch_size
    # elements are processed.
    checkpoint_callback_last = ModelCheckpoint(
        dirpath=save_dir,
        filename="last-{epoch:02d}-{step:05d}-{train_loss:.2f}",
        every_n_train_steps=args["train_args"]["save_freq_steps"],
        save_top_k=1,
    )

    # Save a checkpoint at the end of every epoch, delete previous one
    checkpoint_callback_epoch_end = ModelCheckpoint(
        dirpath=save_dir,
        filename="epoch-{epoch:02d}-{train_loss:.2f}",
        every_n_epochs=args["train_args"]["save_freq_epochs"],
        save_on_train_epoch_end=True,
        save_top_k=1,
        mode="max",
        monitor="epoch"
    )

    ckpt_callbacks = [checkpoint_callback_last, checkpoint_callback_epoch_end]

    if args["train_args"]["max_epochs"] % args["train_args"]["save_freq_epochs"] != 0:
        checkpoint_callback_train_end = ModelCheckpoint(
            dirpath=save_dir,
            filename="epoch-{epoch:02d}-{train_loss:.2f}",
            every_n_epochs=args["train_args"]["max_epochs"]
        )
        ckpt_callbacks.append(checkpoint_callback_train_end)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    strategy = DeepSpeedStrategy(
        **args["train_args"]["deepspeed_args"]
    )

    trainer = pl.Trainer(
        max_epochs=args["train_args"]["max_epochs"],
        accelerator=args["train_args"]["accelerator"],
        devices=args["train_args"]["devices"],
        strategy=strategy,
        precision=args["train_args"]["precision"],
        accumulate_grad_batches=calculate_accumulation_steps(args),
        gradient_clip_val=args["train_args"]["gradient_clipping"] if "gradient_clipping" in args["train_args"] else None,
        callbacks=ckpt_callbacks + [lr_monitor],
        log_every_n_steps=1,
        num_nodes=get_num_nodes(),
        logger=wandb_logger,
        use_distributed_sampler=not args["data"]["dataloader_args"]["length_grouped_sampler"]
    )

    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=resume_ckpt_path)


if __name__ == "__main__":
    main()
