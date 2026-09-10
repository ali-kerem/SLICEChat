from typing import Any, Dict

import torch
import pytorch_lightning as pl
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import get_cosine_schedule_with_warmup

from src.mllms import BaseMLLM


class MLLMLightningModule(pl.LightningModule):
    def __init__(self, 
                 model: BaseMLLM,
                 optimizer_args: Dict[str, Any] = None,
                 cpu_adam: bool = False):
        super().__init__()
        self.model = model
        self.optimizer_args = optimizer_args
        self.cpu_adam = cpu_adam
        self.training_losses_batch = []
        self.save_hyperparameters(ignore=['model'])

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self(batch)
        loss = outputs.loss
        self.training_losses_batch.append(loss.detach())
        return loss

    def on_train_epoch_start(self):
        """Update sampler epoch for proper shuffling across epochs."""
        # Access the train dataloader's sampler
        if self.trainer.train_dataloader is not None:
            sampler = self.trainer.train_dataloader.loaders.sampler if hasattr(self.trainer.train_dataloader, 'loaders') else self.trainer.train_dataloader.sampler
            
            # Call set_epoch if the sampler supports it (e.g., LengthGroupedDistributedSampler)
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(self.current_epoch)
                print(f"Set sampler epoch to {self.current_epoch}")

    def on_before_optimizer_step(self, optimizer):
        # This hook is called on every batch when using DeepSpeed, but we only want to log on an update step.
        is_update_step = (self.trainer.fit_loop.epoch_loop.batch_idx + 1) % self.trainer.accumulate_grad_batches == 0

        if is_update_step or self.trainer.is_last_batch:
            grad_norm = self.trainer.strategy.deepspeed_engine.get_global_grad_norm()
            if grad_norm is not None:
                self.log("grad_norm", grad_norm, prog_bar=True, on_step=True, on_epoch=False)
            
            if self.training_losses_batch:
                avg_loss = torch.stack(self.training_losses_batch).mean()
                self.log("train_loss", avg_loss, prog_bar=True, on_step=True, on_epoch=False)
                self.training_losses_batch.clear()

            # Log the learning rate to the progress bar
            self.log("lr", optimizer.param_groups[0]['lr'], prog_bar=True, on_step=True, on_epoch=False, logger=False)

    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer_args["learning_rate"] = float(self.optimizer_args["learning_rate"])
        self.optimizer_args["betas"] = tuple(self.optimizer_args["betas"])
        optimizer_type = self.optimizer_args["type"].lower()
        if optimizer_type not in {"adam", "adamw"}:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_args['type']}")

        if self.cpu_adam:
            optimizer = DeepSpeedCPUAdam(trainable_params, lr=self.optimizer_args["learning_rate"],
                                        weight_decay=self.optimizer_args["weight_decay"],
                                        betas=self.optimizer_args["betas"],
                                        adamw_mode=optimizer_type == "adamw")
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(trainable_params, lr=self.optimizer_args["learning_rate"],
                                                            weight_decay=self.optimizer_args["weight_decay"],
                                                            betas=self.optimizer_args["betas"])
        else:
            optimizer = torch.optim.AdamW(trainable_params, lr=self.optimizer_args["learning_rate"],
                                                            weight_decay=self.optimizer_args["weight_decay"],
                                                            betas=self.optimizer_args["betas"])
        
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.optimizer_args["warmup_ratio"] * total_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
