# SLICEChat

Official implementation of [SLICEChat: Progressive In-Encoder Token Pruning for Whole-Slide Pathology Language Models](https://arxiv.org/abs/2609.24894).

This repository contains the training and inference code for SLICEChat. For training the whole-slide image encoder, see [SLICEChat-encoder](https://github.com/ali-kerem/SLICEChat-encoder).

## Requirements

- Linux with an NVIDIA CUDA-capable GPU
- Python 3.11
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management

## Installation

From the repository root:

```bash
uv sync
source .venv/bin/activate
```

The pinned packages target Python 3.11, PyTorch 2.9.1, and CUDA 12.6 on Linux x86_64. The PyTorch, torchvision, FlashAttention, Mamba, and causal-conv1d versions and wheel sources in `pyproject.toml` must be compatible with your hardware and with each other; adjust them together if needed.

## Dataset preparation

See [data/README.md](data/README.md) for the pinned annotation sources, exclusions and corrections, JSON manifest generation, and TRIDENT feature extraction and `.h5`-to-`.pt` conversion. The guide produces two training manifests and three benchmark manifests under `data/generated/`.

## VQA inference

Download the released checkpoints from [HuggingFace](https://huggingface.co/alike01b/SLICEChat):

```bash
hf download alike01b/SLICEChat --local-dir /path/to/SLICEChat
```

Choose the checkpoint for the model you want to run and pass its path as `--ckpt_dir`:

- SlideInstruct-trained model: `/path/to/SLICEChat/slideinstruct_ckpt/checkpoints/checkpoint`
- WSI-Bench-trained model: `/path/to/SLICEChat/wsibench_ckpt/checkpoints/checkpoint`

Keep the downloaded directory structure and configuration files intact.

Edit the other arguments in `scripts/vqa.sh`:

- `--manifest-path`: `data/generated/slidebench_vqa_tcga.json`, `data/generated/slidebench_vqa_bcnb.json`, or `data/generated/wsibench.json`.
- `--data_dir`: the benchmark's tensor root containing `features/` and `coords/`.
- `--output-path`: exact output JSON filename, for example `/path/to/results.json`.

Select the GPU with `CUDA_VISIBLE_DEVICES`, then run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/vqa.sh
```

The script saves each question, reference answer, and `model_response` directly to `--output-path`, creating parent directories if needed. An existing output file is overwritten.


## Training

Training uses PyTorch Lightning with DeepSpeed. Select the visible GPUs with `CUDA_VISIBLE_DEVICES` and set `train_args.devices` in the training configuration (`auto` uses all visible GPUs).

The scripts are editable run templates. Configure `scripts/configs/train.yaml` before starting a run. See [ARGUMENTS.md](ARGUMENTS.md) for the complete training configuration and release-script option reference. The main settings to edit are:

- `model.mllm_args.encoder_kwargs.path`: path to a pretrained encoder checkpoint file (`.pt` or `.pth`), not a directory. Follow the training instructions in [SLICEChat-encoder](https://github.com/ali-kerem/SLICEChat-encoder) to create it. Preserve the saved run's folder structure: keep `model_cfg.yaml` in the run directory and the checkpoint file under its `checkpoints/` subdirectory (for example, `<encoder-run>/checkpoints/epoch_20.pt`).
- `data.dataset_args.labels_path`: training manifest, such as `data/generated/slideinstruct_train.json` or `data/generated/wsibench_train.json`.
- `data.dataset_args.data_dir`: tensor root containing `features/` and `coords/`.
- `data.dataloader_args.batch_size`: per-GPU micro-batch size.
- `train_args.global_batch_size`: total samples per optimizer update, including all GPUs and gradient accumulation.
- `wandb.enabled` and `wandb.project_name`: whether and where to log to Weights & Biases. Authenticate your account before enabling it.

Gradient accumulation is calculated automatically. Choose a global batch size that is a positive multiple of `batch_size × total_gpus`.

If GPU memory is limited, lower the per-GPU batch size. You can also enable LoRA with `train_args.use_lora: true` and supply the adapter settings under `train_args.lora_args`. Alternatively, enable `train_args.deepspeed_args.offload_optimizer: true` to move optimizer state to CPU memory for higher memory efficiency but slower training.

The paper experiments used `data.dataloader_args.seed: 0` and `train_args.deterministic: true`, as in the encoder experiments. These settings improve reproducibility but do not guarantee identical results: Mamba and FlashAttention kernels can limit determinism.

Set `run_name` and the base `log_dir` at the top of `scripts/train.sh`, then launch:

```bash
bash scripts/train.sh
```

The first run may download the language model and tokenizer from Hugging Face. Provide network access or cache the required files beforehand.

Fresh runs create `<log_dir>/<timestamp>_<run_name>/`, containing `args.yaml`, `output.log`, and `checkpoints/`. Checkpoint frequency is controlled by `train_args.save_freq_epochs` and `train_args.save_freq_steps`.

### Starting from a SLICEChat checkpoint

To initialize a new training run from a complete SLICEChat checkpoint rather than the original encoder and language-model weights, set:

```yaml
model:
  ckpt_path: /path/to/run/checkpoints/epoch-end.ckpt

train_args:
  resume_from_checkpoint: false
```

Keep the original run's `args.yaml` above `checkpoints/`. The loader restores the saved architecture and model weights; the new run uses the current training settings and starts with fresh optimizer and trainer state. Set a new run name and log directory in `scripts/train.sh` and launch normally.

For weights-only loading and inference, the first load exports all model weights to `consolidated_fp32_model/` inside the checkpoint directory, in shards of up to 5 GB. Later loads reuse that export. The original DeepSpeed files are not deleted. Keep them for full training resume; after a successful export, you may manually remove them if only weights-only loading is needed.

### Resuming an interrupted run

To restore model, optimizer, and trainer state, pass the latest epoch-end checkpoint and enable resume:

```yaml
model:
  ckpt_path: /path/to/run/checkpoints/epoch-end.ckpt

train_args:
  resume_from_checkpoint: true
```

Use the same training launcher. The run name and training parameters are restored from the checkpoint's saved configuration, and logs go to the original run directory derived from `model.ckpt_path`.

Only epoch-end checkpoints are supported for full resume. Mid-epoch checkpoints cannot reliably restore the dataloader state.

The new log file has a `_resume` suffix: `output_resume.log`. If it already exists, another suffix is appended, such as `output_resume_resume.log`, so previous logs are preserved.

## Citation

```bibtex
@misc{bozkurt2026slicechatprogressiveinencodertoken,
      title={SLICEChat: Progressive In-Encoder Token Pruning for Whole-Slide Pathology Language Models},
      author={Ali Kerem Bozkurt and Baris Cem Bakay and Ibrahim Kulac and Cigdem Gunduz-Demir and Erkut Erdem and Aykut Erdem},
      year={2026},
      eprint={2609.24894},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.24894},
}
```
