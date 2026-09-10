# Argument reference

This page describes the training YAML and the release scripts' command-line options.

Values labeled **Template** come from `scripts/configs/train.yaml`. Keep the template's required sections and edit their values.

## Model

Keys in this table are relative to `model`:

| Key | Meaning | Template |
| --- | --- | --- |
| `llm_name_or_path` | Hugging Face model ID or local model directory, also used to select the tokenizer. | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `modality` | Model/preprocessor family. The supported release option is `slide`. | `slide` |
| `ckpt_path` | Complete SLICEChat checkpoint directory for weights-only initialization or full resume; not an encoder checkpoint file. | Omitted |

Keys below are relative to `model.mllm_args`:

| Key | Meaning | Template |
| --- | --- | --- |
| `encoder_type` | Slide encoder implementation. Only `hybrid` is registered. | `hybrid` |
| `encoder_kwargs.path` | Encoder `.pt`/`.pth` checkpoint file. Keep `<encoder-run>/model_cfg.yaml` above its `checkpoints/` directory. A YAML configuration path instead creates a randomly initialized encoder for fresh training. | Set your encoder path |
| `freeze_llm` | Disables language-model gradients and keeps it in evaluation mode during training. | `false` |
| `freeze_encoder` | Disables encoder gradients and keeps it in evaluation mode during training. The projector remains trainable. | `true` |
| `projector_type` | `mlp_gelu`: two linear layers with GELU between them. `linear`: RMS normalization followed by one linear layer. | `mlp_gelu` |
| `gradient_checkpointing` | Enables language-model activation checkpointing to reduce activation memory at the cost of extra computation. | `true` |
| `qwen2_5_vl_mrope.enabled` | Selects the Qwen2.5-VL slide implementation, which uses slide coordinates for multimodal rotary positional embeddings. Requires a Qwen2.5-VL configuration. | `true` |
| `qwen2_5_vl_mrope.load_visual_tower` | Whether to instantiate Qwen's original image tower. Leave disabled for the pre-extracted slide-feature workflow. | `false` |

When mRoPE is disabled, the factory selects the generic slide model backed by `AutoModelForCausalLM`; the language-model source must support that class. Qwen slide positions use rounded, minimum-shifted x/y coordinates, temporal position zero.

Encoder architecture and pruning settings come from its saved `model_cfg.yaml`. Checkpoint-based SLICEChat loading constructs the models from configuration before restoring weights, so those configuration files must remain available.

## Data

### Preprocessing

Keys are relative to `data.preprocessor_args`:

| Key | Meaning | Template |
| --- | --- | --- |
| `use_image_start_end_tokens` | Wraps the slide placeholder in the tokenizer's vision boundary tokens. For Qwen these are `<\|vision_start\|>` and `<\|vision_end\|>`. | `true` |
| `remove_system_prompt` | Modifies the tokenizer's chat template to remove system-message handling. | `false` |

### Dataset

Keys are relative to `data.dataset_args`:

| Key | Meaning | Template |
| --- | --- | --- |
| `dataset_name` | Dataset implementation. Only `wsi` is supported. | `wsi` |
| `labels_path` | Training JSON manifest containing filename stems and conversations. | Set your generated training manifest |
| `data_dir` | Root containing paired `features/<filename>.pt` and `coords/<filename>.pt`. | Set your tensor root |
| `token_compression_rate` | Estimated fraction of slide tokens removed, used only to estimate lengths for sampling. `0.8` estimates that 20% remain; this does not configure actual encoder pruning. | `0.8` |
| `system_prompt` | Optional system-message text prepended to each conversation. `null` inserts no explicit system message; the tokenizer template may still supply its default. | `null` |

### Dataloader

Keys are relative to `data.dataloader_args`:

| Key | Meaning | Template |
| --- | --- | --- |
| `batch_size` | Per-GPU micro-batch size. | `4` |
| `num_workers` | Data-loading subprocesses per training process. `0` loads in-process; workers persist across epochs when this is positive. | `4` |
| `shuffle` | Enables shuffling in the active sampler or dataloader. Length grouping still sorts by estimated length when this is false. | `true` |
| `length_grouped_sampler` | Groups similarly sized examples across ranks to reduce padding; computes estimated lengths during setup. The custom sampler drops the incomplete final global micro-batch. | `true` |
| `seed` | Seed for training RNGs, dataloader workers, and sampling. The custom sampler adds the epoch to its seed. | `0` |

## Training and DeepSpeed

Keys are relative to `train_args`:

| Key | Meaning | Template |
| --- | --- | --- |
| `resume_from_checkpoint` | Restores full training state and saved configuration rather than starting a new run. Requires `model.ckpt_path`. | `false` |
| `deterministic` | Seeds reproducibility-related settings, disables cuDNN benchmarking, and requests deterministic algorithms with warnings instead of errors. Fused kernels can still limit determinism. | `true` |
| `max_epochs` | Total training epochs, not additional epochs after resume. | `3` |
| `global_batch_size` | Target samples per optimizer update across all GPUs and accumulated micro-batches. | `64` |
| `gradient_clipping` | Gradient clipping value passed to Lightning/DeepSpeed. | `1.0` |
| `save_freq_epochs` | Epoch-end checkpoint interval in epochs. | `1` |
| `save_freq_steps` | Periodic checkpoint interval in optimizer steps, not individual micro-batches. | `1000` |
| `precision` | Precision mode passed to Lightning's DeepSpeed integration. The model loaders themselves initialize the LLM in BF16. | `bf16-mixed` |
| `accelerator` | Lightning accelerator. The provided model and encoder workflow requires CUDA GPUs. | `gpu` |
| `devices` | `auto`, an integer GPU count, or a list of visible GPU indices. | `auto` |
| `use_tensor_cores` | Calls `torch.set_float32_matmul_precision("medium")` for float32 matrix multiplications; does not replace the `precision` setting. | `true` |
| `deepspeed_args.stage` | DeepSpeed ZeRO stage; the template uses stage 3 to partition parameters, gradients, and optimizer state. | `3` |
| `deepspeed_args.offload_optimizer` | Enables CPU optimizer offload through `DeepSpeedCPUAdam`, supporting both `optimizer.type: Adam` and `AdamW`. | `false` |

`deepspeed_args` is passed to Lightning's standard `DeepSpeedStrategy`. DeepSpeed controls mixed-precision execution; do not assume that its `bf16-mixed` behavior is identical to ordinary Lightning autocast without DeepSpeed, or that changing this option alone changes model-loading dtypes.

Accumulation is computed as:

```text
global_batch_size // (batch_size × total_gpus)
```

Choose an exact positive multiple to avoid rounding down the effective batch size. For example, `64 // (4 × 4) = 4` micro-batches per optimizer update. The node count is read from `SLURM_NNODES` when present, otherwise it is `1`.

Periodic and epoch-end checkpoints use separate callbacks, each configured with `save_top_k=1`; epoch-end selection keeps the latest epoch. A final-epoch callback is added if `max_epochs` is not divisible by `save_freq_epochs`. Checkpoints include frozen model parameters. Only epoch-end checkpoints are supported for full training resume.

### Checkpoint configuration precedence

- With no `model.ckpt_path`, training initializes the language model and encoder from their configured sources.
- With `model.ckpt_path` and `train_args.resume_from_checkpoint: false`, the script restores architecture and preprocessing settings from the saved `args.yaml`, loads model weights, and starts a new optimizer and scheduler. Current training settings and data paths are retained.
- With `train_args.resume_from_checkpoint: true`, the saved configuration is restored, including the run name and training settings. The log directory is derived from the checkpoint path, not the shell's new run name. Full resume requires the DeepSpeed training state and an epoch-end checkpoint.

Keep the saved run structure: `<run>/args.yaml` and `<run>/checkpoints/<checkpoint-directory>/`. See the README for [checkpoint loading](README.md#starting-from-a-slicechat-checkpoint) and [resume limitations](README.md#resuming-an-interrupted-run).

## Optimizer and scheduler

Keys are relative to `optimizer`:

| Key | Meaning | Template |
| --- | --- | --- |
| `type` | `Adam` or `AdamW` (case-insensitive). | `AdamW` |
| `learning_rate` | Optimizer learning rate. | `2e-5` |
| `weight_decay` | Weight decay for the selected optimizer, including CPU offload. | `0.0` |
| `betas` | Adam momentum coefficients, including CPU offload. | `[0.9, 0.999]` |
| `warmup_ratio` | Fraction of estimated optimizer steps used for learning-rate warmup. | `0.03` |

The scheduler is always cosine decay with warmup. Total steps come from Lightning's `estimated_stepping_batches`; warmup steps are `int(warmup_ratio × total_steps)`.

## Optional LoRA

These optional keys are under `train_args` and are omitted from the template:

| Key | Meaning | Default |
| --- | --- | --- |
| `use_lora` | Wraps the language model with PEFT adapters when `model.mllm_args.freeze_llm` is false. | `false` |
| `lora_args` | Arguments passed to PEFT's `LoraConfig`; required when LoRA is enabled. The task type is set to `CAUSAL_LM` in code. | No configuration supplied |

For example, merge this into the existing `train_args` section and keep `model.mllm_args.freeze_llm: false`:

```yaml
train_args:
  use_lora: true
  lora_args:
    r: 8
    lora_alpha: 16
    lora_dropout: 0.05
    target_modules: [q_proj, v_proj]
```

Here, `r` is the adapter rank, `lora_alpha` controls adapter scaling, `lora_dropout` is adapter dropout, and `target_modules` selects layers by name.

## Weights & Biases

| Key | Meaning | Template |
| --- | --- | --- |
| `wandb.enabled` | Enables the W&B logger and configuration logging. | `false` |
| `wandb.project_name` | W&B project for the run. The run name comes from the launcher or resumed configuration. | `patch-mllm-train` |
