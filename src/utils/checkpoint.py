import os
import glob
from tempfile import TemporaryDirectory
from pathlib import Path
import torch
import yaml


def get_state_dict(ckpt_dir):
    if "consolidated_fp32_model" in os.listdir(ckpt_dir):
        consolidated_dir = f"{ckpt_dir}/consolidated_fp32_model"
        single_shard_path = os.path.join(consolidated_dir, "pytorch_model.bin")
        
        # Check for single shard first (pytorch_model.bin)
        if os.path.exists(single_shard_path):
            state_dict = torch.load(single_shard_path, map_location="cpu")
        else:
            # Multiple shards (pytorch_model-00001-of-00006.bin, etc.)
            state_dict = {}
            shards = sorted(glob.glob(f"{consolidated_dir}/pytorch_model-*.bin"))
            assert len(shards) > 0, f"No shards found in {consolidated_dir}"
            for shard in shards:
                sd = torch.load(shard, map_location="cpu")
                state_dict.update(sd)
    else:
        from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

        # ZeRO model-state files alone do not contain the full trainable weights.
        # Export all weights, including frozen parameters, without modifying the shards.
        tag = "checkpoint" if os.path.isdir(os.path.join(ckpt_dir, "checkpoint")) else None
        consolidated_dir = os.path.join(ckpt_dir, "consolidated_fp32_model")
        print(f"Exporting FP32 model weights to {consolidated_dir}")
        # Publish only a complete export, so an interrupted conversion is not reused.
        with TemporaryDirectory(prefix=".consolidating-", dir=ckpt_dir) as temp_dir:
            output_dir = os.path.join(temp_dir, "consolidated_fp32_model")
            convert_zero_checkpoint_to_fp32_state_dict(
                ckpt_dir, output_dir, tag=tag, exclude_frozen_parameters=False,
                max_shard_size="5GB", safe_serialization=False,
            )
            os.rename(output_dir, consolidated_dir)
        return get_state_dict(ckpt_dir)

    return {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }


def load_checkpoint(model, ckpt_path, *, strict=False):
    state_dict = get_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if len(unexpected) > 0:
        print(f"\033[1;31mWARNING: UNEXPECTED NUMBER OF KEYS WHEN LOADING CHECKPOINT:\033[0m {len(unexpected)}")
        unexpected_prefixes = set(k.split(".")[0] for k in unexpected)
        print(f"\033[38;5;208mUnexpected prefixes:\033[0m {unexpected_prefixes}")
    if len(missing) > 0:
        print(f"\033[38;5;208mWarning: Missing number of keys when loading checkpoint:\033[0m {len(missing)}")
        missing_prefixes = set(k.split(".")[0] for k in missing)
        print(f"\033[38;5;208mMissing prefixes:\033[0m {missing_prefixes}")


def get_checkpoint_config(ckpt_path):
    ckpt_config_dir = Path(ckpt_path).resolve().parent.parent
    with (ckpt_config_dir / "args.yaml").open("r") as f:
        ckpt_config = yaml.safe_load(f)
    encoder_kwargs = ckpt_config["model"]["mllm_args"].get("encoder_kwargs", {})
    encoder_path = encoder_kwargs.get("path")
    if encoder_path is not None and not Path(encoder_path).is_absolute():
        encoder_kwargs["path"] = str((ckpt_config_dir / encoder_path).resolve())
    return ckpt_config
