import yaml
from pathlib import Path
from typing import Tuple

import torch

from slicechat_encoder.models.utils.factory import make_encoder_from_ckpt, make_encoder_from_config

from .base_slide_encoder import BaseSlideEncoder


class HybridSlideEncoder(BaseSlideEncoder):
    def __init__(self, path: str, init_from_config: bool = False):
        super().__init__()

        path = Path(path)
        is_checkpoint = path.suffix in {".pt", ".pth"}
        if is_checkpoint and not init_from_config:
            self.model = make_encoder_from_ckpt(str(path), "cuda")
        else:
            # Match slicechat_encoder.make_encoder_from_ckpt's config lookup.
            config_path = path.parent.parent / "model_cfg.yaml" if is_checkpoint else path
            with config_path.open("r") as f:
                config = yaml.safe_load(f)
            if "vision_cfg" in config:
                encoder_config = config["vision_cfg"]
            elif "encoder_config" in config: # For MAE-only trained checkpoint
                encoder_config = config["encoder_config"]
            else:
                raise ValueError(f"Expected vision_cfg or encoder_config in {config_path}")
            self.model = make_encoder_from_config(encoder_config)

        self.embed_dim = self.model.embed_dim
    
    def forward(self, features: torch.Tensor, coords: torch.Tensor, padding_mask: torch.Tensor, return_positions: bool = False, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.model(token_embeddings=features, positions=coords, encoder_padding_mask=padding_mask, return_pooled_output=False)

        if return_positions:
            return out["encoder_out"], out["padding_mask"], out["positions"]

        return out["encoder_out"], out["padding_mask"]
