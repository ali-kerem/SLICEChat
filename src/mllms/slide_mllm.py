from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

from src.configs.special_tokens import SpecialTokenDefinition
from src.models.slide_encoders import SlideEncoderFactory

from .base_mllm import BaseMLLM


class SlideMLLM(BaseMLLM):
    def __init__(self,
                 llm_name_or_path: str,
                 placeholder_tokens: Dict[str, int] = None,
                 freeze_llm: bool = False,
                 gradient_checkpointing: bool = False,
                 freeze_encoder: bool = False,
                 encoder_type: str = None,
                 encoder_kwargs: Optional[Dict[str, Any]] = None,
                 projector_type: str = None,
                 init_from_config: bool = False,
                ):
        
        super().__init__(llm_name_or_path=llm_name_or_path,
                         placeholder_tokens=placeholder_tokens,
                         freeze_llm=freeze_llm,
                         gradient_checkpointing=gradient_checkpointing,
                         freeze_encoder=freeze_encoder,
                         encoder_type=encoder_type,
                         encoder_kwargs=encoder_kwargs,
                         projector_type=projector_type,
                         init_from_config=init_from_config)

        self.modality_encoder = SlideEncoderFactory.create(
            encoder_type=self.encoder_type,
            init_from_config=init_from_config,
            **(self.encoder_kwargs or {})
        )

        self._post_init_()

    def _prepare_multimodal_inputs(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        
        if padding_mask is None:
            padding_mask = (coords == -1).any(dim=-1).long()
        
        if len(features.shape) == 2:
            features = features.unsqueeze(0)
            coords = coords.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)
        
        if self.freeze_encoder:
            with torch.no_grad():
                slide_embeds, padding_mask = self.modality_encoder(features, coords, padding_mask)
        else:
            slide_embeds, padding_mask = self.modality_encoder(features, coords, padding_mask)

        projected_slide_embeds = self.modality_projector(slide_embeds)

        return {self.placeholder_tokens[SpecialTokenDefinition.SLIDE_PLACEHOLDER.name]: (projected_slide_embeds, padding_mask)}
