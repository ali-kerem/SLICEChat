from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration as TransformersQwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel as TransformersQwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLTextModel,
    Qwen2_5_VisionTransformerPretrainedModel,
)


class Qwen2_5_VLSlideModel(TransformersQwen2_5_VLModel):
    def __init__(self, config):
        Qwen2_5_VLPreTrainedModel.__init__(self, config)
        self.visual = (
            Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
            if getattr(config, "load_visual_tower", True)
            else None
        )
        self.language_model = Qwen2_5_VLTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        if self.visual is None:
            raise ValueError("Qwen2.5-VL visual tower was not loaded for this slide-only model.")
        return super().get_image_features(pixel_values, image_grid_thw=image_grid_thw, **kwargs)


class Qwen2_5_VLForConditionalGeneration(TransformersQwen2_5_VLForConditionalGeneration):
    _keys_to_ignore_on_load_unexpected = [r"model\.visual\."]

    def __init__(self, config):
        Qwen2_5_VLPreTrainedModel.__init__(self, config)
        self.model = Qwen2_5_VLSlideModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        slide_position_ids: Optional[torch.LongTensor] = None,
        slide_rope_deltas: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )

        if slide_position_ids is not None and model_inputs.get("inputs_embeds") is not None:
            inputs_embeds = model_inputs["inputs_embeds"]
            expected_shape = (4, inputs_embeds.shape[0], inputs_embeds.shape[1])
            if tuple(slide_position_ids.shape) != expected_shape:
                raise ValueError(
                    "slide_position_ids must have shape "
                    f"{expected_shape} for the generation prefill step, got {tuple(slide_position_ids.shape)}."
                )

            model_inputs["position_ids"] = slide_position_ids.to(device=inputs_embeds.device)
            if slide_rope_deltas is not None:
                self.model.rope_deltas = slide_rope_deltas.to(device=inputs_embeds.device)

        return model_inputs

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs: Any,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        slide_position_ids = model_kwargs.pop("slide_position_ids", None)
        slide_rope_deltas = model_kwargs.pop("slide_rope_deltas", None)

        input_ids, model_kwargs = super()._expand_inputs_for_generation(
            expand_size=expand_size,
            is_encoder_decoder=is_encoder_decoder,
            input_ids=input_ids,
            **model_kwargs,
        )

        if slide_position_ids is not None:
            if expand_size != 1:
                slide_position_ids = slide_position_ids.repeat_interleave(expand_size, dim=1)
            model_kwargs["slide_position_ids"] = slide_position_ids

        if slide_rope_deltas is not None:
            if expand_size != 1:
                slide_rope_deltas = slide_rope_deltas.repeat_interleave(expand_size, dim=0)
            model_kwargs["slide_rope_deltas"] = slide_rope_deltas

        return input_ids, model_kwargs


__all__ = [
    "Qwen2_5_VLConfig",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5_VLSlideModel",
]
