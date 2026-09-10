import inspect
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.configs import IGNORE_INDEX
from src.configs.special_tokens import SpecialTokenDefinition
from src.models.slide_encoders import SlideEncoderFactory

from .base_mllm import BaseMLLM
from .qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration


class Qwen2_5VLSlideMLLM(BaseMLLM):
    def __init__(
        self,
        llm_name_or_path: str,
        placeholder_tokens: Dict[str, int] = None,
        freeze_llm: bool = False,
        gradient_checkpointing: bool = False,
        freeze_encoder: bool = False,
        encoder_type: str = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        projector_type: str = None,
        qwen2_5_vl_mrope: Optional[Dict[str, Any]] = None,
        init_from_config: bool = False,
    ):
        nn.Module.__init__(self)

        self.llm_name_or_path = llm_name_or_path
        self.placeholder_tokens = placeholder_tokens or {}
        self.freeze_llm = freeze_llm
        self.freeze_encoder = freeze_encoder
        self.encoder_type = encoder_type
        self.encoder_kwargs = encoder_kwargs
        self.projector_type = projector_type

        self.qwen2_5_vl_mrope = {
            "enabled": True,
            "load_visual_tower": False,
            **(qwen2_5_vl_mrope or {}),
        }

        self.llm_config = Qwen2_5_VLConfig.from_pretrained(
            llm_name_or_path,
            local_files_only=os.path.exists(llm_name_or_path),
        )
        if getattr(self.llm_config, "model_type", None) != "qwen2_5_vl":
            raise ValueError(
                f"Qwen2_5VLSlideMLLM requires a qwen2_5_vl config, got {self.llm_config.model_type!r}."
            )
        self.llm_config.load_visual_tower = bool(self.qwen2_5_vl_mrope["load_visual_tower"])

        if init_from_config:
            self.llm = Qwen2_5_VLForConditionalGeneration._from_config(
                self.llm_config,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16,
            )
        else:
            self.llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                llm_name_or_path,
                config=self.llm_config,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16,
                local_files_only=os.path.exists(llm_name_or_path),
            )

        if gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()

        if self.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        self.hidden_size = self.llm_config.text_config.hidden_size
        self.add_token_type_ids = "token_type_ids" in inspect.signature(self.llm.forward).parameters

        self.modality_encoder = SlideEncoderFactory.create(
            encoder_type=self.encoder_type,
            init_from_config=init_from_config,
            **(self.encoder_kwargs or {}),
        )

        self._post_init_()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.llm.model.language_model.embed_tokens

    def _prepare_multimodal_inputs(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if padding_mask is None:
            padding_mask = (coords == -1).any(dim=-1).long()

        if len(features.shape) == 2:
            features = features.unsqueeze(0)
            coords = coords.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)

        if self.freeze_encoder:
            with torch.no_grad():
                slide_embeds, padding_mask, slide_positions = self.modality_encoder(
                    features,
                    coords,
                    padding_mask,
                    return_positions=True,
                )
        else:
            slide_embeds, padding_mask, slide_positions = self.modality_encoder(
                features,
                coords,
                padding_mask,
                return_positions=True,
            )

        if slide_positions.shape[:2] != slide_embeds.shape[:2]:
            raise ValueError(
                "Slide positions must align with slide embeddings after pruning/sorting. "
                f"Got positions {tuple(slide_positions.shape)} and embeds {tuple(slide_embeds.shape)}."
            )

        projected_slide_embeds = self.modality_projector(slide_embeds)
        # Preserve the token order returned by the slide encoder.
        return projected_slide_embeds, padding_mask, slide_positions

    def _slide_coords_to_mrope_ids(
        self,
        slide_positions: torch.Tensor,
        placeholder_position: int,
    ) -> torch.LongTensor:
        if slide_positions.numel() == 0:
            return torch.empty(3, 0, dtype=torch.long, device=slide_positions.device)

        if slide_positions.shape[-1] < 2:
            raise ValueError(f"Slide positions must have at least x/y coordinates, got {slide_positions.shape}.")

        xy = slide_positions[..., :2].float()
        if not torch.isfinite(xy).all():
            raise ValueError("Slide positions contain NaN or infinite values.")

        xy = xy - xy.min(dim=0, keepdim=True).values
        x = torch.round(xy[:, 0]).long()
        y = torch.round(xy[:, 1]).long()
        t = torch.zeros_like(x)  # A slide has a single temporal position: 0.

        return torch.stack([t, y, x], dim=0) + int(placeholder_position)

    def _build_text_positions(self, start: int, length: int, device: torch.device) -> torch.LongTensor:
        if length == 0:
            return torch.empty(4, 0, dtype=torch.long, device=device)
        pos = torch.arange(start, start + length, dtype=torch.long, device=device)
        return pos.view(1, -1).expand(4, -1)

    def _get_pad_token_id(self) -> int:
        pad_token_id = self.llm.generation_config.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.llm_config.text_config.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.llm_config.text_config.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0
        return int(pad_token_id)

    def _merge_slide_inputs(
        self,
        slide_embeds: torch.Tensor,
        slide_padding_mask: torch.Tensor,
        slide_positions: torch.Tensor,
        text_input_ids: List[torch.LongTensor],
        labels: Optional[List[torch.LongTensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        device = slide_embeds.device
        placeholder_token_id = self.placeholder_tokens[SpecialTokenDefinition.SLIDE_PLACEHOLDER.name]

        merged_embeds = []
        merged_position_ids = []
        merged_labels = []
        merged_lengths = []
        max_mrope_positions = []

        for batch_idx, input_ids in enumerate(text_input_ids):
            input_ids = input_ids.to(device)
            matches = (input_ids == placeholder_token_id).nonzero(as_tuple=True)[0]
            if len(matches) > 1:
                raise ValueError("Qwen2_5VLSlideMLLM currently supports one slide placeholder per sample.")

            if len(matches) == 0:
                text_embeds = self.get_input_embeddings()(input_ids)
                position_ids = self._build_text_positions(0, input_ids.shape[0], device)
                merged_embeds.append(text_embeds)
                merged_position_ids.append(position_ids)
                merged_lengths.append(input_ids.shape[0])
                max_mrope_positions.append(
                    position_ids[1:].max() if input_ids.numel() else torch.tensor(-1, device=device)
                )
                if labels is not None:
                    merged_labels.append(labels[batch_idx].to(device))
                continue

            placeholder_position = int(matches[0].item())
            valid_len = int((~slide_padding_mask[batch_idx].bool()).sum().item())
            if valid_len <= 0:
                raise ValueError("Slide placeholder was present but the slide encoder returned zero valid tokens.")

            before_ids = input_ids[:placeholder_position]
            after_ids = input_ids[placeholder_position + 1 :]
            before_embeds = self.get_input_embeddings()(before_ids)
            after_embeds = self.get_input_embeddings()(after_ids)
            valid_slide_embeds = slide_embeds[batch_idx, :valid_len]
            valid_slide_positions = slide_positions[batch_idx, :valid_len]

            slide_mrope_ids = self._slide_coords_to_mrope_ids(valid_slide_positions, placeholder_position)
            slide_text_positions = torch.arange(
                placeholder_position,
                placeholder_position + valid_len,
                dtype=torch.long,
                device=device,
            ).view(1, -1)
            slide_position_ids = torch.cat([slide_text_positions, slide_mrope_ids], dim=0)

            before_position_ids = self._build_text_positions(0, before_ids.shape[0], device)
            after_start = int(slide_mrope_ids.max().item()) + 1
            if after_ids.shape[0] > 0:
                after_text_positions = torch.arange(
                    placeholder_position + valid_len,
                    placeholder_position + valid_len + after_ids.shape[0],
                    dtype=torch.long,
                    device=device,
                ).view(1, -1)
                after_mrope_positions = torch.arange(
                    after_start,
                    after_start + after_ids.shape[0],
                    dtype=torch.long,
                    device=device,
                ).view(1, -1).expand(3, -1)
                after_position_ids = torch.cat([after_text_positions, after_mrope_positions], dim=0)
            else:
                after_position_ids = torch.empty(4, 0, dtype=torch.long, device=device)

            sample_embeds = torch.cat([before_embeds, valid_slide_embeds, after_embeds], dim=0)
            sample_position_ids = torch.cat([before_position_ids, slide_position_ids, after_position_ids], dim=1)

            merged_embeds.append(sample_embeds)
            merged_position_ids.append(sample_position_ids)
            merged_lengths.append(sample_embeds.shape[0])
            max_mrope_positions.append(sample_position_ids[1:].max())

            if labels is not None:
                label = labels[batch_idx].to(device)
                modality_labels = torch.full((valid_len,), IGNORE_INDEX, dtype=torch.long, device=device)
                merged_labels.append(
                    torch.cat([label[:placeholder_position], modality_labels, label[placeholder_position + 1 :]], dim=0)
                )

        max_len = max(merged_lengths)
        hidden_size = slide_embeds.shape[-1]
        padded_embeds = slide_embeds.new_zeros((len(merged_embeds), max_len, hidden_size))
        attention_mask = torch.zeros((len(merged_embeds), max_len), dtype=torch.long, device=device)
        position_ids = torch.ones((4, len(merged_embeds), max_len), dtype=torch.long, device=device)

        for batch_idx, (sample_embeds, sample_position_ids) in enumerate(zip(merged_embeds, merged_position_ids)):
            sample_len = sample_embeds.shape[0]
            padded_embeds[batch_idx, :sample_len] = sample_embeds
            attention_mask[batch_idx, :sample_len] = 1
            position_ids[:, batch_idx, :sample_len] = sample_position_ids

        return_dict = {
            "inputs_embeds": padded_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "rope_deltas": torch.stack(
                [(max_position.to(device=device) + 1 - max_len).long() for max_position in max_mrope_positions]
            ).view(-1, 1),
        }

        if labels is not None:
            return_dict["labels"] = pad_sequence(merged_labels, batch_first=True, padding_value=IGNORE_INDEX)

        return return_dict

    def forward(
        self,
        input_ids: List[torch.LongTensor],
        labels: Optional[List[torch.LongTensor]] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        modality_args, hf_kwargs = self._separate_kwargs(kwargs)

        if modality_args:
            slide_embeds, slide_padding_mask, slide_positions = self._prepare_multimodal_inputs(**modality_args)
            inputs = self._merge_slide_inputs(
                slide_embeds=slide_embeds,
                slide_padding_mask=slide_padding_mask,
                slide_positions=slide_positions,
                text_input_ids=input_ids,
                labels=labels,
            )
            self.llm.model.rope_deltas = inputs.pop("rope_deltas")
        else:
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self._get_pad_token_id())
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
            attention_mask.masked_fill_(input_ids == self._get_pad_token_id(), 0)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if labels is not None else None,
            }
            inputs = {key: value for key, value in inputs.items() if value is not None}

        return self.llm(**inputs, **hf_kwargs)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        **kwargs: Any,
    ) -> torch.LongTensor:
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)

        modality_args, hf_kwargs = self._separate_kwargs(kwargs)
        if modality_args:
            slide_embeds, slide_padding_mask, slide_positions = self._prepare_multimodal_inputs(**modality_args)
            merged_inputs = self._merge_slide_inputs(
                slide_embeds=slide_embeds,
                slide_padding_mask=slide_padding_mask,
                slide_positions=slide_positions,
                text_input_ids=[ids for ids in input_ids],
            )
            dummy_input_ids = torch.zeros(
                merged_inputs["attention_mask"].shape,
                dtype=torch.long,
                device=input_ids.device,
            )
            generated_ids = self.llm.generate(
                input_ids=dummy_input_ids,
                inputs_embeds=merged_inputs["inputs_embeds"],
                attention_mask=merged_inputs["attention_mask"],
                slide_position_ids=merged_inputs["position_ids"],
                slide_rope_deltas=merged_inputs["rope_deltas"],
                **hf_kwargs,
            )
            return generated_ids[:, dummy_input_ids.shape[1]:]

        return self.llm.generate(input_ids=input_ids, **hf_kwargs)
