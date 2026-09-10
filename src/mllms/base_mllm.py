import os
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union, Optional, List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.configs import IGNORE_INDEX
from src.models.projectors import ProjectorFactory


class BaseMLLM(nn.Module, ABC):
    def __init__(self,
                 llm_name_or_path: str,
                 placeholder_tokens: Dict[str, int], # e.g., {"<slide_placeholder>": 32001}
                 freeze_llm: bool = False,
                 gradient_checkpointing: bool = False,
                 freeze_encoder: bool = False,
                 encoder_type: str = None,
                 encoder_kwargs: Optional[Dict[str, Any]] = None,
                 projector_type: str = None,
                 init_from_config: bool = False,
                 ):
        super().__init__()
        self.llm_name_or_path = llm_name_or_path
        self.placeholder_tokens = placeholder_tokens
        self.llm_config = AutoConfig.from_pretrained(llm_name_or_path)

        self.freeze_llm = freeze_llm
        self.freeze_encoder = freeze_encoder
        self.encoder_type = encoder_type
        self.encoder_kwargs = encoder_kwargs
        self.projector_type = projector_type

        if init_from_config:
            self.llm = AutoModelForCausalLM.from_config(
                self.llm_config,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16,
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
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

        try:
            self.hidden_size = self.llm_config.hidden_size
        except:
            self.hidden_size = self.llm_config.text_config.hidden_size # For some models like Gemma etc.

        self.add_token_type_ids = 'token_type_ids' in inspect.signature(self.llm.forward).parameters

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.modality_encoder.eval()
        if self.freeze_llm:
            self.llm.eval()

    def _post_init_(self):
        if self.freeze_encoder:
            for param in self.modality_encoder.parameters():
                param.requires_grad = False
        
        self.modality_projector = ProjectorFactory.create(
            projector_type=self.projector_type,
            encoder_dim=self.modality_encoder.embed_dim,
            llm_dim=self.hidden_size
        )

        self._define_modality_args()

    def _define_modality_args(self):
        """Automatically extract required and optional modality arguments from _prepare_multimodal_inputs signature."""
        sig = inspect.signature(self._prepare_multimodal_inputs)
        self.required_modality_args = []
        self.optional_modality_args = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            if param.default == inspect.Parameter.empty:
                self.required_modality_args.append(param_name)
            else:
                self.optional_modality_args.append(param_name)

    def _separate_kwargs(self, kwargs: Any):
        modality_args = {arg: kwargs[arg] for arg in kwargs if arg in self.required_modality_args or arg in self.optional_modality_args}
        hf_kwargs = {arg: kwargs[arg] for arg in kwargs if arg not in modality_args}

        if modality_args:
            if not all(arg in modality_args for arg in self.required_modality_args):
                raise ValueError(f"Missing required modality arguments: {[arg for arg in self.required_modality_args if arg not in kwargs]}")

        return modality_args, hf_kwargs

    def get_input_embeddings(self) -> nn.Embedding:
        """Provides access to the LLM's input token embeddings."""
        return self.llm.get_input_embeddings()

    @abstractmethod
    def _prepare_multimodal_inputs(
        self,
        **modality_specific_kwargs: Any
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Subclasses MUST implement this method.
        It is responsible for:
        1. Taking raw or preprocessed modality inputs from `modality_specific_kwargs`.
        2. Processing them through any modality-specific encoders (if owned by the subclass).
        3. Projecting the modality features to the LLM's hidden dimension.
        4. Returning a dictionary with keys as the placeholder token ids and values as the projected modality features 
            and padding mask.

        Args:
            **modality_specific_kwargs: Keyword arguments containing the inputs for each modality
                                       (e.g., patches, features, coords).

        Returns:
            Dict[str, Tuple[torch.Tensor, torch.Tensor]]: A dictionary with keys as the placeholder token ids and values
            as the projected modality features and padding mask (e.g., {"<slide_placeholder>": (slide_embeds, padding_mask)}).
        """
        raise NotImplementedError("Subclasses must implement `_prepare_multimodal_inputs`")
    
    def _merge_multimodal_inputs(self,
                                 modality_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                                 text_input_ids: List[torch.LongTensor],
                                 labels: Optional[List[torch.LongTensor]] = None,
                                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get the placeholder token id and the modality input embeds
        placeholder_token_id = next(iter(modality_dict.keys()))
        modality_input_embeds, modality_padding_mask = modality_dict[placeholder_token_id]

        device = modality_input_embeds.device
        B = len(text_input_ids)

        placeholder_token_positions = []
        text_input_lens = []
        modality_input_lens = []

        for i, input_ids in enumerate(text_input_ids):
            matches = (input_ids == placeholder_token_id).nonzero(as_tuple=True)[0]
            if len(matches) == 0:
                placeholder_token_positions.append(-1)
                text_input_lens.append(input_ids.shape[0])
                modality_input_lens.append(0)  # No modality tokens for text-only samples
            else:
                placeholder_token_positions.append(matches[0].item())
                text_input_lens.append(input_ids.shape[0] - 1)
                modality_input_lens.append((~(modality_padding_mask[i].bool())).sum().item())

        placeholder_token_positions = torch.tensor(placeholder_token_positions, dtype=torch.long, device=device)
        text_input_lens = torch.tensor(text_input_lens, dtype=torch.long, device=device)
        modality_input_lens = torch.tensor(modality_input_lens, dtype=torch.long, device=device)

        input_ids_lens = text_input_lens + modality_input_lens # [B], length of each input in the batch
        max_input_ids_len = int(torch.max(input_ids_lens).item()) # max length of inputs in the batch

        if self.add_token_type_ids:
            token_type_ids_list = []

        inputs_embeds = []

        # Concatenating text embeds, modality embeds, and padding embeds.
        for i in range(B):
            if modality_input_lens[i] > 0:
                seq_len = input_ids_lens[i]
                modality_len = modality_input_lens[i]
                placeholder_position = placeholder_token_positions[i]

                pad_len = max_input_ids_len - seq_len
                pad_ids = torch.full((pad_len,), self.llm.generation_config.pad_token_id, dtype=torch.long, device=device)

                input_ids = torch.cat([text_input_ids[i][:placeholder_position], 
                                        text_input_ids[i][placeholder_position + 1:],
                                        pad_ids], dim=0)
                
                text_embeds = self.get_input_embeddings()(input_ids)
                text_embeds_before = text_embeds[:placeholder_position] # Before the placeholder token
                text_embeds_after = text_embeds[placeholder_position:] # After the placeholder token
                modality_embeds = modality_input_embeds[i][:modality_len] # Cut out the padding

                inputs_embeds.append(torch.cat([text_embeds_before, modality_embeds, text_embeds_after], dim=0))

                if labels is not None:
                    label = labels[i]
                    
                    # Split the label tensor at the placeholder position
                    label_before = label[:placeholder_position]
                    label_after = label[placeholder_position + 1:] # Skip the original placeholder_token label

                    # Create ignore labels for the modality part
                    modality_labels = torch.full((modality_len,), IGNORE_INDEX, dtype=torch.long, device=device)

                    # Reconstruct the labels tensor with modality labels inserted
                    labels[i] = torch.cat([label_before, modality_labels, label_after], dim=0)
            
                if self.add_token_type_ids:
                    token_type_ids = torch.zeros(max_input_ids_len, device=device)
                    token_type_ids[placeholder_position : placeholder_position + modality_len] = 1 # Fill the modality embeds indices with 1
                    token_type_ids_list.append(token_type_ids)
            else:
                seq_len = input_ids_lens[i]
                pad_len = max_input_ids_len - seq_len
                pad_ids = torch.full((pad_len,), self.llm.generation_config.pad_token_id, dtype=torch.long, device=device)
                input_ids = torch.cat([text_input_ids[i], pad_ids], dim=0)
                
                inputs_embeds.append(self.get_input_embeddings()(input_ids))

        inputs_embeds = torch.stack(inputs_embeds)

        attention_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long, device=device)
        attention_mask.masked_fill_(torch.arange(max_input_ids_len, device=device) >= input_ids_lens[:, None], 0)

        return_dict = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }

        if labels is not None:
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            return_dict["labels"] = labels

        if self.add_token_type_ids:
            token_type_ids = torch.stack(token_type_ids_list)
            return_dict["token_type_ids"] = token_type_ids

        return return_dict


    def forward(
        self,
        input_ids: List[torch.LongTensor],
        labels: Optional[List[torch.LongTensor]] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        modality_args, hf_kwargs = self._separate_kwargs(kwargs)

        if modality_args:
            modality_dict = self._prepare_multimodal_inputs(**modality_args)
            # This gives a dict of placeholder_token_id -> modality_input_embeds, modality_padding_mask

            inputs = self._merge_multimodal_inputs(
                modality_dict=modality_dict,
                text_input_ids=input_ids,
                labels=labels
            )
        else:
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.llm.generation_config.pad_token_id)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
            attention_mask.masked_fill_(input_ids == self.llm.generation_config.pad_token_id, 0)
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

        return self.llm(
            **inputs,
            **hf_kwargs
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        **kwargs: Any # Contains both modality inputs and HuggingFace generate() kwargs
    ) -> torch.LongTensor:

        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)

        modality_args, hf_kwargs = self._separate_kwargs(kwargs)

        if modality_args:            
            modality_dict = self._prepare_multimodal_inputs(**modality_args)

            merged_inputs = self._merge_multimodal_inputs(
                modality_dict=modality_dict,
                text_input_ids=input_ids,
            )

            # We need to pass this to bypass the length validation in GenerationMixin.generate
            input_ids = torch.zeros(merged_inputs["attention_mask"].shape, dtype=torch.long, device=input_ids.device)
        else:
            merged_inputs = {}

        return self.llm.generate(
            input_ids=input_ids,
            **merged_inputs,
            **hf_kwargs
        )
