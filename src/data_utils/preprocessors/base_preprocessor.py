import os
from typing import List, Optional

import torch
from transformers import AutoTokenizer

from src.configs import IGNORE_INDEX, SpecialTokenDefinition


class BasePreprocessor():
    def __init__(self,
                 tokenizer_name_or_path: Optional[str] = None,
                 special_tokens: Optional[List[SpecialTokenDefinition]] = None,
                 use_image_start_end_tokens: bool = False,
                 remove_system_prompt: bool = False):
        
        special_tokens_dict = SpecialTokenDefinition.to_tokenizer_dict(special_tokens) if special_tokens is not None else {}

        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.use_image_start_end_tokens = use_image_start_end_tokens
        
        if tokenizer_name_or_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                extra_special_tokens=special_tokens_dict,
                trust_remote_code=True,
                local_files_only=os.path.exists(tokenizer_name_or_path),
                use_fast=True
            )
        else:
            self.tokenizer = None

        if remove_system_prompt:
            self._set_no_system_template()
        
        self._detect_generation_prompt_ids()

    def __call__(self, conv: List[dict], return_labels: bool = False):
        """
        Input:
        - conv: List[dict] (conversation)
        - return_labels: bool (whether to return the labels)
          Should be set to True for training, False for inference.
        """
        conv, modality_specific_kwargs = self.preprocess_modality(conv)

        if return_labels:
            assert conv[-1]["role"] == "assistant", "Conversation should end with assistant"
            
            input_ids = self._tokenize_conv(conv)
            labels = [IGNORE_INDEX] * len(input_ids)
            
            gen_prompt_ids = self.generation_prompt_ids
            gen_len = len(gen_prompt_ids)
            eos_id = self.tokenizer.eos_token_id
            
            i = 0
            while i <= len(input_ids) - gen_len:
                # Check for generation prompt pattern match
                if input_ids[i:i + gen_len] == gen_prompt_ids:
                    # Found the pattern - copy tokens from here until EOS (inclusive)
                    j = i + gen_len
                    while j < len(input_ids):
                        labels[j] = input_ids[j]
                        if input_ids[j] == eos_id:
                            j += 1
                            break
                        j += 1
                    i = j  # Continue search after this assistant response
                else:
                    i += 1
            
            return {
                "input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                **modality_specific_kwargs
            }
    
        else:  # Inference
            assert conv[-1]["role"] == "user", "Conversation should end with user"
            input_ids = self._tokenize_conv(conv, add_generation_prompt=True)
            return {
                "input_ids": torch.tensor(input_ids),
                **modality_specific_kwargs
            }

    def _tokenize_conv(self, conv: List[dict], add_generation_prompt: bool = False) -> List[int]:
        conv = self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)
        conv = conv.replace("<end_of_image><end_of_turn>", "<end_of_image>\n\n<end_of_turn>") # \n\n is ignored by the Gemma tokenizer when applying chat template
        conv = self.tokenizer(conv, add_special_tokens=False)["input_ids"]
        return conv
    
    # This function returns a dictionary of placeholder token names and their corresponding IDs that are defined at the
    # tokenizer initialization.
    def get_placeholder_token_name_to_ids(self):
        token_name_to_ids = {}
        for token_name, token_string in self.tokenizer._special_tokens_map.items():
            if not "placeholder" in token_name:
                continue
            token_id = self.tokenizer.convert_tokens_to_ids(token_string)
            token_name_to_ids[token_name] = token_id
            
        return token_name_to_ids

    def _detect_generation_prompt_ids(self):
        """Detect the generation prompt by comparing template output with/without it."""
        dummy_conv = [{"role": "user", "content": "test"}]
        
        text_without = self.tokenizer.apply_chat_template(
            dummy_conv, tokenize=False, add_generation_prompt=False
        )
        text_with = self.tokenizer.apply_chat_template(
            dummy_conv, tokenize=False, add_generation_prompt=True
        )
        
        # The generation prompt is the suffix difference
        assert text_with.startswith(text_without), "Generation prompt should be appended"
        generation_prompt_text = text_with[len(text_without):]
        
        # Tokenize the generation prompt pattern
        self.generation_prompt_ids = self.tokenizer(
            generation_prompt_text, add_special_tokens=False
        )["input_ids"]

    def _set_no_system_template(self):
        """
        Remove system message handling from the chat template.
        Supports Qwen, Gemma, Llama 2, and similar template formats.
        """
        import re
        template = self.tokenizer.chat_template

        # Test a dummy conversation (will be compared with the text without system prompt)
        dummy_conv = [{"role": "user", "content": "test"}, {"role": "assistant", "content": "test"}]
        tokens_with_sys_prompt = self.tokenizer.apply_chat_template(dummy_conv, return_dict=False)
        
        def find_and_remove_system_block(text, replacement=''):
            """
            Find if-else-endif blocks checking messages[0]['role'] == 'system'
            and replace them. Properly handles nested if blocks.
            """
            match = re.search(
                r"\{%-?\s*if\s+messages\[0\]\[.role.\]\s*==\s*.system.\s*-?%\}",
                text
            )
            if not match:
                return text, False
            
            start = match.start()
            pos = match.end()
            depth = 1
            
            while pos < len(text) and depth > 0:
                next_if = re.search(r"\{%-?\s*if\s", text[pos:])
                next_endif = re.search(r"\{%-?\s*endif\s*-?%\}", text[pos:])
                
                next_if_pos = pos + next_if.start() if next_if else float('inf')
                next_endif_pos = pos + next_endif.start() if next_endif else float('inf')
                
                if next_if_pos < next_endif_pos:
                    depth += 1
                    pos = pos + next_if.end()
                elif next_endif_pos < float('inf'):
                    depth -= 1
                    pos = pos + next_endif.end()
                else:
                    break
            
            if depth == 0:
                return text[:start] + replacement + text[pos:], True
            return text, False
        
        # Detect template style based on variable names used
        has_loop_messages = 'loop_messages' in template
        has_first_user_prefix = 'first_user_prefix' in template
        has_system_message = 'system_message' in template
        
        if has_loop_messages:
            # Variable-style templates that need loop_messages to be defined
            # Single replacement only (don't loop - our replacement contains a system check)
            
            if has_first_user_prefix:
                # Gemma-style: uses first_user_prefix for system content
                replacement = (
                    "{%- set first_user_prefix = '' -%}\n"
                    "    {%- if messages[0]['role'] == 'system' -%}\n"
                    "        {%- set loop_messages = messages[1:] -%}\n"
                    "    {%- else -%}\n"
                    "        {%- set loop_messages = messages -%}\n"
                    "    {%- endif -%}\n"
                )
            elif has_system_message:
                # Llama 2-style: uses system_message variable
                replacement = (
                    "{% set system_message = false %}"
                    "{% if messages[0]['role'] == 'system' %}"
                    "{% set loop_messages = messages[1:] %}"
                    "{% else %}"
                    "{% set loop_messages = messages %}"
                    "{% endif %}"
                )
            else:
                # Unknown loop_messages style, try generic
                replacement = (
                    "{% if messages[0]['role'] == 'system' %}"
                    "{% set loop_messages = messages[1:] %}"
                    "{% else %}"
                    "{% set loop_messages = messages %}"
                    "{% endif %}"
                )
            
            template, _ = find_and_remove_system_block(template, replacement)
        else:
            # Output-style (Qwen): loop to remove all system blocks
            replacement = ''
            modified = True
            while modified:
                template, modified = find_and_remove_system_block(template, replacement)
        
        self.tokenizer.chat_template = template

        # Compare the template output with/without system prompt
        tokens_without_sys_prompt = self.tokenizer.apply_chat_template(dummy_conv, return_dict=False)
        # Sanity check: 
        # 1. We expect the text to be at least 5 tokens (2 for "test"s, 1 for assistant, 1 for user, 1 for eos)
        # 2. We expect the text to be a suffix of the text with system prompt
        if not (len(tokens_without_sys_prompt) >= 5 and tokens_with_sys_prompt[-len(tokens_without_sys_prompt):] == tokens_without_sys_prompt):
            raise ValueError("System prompt removal failed. Please add compatibility for the tokenizer or pass remove_system_prompt=False.")
