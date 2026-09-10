from typing import List, Optional
import warnings

from src.configs import SpecialTokenDefinition

from .base_preprocessor import BasePreprocessor


class SlideMLLMPreprocessor(BasePreprocessor):
    def __init__(self,
                 tokenizer_name_or_path: Optional[str] = None,
                 **kwargs
                 ):
        self.placeholder_token = SpecialTokenDefinition.SLIDE_PLACEHOLDER

        super().__init__(tokenizer_name_or_path=tokenizer_name_or_path,
                         special_tokens=[self.placeholder_token],
                         **kwargs)

    def preprocess_modality(self, conv: List[dict]):
        modality_specific_kwargs = {}

        for idx, turn in enumerate(conv):
            if turn["role"] == "user":
                if isinstance(turn["content"], str):
                    continue
                has_slide = any(d.get("type") in {"features", "coords"} for d in turn["content"])
                if has_slide:
                    features_idx = next((i for i, d in enumerate(turn["content"]) if d.get("type") in {"features"}), -1)
                    coords_idx = next((i for i, d in enumerate(turn["content"]) if d.get("type") in {"coords"}), -1)

                    features = turn["content"][features_idx]["features"]
                    coords = turn["content"][coords_idx]["coords"]

                    text_idx = next((i for i, d in enumerate(turn["content"]) if d.get("type") == "text"), -1)

                    assert text_idx != -1, "No text data in the user prompt" 

                    slide_tokens = self.placeholder_token.string

                    if self.use_image_start_end_tokens:
                        additional_special_tokens = self.tokenizer.special_tokens_map.get("additional_special_tokens", [])
                        if "<|vision_start|>" in additional_special_tokens and "<|vision_end|>" in additional_special_tokens:
                            slide_tokens = "<|vision_start|>" + slide_tokens + "<|vision_end|>"
                        elif "boi_token" in self.tokenizer.init_kwargs and "eoi_token" in self.tokenizer.init_kwargs:
                            slide_tokens = self.tokenizer.init_kwargs["boi_token"] + slide_tokens + self.tokenizer.init_kwargs["eoi_token"]
                        else:
                            warnings.warn("You are using image start and end tokens for slides, but they are not defined in the tokenizer.init_kwargs, so we will use <start_of_image> and <end_of_image> instead.")
                            slide_tokens = "<start_of_image>" + slide_tokens + "<end_of_image>"

                    text_with_slide_tokens = slide_tokens + "\n\n" + turn["content"][text_idx]["text"]

                    conv[idx]["content"] = text_with_slide_tokens

                    modality_specific_kwargs = {
                        "features" : features,
                        "coords" : coords
                    }

                    break
        
        return conv, modality_specific_kwargs
