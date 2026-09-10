from typing import Literal
from .base_mllm import BaseMLLM
from .slide_mllm import SlideMLLM


class MLLMFactory:
    _mllms = {
        "slide": SlideMLLM,
    }

    @staticmethod
    def create(
        mllm_type: Literal["slide"],
        **kwargs
    ) -> BaseMLLM:
        """
        Factory method to create mllms
        
        Args:
            mllm_type: Type of mllm to create, modality argument from the args ("slide").
            **kwargs: Arguments for the mllm. Can include both base class arguments and mllm-specific arguments.
        
        Returns:
            BaseMLLM: Instance of the specified mllm
        
        Raises:
            ValueError: If mllm_type is not supported
        """
        
        mllm_class = MLLMFactory._get_mllm_class(mllm_type, kwargs)
        return mllm_class(**kwargs)

    @staticmethod
    def from_pretrained(ckpt_path, placeholder_tokens, *, config=None) -> BaseMLLM:
        """Load a complete SLICEChat checkpoint without loading its upstream weights.

        Reads the run's args.yaml unless config is supplied. The original LLM
        config and encoder model_cfg.yaml must still be available.
        """
        from src.utils.checkpoint import get_checkpoint_config, load_checkpoint
        from src.utils.lora import apply_lora

        config = get_checkpoint_config(ckpt_path) if config is None else config
        model_args = dict(config["model"]["mllm_args"])
        model_args["init_from_config"] = True
        model = MLLMFactory.create(
            mllm_type=config["model"]["modality"],
            llm_name_or_path=config["model"]["llm_name_or_path"],
            placeholder_tokens=placeholder_tokens,
            **model_args,
        )
        if config.get("train_args", {}).get("use_lora", False) and not model.freeze_llm:
            apply_lora(model, config)
        load_checkpoint(model, ckpt_path, strict=True)
        return model

    @staticmethod
    def _get_mllm_class(mllm_type, kwargs):
        if mllm_type not in MLLMFactory._mllms:
            raise ValueError(f"Unsupported mllm type: {mllm_type}. Available mllms: {list(MLLMFactory._mllms.keys())}")

        qwen2_5_vl_enabled = (kwargs.get("qwen2_5_vl_mrope") or {}).get("enabled", False)

        if mllm_type == "slide" and qwen2_5_vl_enabled:
            from .qwen2_5_vl_slide_mllm import Qwen2_5VLSlideMLLM
            mllm_class = Qwen2_5VLSlideMLLM
            
        else:
            mllm_class = MLLMFactory._mllms[mllm_type]
            kwargs.pop("qwen2_5_vl_mrope", None)
        return mllm_class
    
