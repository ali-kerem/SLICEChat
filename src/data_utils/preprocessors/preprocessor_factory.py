from typing import Literal

from .base_preprocessor import BasePreprocessor
from .slide_mllm_preprocessor import SlideMLLMPreprocessor


class PreprocessorFactory:
    _preprocessors = {
        "slide": SlideMLLMPreprocessor,
    }

    @staticmethod
    def create(
        preprocessor_type: Literal["slide"],
        **kwargs
    ) -> BasePreprocessor:
        """
        Factory method to create preprocessors
        
        Args:
            preprocessor_type: Type of preprocessor to create, modality argument from the args ("slide").
            **kwargs: Arguments for the preprocessor. Can include both base class arguments and preprocessor-specific arguments.
        
        Returns:
            BasePreprocessor: Instance of the specified preprocessor
        
        Raises:
            ValueError: If preprocessor_type is not supported
        """
        if preprocessor_type not in PreprocessorFactory._preprocessors:
            raise ValueError(f"Unsupported preprocessor type: {preprocessor_type}. Available preprocessors: {list(PreprocessorFactory._preprocessors.keys())}")

        preprocessor_class = PreprocessorFactory._preprocessors[preprocessor_type]
        return preprocessor_class(**kwargs)
