from typing import Literal

from .base_slide_encoder import BaseSlideEncoder
from .hybrid_slide_encoder import HybridSlideEncoder


class SlideEncoderFactory:
    _encoders = {
        "hybrid": HybridSlideEncoder,
    }

    @staticmethod
    def create(
        encoder_type: Literal["hybrid"],
        **kwargs
    ) -> BaseSlideEncoder:
        """
        Factory method to create slide encoders
        
        Args:
            encoder_type: Type of encoder to create ("hybrid").
            **kwargs: Arguments for the encoder. Can include both base class arguments and encoder-specific arguments.
        
        Returns:
            BaseSlideEncoder: Instance of the specified slide encoder
        
        Raises:
            ValueError: If encoder_type is not supported
        """
        
        if encoder_type not in SlideEncoderFactory._encoders:
            raise ValueError(f"Unsupported encoder type: {encoder_type}. Available encoders: {list(SlideEncoderFactory._encoders.keys())}")

        encoder_class = SlideEncoderFactory._encoders[encoder_type]
        return encoder_class(**kwargs)
