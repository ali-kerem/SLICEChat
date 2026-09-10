from typing import Literal

from .base_collator import BaseCollator
from .slide_collator import SlideCollator

class CollatorFactory:
    _collators = {
        "slide": SlideCollator,
    }

    @staticmethod
    def create(
        modality: Literal["slide"],
    ) -> BaseCollator:
        """
        Factory method to create collators
        
        Args:
            modality: Type of modality to create, modality argument from the args ("slide").
        
        Returns:
            BaseCollator: Instance of the specified collator    
        Raises:
            ValueError: If modality is not supported
        """
        if modality not in CollatorFactory._collators:
            raise ValueError(f"Unknown modality: {modality}. Available modalities: {list(CollatorFactory._collators.keys())}")

        collator_class = CollatorFactory._collators[modality]
        return collator_class()
