from typing import Literal

from .base_projector import BaseProjector
from .mlp_gelu import MLP_GELU
from .linear import Linear


class ProjectorFactory:
    _projectors = {
        "mlp_gelu": MLP_GELU,
        "linear": Linear,
    }

    @staticmethod
    def create(
        projector_type: Literal["mlp_gelu", "linear"],
        **kwargs
    ) -> BaseProjector:
        """
        Factory method to create projectors
        
        Args:
            projector_type: Type of projector to create ("mlp_gelu" or "linear").
            **kwargs: Arguments for the projector. Can include both base class arguments and projector-specific arguments.
        
        Returns:
            BaseProjector: Instance of the specified projector
        
        Raises:
            ValueError: If projector_type is not supported
        """
        
        if projector_type not in ProjectorFactory._projectors:
            raise ValueError(f"Unsupported projector type: {projector_type}. Available projectors: {list(ProjectorFactory._projectors.keys())}")

        projector_class = ProjectorFactory._projectors[projector_type]
        return projector_class(**kwargs)
