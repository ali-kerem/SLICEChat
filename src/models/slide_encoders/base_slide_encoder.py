from typing import Tuple
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseSlideEncoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the slide encoder
        Args:
            features: Tensor of shape [B, N, D] where:
                    B is batch size
                    D is the embedding dimension
                    N is number of features
            coords: Tensor of shape [B, N, 2] where:
                    B is batch size
                    N is number of features
                    x, y are coordinates of the feature map
        Returns:
            embeddings: Tensor of shape [B, N, embedding_dim] where N is number of features
            padding_mask: Tensor of shape [B, N] where N is number of features
        """
        pass
