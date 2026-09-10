from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseProjector(nn.Module, ABC):
    def __init__(self, encoder_dim: int, llm_dim: int):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the projector
        Args:
            x: Tensor of shape [B, N, D] where:
                    B is batch size
                    N is number of features
                    D is the embedding dimension of the encoder
        Returns:
            x: Tensor of shape [B, N, D] where D is the embedding dimension of the LLM
        """
        pass
