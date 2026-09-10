import torch
import torch.nn as nn

from .base_projector import BaseProjector

class MLP_GELU(BaseProjector):
    def __init__(self, encoder_dim: int, llm_dim: int):
        super().__init__(encoder_dim, llm_dim)
        
        self.projector = nn.Sequential(
            nn.Linear(self.encoder_dim, self.llm_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.llm_dim, self.llm_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)