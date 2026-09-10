import torch
import torch.nn as nn

from .base_projector import BaseProjector

class Linear(BaseProjector):
    def __init__(self, encoder_dim: int, llm_dim: int, eps: float = 1e-6):
        super().__init__(encoder_dim, llm_dim)
        
        self.eps = eps

        self.projector = nn.Sequential(
            nn.RMSNorm(self.encoder_dim, eps=self.eps),
            nn.Linear(self.encoder_dim, self.llm_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)
