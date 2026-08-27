"""UNI+MLP abundance head (no graph) — ablation vs BEACON GAT."""

from __future__ import annotations

import torch
import torch.nn as nn


class UniAbundanceMLP(nn.Module):
    """
    Spot-wise MLP on frozen UNI 1024-d features.

    Mirrors BEACON's post-GAT MLP capacity (128 → 32 → 1) with a linear
    projection from 1024 instead of GAT message passing.
    """

    def __init__(self, input_dim: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.net(x))
