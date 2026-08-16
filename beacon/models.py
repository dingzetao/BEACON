"""GAT–MLP abundance prediction head used by BEACON."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "BEACON requires torch-geometric. Install with: pip install torch-geometric"
    ) from e


class SpatialGATAbundanceHead(nn.Module):
    """Graph attention decoder that maps UNI patch embeddings to cell abundances."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 64,
        out_dim: int = 128,
        num_heads: int = 4,
        dropout1: float = 0.3,
        dropout2: float = 0.2,
    ):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout1)
        self.gat2 = GATConv(
            hidden_dim * num_heads, out_dim, heads=1, concat=False, dropout=dropout2
        )
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        return torch.relu(self.mlp(x))
