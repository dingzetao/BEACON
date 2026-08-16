"""Training utilities for the BEACON GAT abundance head."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .models import SpatialGATAbundanceHead


def set_seed(seed: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_gat(
    train_graphs: dict,
    device: torch.device,
    epochs: int = 300,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    input_dim: int = 1024,
    log_every: int = 10,
) -> SpatialGATAbundanceHead:
    """Train one GAT head on a dict of sample graphs (labels assumed log1p-scaled)."""
    model = SpatialGATAbundanceHead(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for graph in train_graphs.values():
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            y = graph.y.to(device)
            optimizer.zero_grad()
            pred = model(x, edge_index)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % log_every == 0 or epoch == 0:
            mean_loss = epoch_loss / max(len(train_graphs), 1)
            print(f"  epoch {epoch + 1}/{epochs}  mean MSE (log1p) = {mean_loss:.6f}")

    return model


def save_model(model: nn.Module, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_model(
    path: str | Path,
    device: torch.device,
    input_dim: int = 1024,
) -> SpatialGATAbundanceHead:
    model = SpatialGATAbundanceHead(input_dim=input_dim).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
