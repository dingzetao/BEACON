"""Build spatial k-NN graphs from offline UNI feature CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import kneighbors_graph

try:
    from torch_geometric.data import Data
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "BEACON requires torch-geometric. Install with: pip install torch-geometric"
    ) from e


def build_pyg_graph_from_csv(
    csv_path: str | Path,
    k_neighbors: int = 6,
    log1p_label: bool = True,
) -> Data:
    """Load `{sample}_UNI_features.csv` and construct a PyG graph."""
    df = pd.read_csv(csv_path)
    required = {"feature_vector", "x", "y", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

    features = np.array([json.loads(vec) for vec in df["feature_vector"]], dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.float64)
    if log1p_label:
        labels = np.log1p(labels)

    coords = df[["x", "y"]].to_numpy(dtype=np.float64)
    knn = kneighbors_graph(
        coords, n_neighbors=k_neighbors, mode="connectivity", include_self=False
    )
    coo = knn.tocoo()
    edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.float32).unsqueeze(1),
    )
    data.coords = coords
    data.sample_id = Path(csv_path).stem.replace("_UNI_features", "")
    return data


def load_sample_graphs(
    feature_dir: str | Path,
    sample_names: list[str],
    k_neighbors: int = 6,
) -> dict[str, Data]:
    feature_dir = Path(feature_dir)
    graphs = {}
    for name in sample_names:
        csv_path = feature_dir / f"{name}_UNI_features.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        graphs[name] = build_pyg_graph_from_csv(csv_path, k_neighbors=k_neighbors)
    return graphs
