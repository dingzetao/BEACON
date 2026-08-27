#!/usr/bin/env python3
"""
04 — Collect BEACON spot predictions (existing UNI CSVs + GAT weights).

Writes:
  {output_dir}/beacon_spots/{paper_name}_{cell_type}.csv
    columns: spot_id, x, y, beacon_pred
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv
except ImportError as e:
    raise ImportError("torch-geometric required") from e

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config_io import load_benchmark_config, resolve_project_path


class SpatialGATAbundanceHead(nn.Module):
    def __init__(self, input_dim: int = 1024, num_heads: int = 4):
        super().__init__()
        self.gat1 = GATConv(input_dim, 64, heads=num_heads, dropout=0.3)
        self.gat2 = GATConv(64 * num_heads, 128, heads=1, concat=False, dropout=0.2)
        self.mlp = nn.Sequential(
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        return torch.relu(self.mlp(x))


def build_graph_from_csv(csv_path: Path, k_neighbors: int) -> Data:
    df = pd.read_csv(csv_path)
    features = np.array([json.loads(v) for v in df["feature_vector"]], dtype=np.float32)
    coords = df[["x", "y"]].to_numpy(dtype=np.float64)
    knn = kneighbors_graph(
        coords, n_neighbors=k_neighbors, mode="connectivity", include_self=False
    )
    coo = knn.tocoo()
    edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
    data = Data(x=torch.tensor(features), edge_index=edge_index)
    data.spot_ids = df["spot_id"].astype(str).tolist()
    data.coords = coords
    return data


@torch.no_grad()
def predict_beacon(
    weights_path: Path,
    feature_csv: Path,
    device: torch.device,
    input_dim: int,
    k_neighbors: int,
) -> pd.DataFrame:
    graph = build_graph_from_csv(feature_csv, k_neighbors)
    model = SpatialGATAbundanceHead(input_dim=input_dim).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    preds_log = model(graph.x.to(device), graph.edge_index.to(device)).cpu().numpy().flatten()
    preds = np.expm1(preds_log)
    return pd.DataFrame(
        {
            "spot_id": graph.spot_ids,
            "x": graph.coords[:, 0],
            "y": graph.coords[:, 1],
            "beacon_pred": preds,
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Collect BEACON spot predictions")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config)
    gat_cfg = cfg.get("gatraining", {})
    input_dim = int(gat_cfg.get("input_dim", 1024))
    k_neighbors = int(gat_cfg.get("k_neighbors", 6))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    out_root = cfg["_output_dir"] / "beacon_spots"
    out_root.mkdir(parents=True, exist_ok=True)

    for cell_type, target_cfg in cfg["targets"].items():
        weights = resolve_project_path(cfg, target_cfg["beacon_weights"])
        feature_dir = resolve_project_path(cfg, target_cfg["uni_feature_dir"])
        print(f"\n=== {cell_type} ===")
        if not weights.is_file():
            print(f"  [!] missing weights: {weights}")
            continue
        for sample in cfg["samples"]:
            paper_name = sample["paper_name"]
            internal = sample["internal_name"]
            feature_csv = feature_dir / f"{internal}_UNI_features.csv"
            if not feature_csv.is_file():
                print(f"  [!] skip {paper_name}: missing {feature_csv}")
                continue
            df = predict_beacon(weights, feature_csv, device, input_dim, k_neighbors)
            out_csv = out_root / f"{paper_name}_{cell_type}.csv"
            df.to_csv(out_csv, index=False)
            print(f"  {paper_name}: {len(df)} spots -> {out_csv}")

    print(f"\nDone. Spot tables: {out_root}")


if __name__ == "__main__":
    main()
