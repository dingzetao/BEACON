#!/usr/bin/env python3
"""
01 — Train UNI+MLP on 6 train slides (offline UNI CSVs); predict all 8 slides.

Matches BEACON (gat_*_t6p2.py):
  - features: encoder/uni_v1/.../{internal}_UNI_features.csv
  - log1p + MSE, Adam lr=1e-4, weight_decay=1e-5, seed=1, epochs=300
  - one optimizer step per train sample per epoch (same loop structure as GAT)
  - no graph / GAT

Writes:
  {output_dir}/weights/{cell_type}_mlp.pth
  {output_dir}/uni_mlp_spots/{paper}_{cell_type}.csv
    columns: spot_id, x, y, uni_mlp_pred

Run on a compute node. Do not run on the login node.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config_io import load_benchmark_config, resolve_project_path
from models import UniAbundanceMLP


def set_seed(seed: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sample_tensors(csv_path: Path) -> dict:
    """Load UNI feature CSV; labels → log1p (same as BEACON build_pyg_graph_from_csv)."""
    df = pd.read_csv(csv_path)
    feats = np.array([json.loads(v) for v in df["feature_vector"]], dtype=np.float32)
    y_log = np.log1p(np.clip(df["label"].to_numpy(dtype=np.float64), 0, None)).astype(
        np.float32
    )
    return {
        "x": torch.tensor(feats, dtype=torch.float32),
        "y": torch.tensor(y_log, dtype=torch.float32).unsqueeze(1),
        "spot_id": df["spot_id"].astype(str).tolist(),
        "coords": df[["x", "y"]].to_numpy(dtype=np.float64),
    }


def train_mlp(
    train_data: dict[str, dict],
    device: torch.device,
    input_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    log_every: int,
) -> UniAbundanceMLP:
    model = UniAbundanceMLP(input_dim=input_dim, dropout=dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()

    print(f">>> UNI+MLP training ({epochs} epochs, {len(train_data)} train slides)")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for name, sample in train_data.items():
            x = sample["x"].to(device)
            y = sample["y"].to(device)
            opt.zero_grad()
            pred = model(x)
            loss = crit(pred, y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        if epoch == 0 or (epoch + 1) % log_every == 0:
            mean_loss = epoch_loss / max(len(train_data), 1)
            print(f"  epoch {epoch+1}/{epochs}  mean MSE(log1p)={mean_loss:.6f}")
    return model


@torch.no_grad()
def predict_sample(model: UniAbundanceMLP, sample: dict, device: torch.device) -> pd.DataFrame:
    model.eval()
    preds_log = model(sample["x"].to(device)).cpu().numpy().flatten()
    preds = np.expm1(preds_log)
    return pd.DataFrame(
        {
            "spot_id": sample["spot_id"],
            "x": sample["coords"][:, 0],
            "y": sample["coords"][:, 1],
            "uni_mlp_pred": preds,
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Train/predict UNI+MLP abundance heads")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config)
    mt = cfg["mlp_training"]
    seed = int(mt.get("seed", 1))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, seed={seed}")

    train_names = set(cfg["train_samples"])
    paper_to_internal = {s["paper_name"]: s["internal_name"] for s in cfg["samples"]}
    all_papers = [s["paper_name"] for s in cfg["samples"]]

    weights_dir = cfg["_output_dir"] / "weights"
    pred_dir = cfg["_output_dir"] / "uni_mlp_spots"
    weights_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    for cell_type, tcfg in cfg["targets"].items():
        feat_dir = resolve_project_path(cfg, tcfg["uni_feature_dir"])
        print(f"\n=== {cell_type}  feature_dir={feat_dir} ===")

        train_data: dict[str, dict] = {}
        for paper in cfg["train_samples"]:
            internal = paper_to_internal[paper]
            csv_path = feat_dir / f"{internal}_UNI_features.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(csv_path)
            train_data[paper] = load_sample_tensors(csv_path)
            print(f"  train {paper} ({internal}): {len(train_data[paper]['spot_id'])} spots")

        model = train_mlp(
            train_data,
            device,
            input_dim=int(mt.get("input_dim", 1024)),
            epochs=int(mt.get("epochs", 300)),
            lr=float(mt.get("lr", 1e-4)),
            weight_decay=float(mt.get("weight_decay", 1e-5)),
            dropout=float(mt.get("dropout", 0.2)),
            log_every=int(mt.get("log_every", 10)),
        )
        wpath = weights_dir / f"{cell_type}_mlp.pth"
        torch.save(model.state_dict(), wpath)
        print(f"saved {wpath}")

        for paper in all_papers:
            internal = paper_to_internal[paper]
            csv_path = feat_dir / f"{internal}_UNI_features.csv"
            if not csv_path.is_file():
                print(f"  [!] skip {paper}: missing {csv_path}")
                continue
            sample = load_sample_tensors(csv_path)
            out = predict_sample(model, sample, device)
            out_csv = pred_dir / f"{paper}_{cell_type}.csv"
            out.to_csv(out_csv, index=False)
            split = "train" if paper in train_names else "val"
            print(f"  predict [{split}] {paper}: {len(out)} -> {out_csv}")


if __name__ == "__main__":
    main()
