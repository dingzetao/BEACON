#!/usr/bin/env python3
"""Create a tiny synthetic Visium-like feature dataset for the smoke-test demo.

This does NOT require UNI / TRIDENT / real H&E images. It writes offline
`*_UNI_features.csv` files compatible with `03_train_gat.py`.

Example:
  python scripts/make_toy_demo.py
  python scripts/03_train_gat.py --config configs/toy_demo.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_sample(
    sample_name: str,
    n_spots: int,
    feat_dim: int,
    seed: int,
    grid: int = 20,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Place spots on a coarse grid with mild jitter
    xs = np.linspace(0, grid * 100, int(np.ceil(np.sqrt(n_spots))))
    ys = np.linspace(0, grid * 100, int(np.ceil(np.sqrt(n_spots))))
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()])[:n_spots]
    coords = coords + rng.normal(0, 5, size=coords.shape)

    # Spatial field -> abundance; features are a noisy linear projection of the field
    field = (
        np.sin(coords[:, 0] / 180.0)
        + 0.5 * np.cos(coords[:, 1] / 220.0)
        + rng.normal(0, 0.1, size=n_spots)
    )
    abundance = np.clip(np.expm1(np.clip(field, -1, 2)), 0, None)

    proj = rng.normal(0, 1, size=(1, feat_dim))
    noise = rng.normal(0, 0.3, size=(n_spots, feat_dim))
    feats = np.tanh(field.reshape(-1, 1) @ proj) + noise

    rows = []
    for i in range(n_spots):
        rows.append(
            {
                "sample_id": sample_name,
                "spot_id": f"{sample_name}_{i}",
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "label": float(abundance[i]),
                "feature_vector": json.dumps(feats[i].astype(float).tolist()),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Create BEACON toy feature CSVs")
    parser.add_argument("--out_dir", default=str(ROOT / "data" / "toy" / "features"))
    parser.add_argument("--feat_dim", type=int, default=64)
    parser.add_argument("--n_spots", type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = {
        "toy_train_A": 1,
        "toy_train_B": 2,
        "toy_train_C": 3,
        "toy_val_A": 4,
        "toy_val_B": 5,
    }
    for name, seed in samples.items():
        df = make_sample(name, n_spots=args.n_spots, feat_dim=args.feat_dim, seed=seed)
        path = out_dir / f"{name}_UNI_features.csv"
        df.to_csv(path, index=False)
        print(f"  wrote {path} ({len(df)} spots)")

    print("\nNext:")
    print("  python scripts/03_train_gat.py --config configs/toy_demo.yaml")


if __name__ == "__main__":
    main()
