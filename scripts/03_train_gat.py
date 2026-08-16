#!/usr/bin/env python3
"""Step 3: train the BEACON GAT–MLP abundance head on offline UNI feature CSVs.

Example (toy demo, no UNI / TRIDENT required):
  python scripts/make_toy_demo.py
  python scripts/03_train_gat.py --config configs/toy_demo.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beacon.evaluate import compute_metrics, plot_spatial_scatter, predict_abundance
from beacon.graph import load_sample_graphs
from beacon.train import load_model, save_model, set_seed, train_gat
from beacon.utils import ensure_dir, load_yaml, resolve_path


def main():
    parser = argparse.ArgumentParser(description="Train BEACON GAT head")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg["train"]
    set_seed(int(train_cfg.get("seed", 1)))

    feature_dir = resolve_path(train_cfg["feature_dir"], ROOT)
    train_names = list(train_cfg["train_samples"])
    val_names = list(train_cfg.get("val_samples", []))
    k = int(train_cfg.get("k_neighbors", 6))
    input_dim = int(train_cfg.get("input_dim", 1024))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"features: {feature_dir}")

    train_graphs = load_sample_graphs(feature_dir, train_names, k_neighbors=k)
    val_graphs = (
        load_sample_graphs(feature_dir, val_names, k_neighbors=k) if val_names else {}
    )

    model = train_gat(
        train_graphs,
        device=device,
        epochs=int(train_cfg.get("epochs", 300)),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        input_dim=input_dim,
        log_every=int(train_cfg.get("log_every", 10)),
    )

    out_dir = ensure_dir(resolve_path(train_cfg["output_dir"], ROOT))
    weight_path = save_model(model, out_dir / "gat_abundance_head.pth")
    print(f"saved weights: {weight_path}")

    fig_dir = ensure_dir(out_dir / "evaluation_figures")
    metrics_csv = out_dir / "metrics.csv"
    rows = []
    all_graphs = {**train_graphs, **val_graphs}
    for name, graph in all_graphs.items():
        split = "train" if name in train_graphs else "val"
        labels, preds = predict_abundance(model, graph, device)
        mets = compute_metrics(labels, preds)
        rows.append({"split": split, "sample": name, **mets})
        plot_spatial_scatter(graph.coords, labels, preds, name, fig_dir)
        print(
            f"  [{split}] {name}: Pearson={mets['Pearson']:.4f}, "
            f"Spearman={mets['Spearman']:.4f}, R2={mets['R2']:.4f}"
        )

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "sample", "Pearson", "Spearman", "R2", "MSE", "MAE", "Cont_Jac", "Bin_Jac"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"metrics: {metrics_csv}")


if __name__ == "__main__":
    main()
