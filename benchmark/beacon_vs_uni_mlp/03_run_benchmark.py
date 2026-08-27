#!/usr/bin/env python3
"""
03 — Benchmark BEACON vs UNI+MLP against cell2location ground truth.

Prerequisites:
  01_train_and_predict.py
  02_collect_beacon_spots.py

Primary table = val_samples (ST4, BC_A1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config_io import load_benchmark_config
from metrics import compute_abundance_metrics, summarize_metrics


def load_ground_truth(
    deconv_csv: Path,
    column: str,
    column_fallback: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(deconv_csv, index_col=0)
    df.index = df.index.astype(str)
    col = column
    if col not in df.columns and column_fallback and column_fallback in df.columns:
        col = column_fallback
    if col not in df.columns:
        raise KeyError(
            f"Column {column!r} not in {deconv_csv}. "
            f"Available (first 8): {list(df.columns[:8])}"
        )
    return pd.DataFrame(
        {"spot_id": df.index.astype(str), "ground_truth": df[col].astype(float)}
    )


def get_split(cfg: dict, paper_name: str) -> str:
    if paper_name in cfg["train_samples"]:
        return "train"
    if paper_name in cfg["val_samples"]:
        return "val"
    return "other"


def plot_spot_scatter(
    coords: np.ndarray,
    values: np.ndarray,
    title: str,
    out_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    xs, ys = coords[:, 0], -coords[:, 1]
    sc = ax.scatter(xs, ys, c=values, cmap="jet", s=18, alpha=0.85, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="BEACON vs UNI+MLP benchmark")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config)
    out_dir = cfg["_output_dir"]
    beacon_dir = out_dir / "beacon_spots"
    mlp_dir = out_dir / "uni_mlp_spots"
    plot_dir = out_dir / "scatter_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    per_sample_rows = []

    for cell_type, target_cfg in cfg["targets"].items():
        gt_col = target_cfg["cell2location_column"]
        gt_fallback = target_cfg.get("cell2location_column_legacy")
        print(f"\n=== {cell_type} ===")

        for sample in cfg["samples"]:
            paper_name = sample["paper_name"]
            split = get_split(cfg, paper_name)
            deconv_csv = Path(sample["deconv_csv"])
            if not deconv_csv.is_file():
                print(f"  [!] skip {paper_name}: missing deconv {deconv_csv}")
                continue

            gt = load_ground_truth(deconv_csv, gt_col, gt_fallback)
            beacon_csv = beacon_dir / f"{paper_name}_{cell_type}.csv"
            mlp_csv = mlp_dir / f"{paper_name}_{cell_type}.csv"

            method_runs = []
            if beacon_csv.is_file():
                beacon = pd.read_csv(beacon_csv)
                beacon["spot_id"] = beacon["spot_id"].astype(str)
                merged_b = gt.merge(beacon, on="spot_id", how="inner")
                if len(merged_b) >= 10:
                    method_runs.append(("BEACON", merged_b, "beacon_pred"))
                else:
                    print(f"  [!] {paper_name}: only {len(merged_b)} matched spots (BEACON)")
            else:
                print(f"  [!] missing {beacon_csv}")

            if mlp_csv.is_file():
                mlp = pd.read_csv(mlp_csv)
                mlp["spot_id"] = mlp["spot_id"].astype(str)
                merged_m = gt.merge(mlp, on="spot_id", how="inner")
                if len(merged_m) >= 10:
                    method_runs.append(("UNI-MLP", merged_m, "uni_mlp_pred"))
                else:
                    print(f"  [!] {paper_name}: only {len(merged_m)} matched spots (UNI-MLP)")
            else:
                print(f"  [!] missing {mlp_csv}")

            if not method_runs:
                continue

            ref = method_runs[0][1]
            vmin = float(np.nanpercentile(ref["ground_truth"], 2))
            vmax = float(np.nanpercentile(ref["ground_truth"], 98))
            plot_spot_scatter(
                ref[["x", "y"]].to_numpy(),
                ref["ground_truth"].to_numpy(),
                f"{paper_name} GT ({cell_type})",
                plot_dir / f"{paper_name}_{cell_type}_ground_truth",
                vmin=vmin,
                vmax=vmax,
            )

            for method_name, merged, pred_col in method_runs:
                mets = compute_abundance_metrics(
                    merged["ground_truth"].to_numpy(),
                    merged[pred_col].to_numpy(),
                )
                row = {
                    "sample": paper_name,
                    "split": split,
                    "cell_type": cell_type,
                    "method": method_name,
                    "n_matched_spots": len(merged),
                    **mets,
                }
                per_sample_rows.append(row)
                print(
                    f"  {paper_name} [{split}] {method_name}: "
                    f"Pearson={mets['Pearson']:.4f}, Spearman={mets['Spearman']:.4f}"
                )
                plot_spot_scatter(
                    merged[["x", "y"]].to_numpy(),
                    merged[pred_col].to_numpy(),
                    f"{paper_name} {method_name} ({cell_type})",
                    plot_dir / f"{paper_name}_{cell_type}_{method_name}",
                    vmin=vmin,
                    vmax=vmax,
                )

    if not per_sample_rows:
        raise RuntimeError("No samples evaluated. Run 01→02 first.")

    per_sample_df = pd.DataFrame(per_sample_rows)
    summary_df = pd.DataFrame(summarize_metrics(per_sample_rows, group_key="split"))
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = out_dir / "metrics_per_sample.csv"
    summary_path = out_dir / "metrics_summary.csv"
    per_sample_df.to_csv(per_sample_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\nPer-sample metrics: {per_sample_path}")
    print(f"Summary metrics:    {summary_path}")
    print("\n=== Primary (val only) ===")
    val = summary_df[summary_df["split"] == "val"]
    for _, row in val.sort_values(["cell_type", "method"]).iterrows():
        print(
            f"  {row['cell_type']:12s} {row['method']:8s} "
            f"Pearson={row['Pearson_mean']:.4f}  Spearman={row['Spearman_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
