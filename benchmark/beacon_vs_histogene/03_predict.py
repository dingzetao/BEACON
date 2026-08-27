#!/usr/bin/env python3
"""
03 — Predict all 8 slides with trained HisToGene-B checkpoints.

Writes:
  {output_dir}/histogene_spots/{paper_name}_{cell_type}.csv
    columns: spot_id, x, y, histogene_pred

Predictions are expm1(model_log1p), clipped at 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
HISTOGENE_DIR = SCRIPT_DIR.parent / "histogene"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(HISTOGENE_DIR))

from config_io import load_benchmark_config
from vis_model import HisToGene  # noqa: E402


def build_model(h: dict) -> HisToGene:
    return HisToGene(
        patch_size=int(h.get("patch_size", 112)),
        n_layers=int(h.get("n_layers", 4)),
        n_genes=int(h.get("n_genes", 1)),
        dim=int(h.get("dim", 1024)),
        learning_rate=float(h.get("lr", 1e-5)),
        dropout=float(h.get("dropout", 0.1)),
        n_pos=int(h.get("n_pos", 128)),
        use_checkpoint=False,
    )


@torch.no_grad()
def predict_slide(model: HisToGene, pt_path: Path, device: torch.device) -> pd.DataFrame:
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    patches = d["patches_flat"].unsqueeze(0).to(device)
    positions = d["positions"].unsqueeze(0).to(device)
    model.eval()
    pred_log = model(patches, positions)
    pred_log = pred_log.squeeze(0).detach().cpu().numpy().reshape(-1)
    pred = np.clip(np.expm1(pred_log), 0, None)
    coords = d["coords_px"].numpy()
    return pd.DataFrame(
        {
            "spot_id": [str(s) for s in d["spot_ids"]],
            "x": coords[:, 0],
            "y": coords[:, 1],
            "histogene_pred": pred.astype(np.float64),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--cell-type",
        choices=["Tumor_EGFR", "Mac_EREG", "both"],
        default="both",
    )
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config)
    h = cfg["histogene"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    out_root = cfg["_output_dir"] / "histogene_spots"
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = cfg["_output_dir"] / "weights"

    targets = (
        list(cfg["targets"].keys())
        if args.cell_type == "both"
        else [args.cell_type]
    )

    for cell_type in targets:
        ckpt_path = ckpt_dir / f"{cell_type}_histogene.pt"
        if not ckpt_path.is_file():
            print(f"[!] missing checkpoint: {ckpt_path}")
            continue
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_h = blob.get("histogene", h)
        model = build_model(model_h).to(device)
        model.load_state_dict(blob["state_dict"])
        print(f"\n=== predict {cell_type} ===")

        prepared = cfg["_prepared_dir"] / cell_type
        for sample in cfg["samples"]:
            paper = sample["paper_name"]
            pt = prepared / f"{paper}.pt"
            if not pt.is_file():
                print(f"  [!] skip {paper}: missing {pt}")
                continue
            df = predict_slide(model, pt, device)
            out_csv = out_root / f"{paper}_{cell_type}.csv"
            df.to_csv(out_csv, index=False)
            print(f"  {paper}: {len(df)} spots -> {out_csv}")

    print(f"\nDone. Spot tables: {out_root}")


if __name__ == "__main__":
    main()
