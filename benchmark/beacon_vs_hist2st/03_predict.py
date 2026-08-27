#!/usr/bin/env python3
"""
03 — Predict all 8 slides with trained Hist2ST-B checkpoints.

Writes:
  {output_dir}/hist2st_spots/{paper_name}_{cell_type}.csv
    columns: spot_id, x, y, hist2st_pred

Predictions are expm1(model_log1p), clipped at 0 (same scale as BEACON).

Run on a GPU compute node (not login node).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
HIST2ST_DIR = SCRIPT_DIR.parent / "hist2st"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(HIST2ST_DIR))

from config_io import load_benchmark_config
from dataset_visium import sanitize_adj
from HIST2ST import Hist2ST  # noqa: E402


def build_model(h: dict) -> Hist2ST:
    return Hist2ST(
        n_genes=int(h.get("n_genes", 1)),
        learning_rate=float(h.get("lr", 1e-5)),
        fig_size=int(h.get("fig_size", 112)),
        kernel_size=int(h.get("kernel_size", 5)),
        patch_size=int(h.get("patch_size", 7)),
        depth1=int(h.get("depth1", 2)),
        depth2=int(h.get("depth2", 8)),
        depth3=int(h.get("depth3", 4)),
        heads=int(h.get("heads", 16)),
        channel=int(h.get("channel", 32)),
        dropout=float(h.get("dropout", 0.2)),
        n_pos=int(h.get("n_pos", 128)),
        zinb=float(h.get("zinb", 0.0)),
        nb=False,
        bake=int(h.get("bake", 0)),
        lamb=float(h.get("lamb", 0.0)),
        policy=str(h.get("policy", "mean")),
        label=None,
        use_checkpoint=False,  # inference: no need
        bake_no_grad=True,
    )


@torch.no_grad()
def predict_slide(model: Hist2ST, pt_path: Path, device: torch.device) -> pd.DataFrame:
    d = torch.load(pt_path, map_location="cpu")
    patches = d["patches"].unsqueeze(0).to(device)  # 1,N,3,H,W
    positions = d["positions"].unsqueeze(0).to(device)  # 1,N,2
    adj = sanitize_adj(d["adj"]).to(device)  # N,N with self-loops
    model.eval()
    pred_log, _, _ = model(patches, positions, adj)
    pred_log = pred_log.squeeze(0).detach().cpu().numpy().reshape(-1)
    pred_log = np.nan_to_num(pred_log, nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.clip(np.expm1(pred_log), 0, None)
    coords = d["coords_px"].numpy()
    return pd.DataFrame(
        {
            "spot_id": [str(s) for s in d["spot_ids"]],
            "x": coords[:, 0],
            "y": coords[:, 1],
            "hist2st_pred": pred.astype(np.float64),
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
    h = cfg["hist2st"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    out_root = cfg["_output_dir"] / "hist2st_spots"
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = cfg["_output_dir"] / "weights"

    targets = (
        list(cfg["targets"].keys())
        if args.cell_type == "both"
        else [args.cell_type]
    )

    for cell_type in targets:
        ckpt_path = ckpt_dir / f"{cell_type}_hist2st.pt"
        if not ckpt_path.is_file():
            print(f"[!] missing checkpoint: {ckpt_path}")
            continue
        blob = torch.load(ckpt_path, map_location=device)
        model_h = blob.get("hist2st", h)
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
