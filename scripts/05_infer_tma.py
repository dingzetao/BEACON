#!/usr/bin/env python3
"""Step 5: apply a trained BEACON model to TMA / WSI cores.

Requires TRIDENT + UNI weights.

Example:
  python scripts/05_infer_tma.py --config configs/example_Tumor_EGFR.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beacon.infer import TridentUniV1InferenceEncoder, infer_dense_heatmap, save_dense_heatmap
from beacon.train import load_model
from beacon.utils import ensure_dir, load_yaml, resolve_path


def _natural_core_sort_key(filename: str):
    m = re.search(r"core_([A-Z])-(\d+)\.", filename)
    if m:
        return (m.group(1), int(m.group(2)))
    return (filename, 0)


def main():
    parser = argparse.ArgumentParser(description="BEACON TMA / core inference")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg["train"]
    tma_cfg = cfg["infer_tma"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = resolve_path(
        tma_cfg.get("weights", train_cfg["output_dir"] + "/gat_abundance_head.pth"), ROOT
    )
    core_dir = resolve_path(tma_cfg["core_image_dir"], ROOT)
    out_dir = ensure_dir(resolve_path(tma_cfg["output_dir"], ROOT))
    summary_csv = resolve_path(tma_cfg.get("summary_csv", str(out_dir / "core_mean_abundance.csv")), ROOT)

    head = load_model(weight_path, device, input_dim=int(train_cfg.get("input_dim", 1024)))
    encoder = TridentUniV1InferenceEncoder(weights_path=tma_cfg.get("uni_weights_path")).to(device)

    cores = sorted(
        [
            f
            for f in core_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        ],
        key=lambda p: _natural_core_sort_key(p.name),
    )
    if not cores:
        raise FileNotFoundError(f"No core images in {core_dir}")

    rows = []
    for idx, path in enumerate(cores, start=1):
        core_id = path.stem.replace("core_", "")
        print(f"[{idx}/{len(cores)}] {core_id}")
        heatmap = infer_dense_heatmap(
            encoder=encoder,
            head=head,
            image_path=path,
            device=device,
            pos_csv=None,
            patch_size=int(tma_cfg.get("patch_size", 128)),
            stride=int(tma_cfg.get("stride", 128)),
            mask_min_fraction=float(tma_cfg.get("mask_min_fraction", 0.10)),
            k_neighbors=int(train_cfg.get("k_neighbors", 6)),
            batch_size=int(tma_cfg.get("batch_size", 64)),
        )
        stem = out_dir / f"Inference_TMA_Dense_GAT_{core_id}"
        save_dense_heatmap(heatmap, stem, title=f"{core_id} dense abundance")
        mean_ab = float(np.nanmean(heatmap))
        n_valid = int(np.sum(~np.isnan(heatmap)))
        rows.append(
            {
                "core_id": core_id,
                "core_filename": path.name,
                "mean_abundance": mean_ab,
                "n_valid_patches": n_valid,
            }
        )
        print(f"  mean_abundance={mean_ab:.6f}, n_patches={n_valid}")

    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"summary: {summary_csv}")


if __name__ == "__main__":
    main()
