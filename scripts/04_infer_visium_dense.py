#!/usr/bin/env python3
"""Step 4: dense sliding-window inference on Visium H&E images.

Requires TRIDENT + UNI weights. Uses a trained GAT head from step 3.

Example:
  python scripts/04_infer_visium_dense.py --config configs/example_Tumor_EGFR.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beacon.infer import TridentUniV1InferenceEncoder, infer_dense_heatmap, save_dense_heatmap
from beacon.train import load_model
from beacon.utils import ensure_dir, load_yaml, resolve_path


def main():
    parser = argparse.ArgumentParser(description="BEACON Visium dense inference")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg["train"]
    infer_cfg = cfg["infer_visium"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = resolve_path(infer_cfg.get("weights", train_cfg["output_dir"] + "/gat_abundance_head.pth"), ROOT)
    out_dir = ensure_dir(resolve_path(infer_cfg["output_dir"], ROOT))

    head = load_model(weight_path, device, input_dim=int(train_cfg.get("input_dim", 1024)))
    encoder = TridentUniV1InferenceEncoder(weights_path=infer_cfg.get("uni_weights_path")).to(device)

    for sample in infer_cfg["samples"]:
        name = sample["name"]
        split = sample.get("split", "sample")
        stem = out_dir / f"Inference_{split}_Dense_GAT_{name}"
        print(f"infer dense: {name}")
        heatmap = infer_dense_heatmap(
            encoder=encoder,
            head=head,
            image_path=sample["image"],
            device=device,
            pos_csv=sample.get("positions"),
            patch_size=int(infer_cfg.get("patch_size", 128)),
            stride=int(infer_cfg.get("stride", 128)),
            mask_min_fraction=float(infer_cfg.get("mask_min_fraction", 0.10)),
            k_neighbors=int(train_cfg.get("k_neighbors", 6)),
            batch_size=int(infer_cfg.get("batch_size", 64)),
        )
        save_dense_heatmap(heatmap, stem, title=f"{name} dense abundance")
        print(f"  saved {stem}.npy")


if __name__ == "__main__":
    main()
