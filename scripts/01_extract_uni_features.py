#!/usr/bin/env python3
"""Step 1: extract frozen UNI features from Visium H&E and attach cell2location labels.

Example:
  python scripts/01_extract_uni_features.py --config configs/example_Tumor_EGFR.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beacon.features import TridentUniV1FeatureExtractor, extract_uni_features_for_sample
from beacon.utils import load_yaml, resolve_path


def main():
    parser = argparse.ArgumentParser(description="BEACON UNI feature extraction")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    feat_cfg = cfg["features"]
    output_dir = resolve_path(feat_cfg["output_dir"], ROOT)
    target_column = feat_cfg["target_column"]
    patch_size = int(feat_cfg.get("patch_size", 128))
    batch_size = int(feat_cfg.get("batch_size", 64))
    weights_path = feat_cfg.get("uni_weights_path")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    model = TridentUniV1FeatureExtractor(weights_path=weights_path).to(device)

    for sample in feat_cfg["samples"]:
        name = sample["name"]
        out = extract_uni_features_for_sample(
            sample_name=name,
            image_path=sample["image"],
            positions_csv=sample["positions"],
            deconv_csv=sample["deconv"],
            target_column=target_column,
            model=model,
            device=device,
            output_dir=output_dir,
            batch_size=batch_size,
            patch_size=patch_size,
        )
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
