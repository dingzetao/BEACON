#!/usr/bin/env python3
"""
01 — Extract frozen CTransPath features for Visium spots (Protocol B).

One forward pass per sample; writes one CSV per (sample, cell_type) with
matching labels (same patch geometry as BEACON UNI: 128×128 center crop).

Writes:
  {output_dir}/ctranspath_features/{cell_type}/{paper_name}_CTransPath_features.csv

Columns: sample_id, spot_id, x, y, label, feature_vector (JSON list, 768-d)

Run on a compute node (GPU recommended). Do not run on the login node.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config_io import load_benchmark_config, resolve_label_column, resolve_project_path
from models import CTransPathFeatureExtractor


def load_positions_flexible(positions_csv: Path) -> pd.DataFrame:
    """Return columns 0=barcode, 1=in_tissue, 4=pxl_row, 5=pxl_col."""
    with open(positions_csv, "r", encoding="utf-8") as f:
        first = f.readline().strip().lower()
    if first.startswith("barcode") or "pxl_row" in first:
        df = pd.read_csv(positions_csv)
        return pd.DataFrame(
            {
                0: df["barcode"].astype(str),
                1: df["in_tissue"].astype(int),
                4: df["pxl_row_in_fullres"].astype(int),
                5: df["pxl_col_in_fullres"].astype(int),
            }
        )
    return pd.read_csv(positions_csv, header=None)


class VisiumPatchDataset(Dataset):
    """128×128 center crop at Visium spot centers (BEACON / uni_v1 geometry)."""

    def __init__(
        self,
        image_path: Path,
        positions_csv: Path,
        spot_ids: list[str],
        patch_size: int = 128,
    ):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pos = load_positions_flexible(Path(positions_csv))
        pos = pos[pos[1] == 1]
        pos_map = {str(r[0]): (int(r[5]), int(r[4])) for _, r in pos.iterrows()}

        half_safe = int(patch_size * 1.5) // 2
        half = patch_size // 2
        self.valid = []
        for spot_id in spot_ids:
            if spot_id not in pos_map:
                continue
            px, py = pos_map[spot_id]
            if not (
                py - half_safe >= 0
                and py + half_safe < self.image.shape[0]
                and px - half_safe >= 0
                and px + half_safe < self.image.shape[1]
            ):
                continue
            self.valid.append({"spot_id": spot_id, "x": px, "y": py, "half": half})

        self.transform = T.Compose(
            [T.ToPILImage(), T.CenterCrop(size=patch_size), T.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.valid)

    def __getitem__(self, idx: int):
        d = self.valid[idx]
        h = d["half"]
        patch = self.image[d["y"] - h : d["y"] + h, d["x"] - h : d["x"] + h]
        return self.transform(patch), d["x"], d["y"], d["spot_id"]


def common_spot_ids(deconv_paths: list[Path]) -> list[str]:
    sets = []
    for p in deconv_paths:
        df = pd.read_csv(p, index_col=0)
        sets.append(set(df.index.astype(str)))
    return sorted(set.intersection(*sets)) if sets else []


@torch.no_grad()
def extract_features(
    sample_name: str,
    image_path: Path,
    positions_csv: Path,
    spot_ids: list[str],
    model: CTransPathFeatureExtractor,
    device: torch.device,
    patch_size: int,
    batch_size: int,
) -> pd.DataFrame:
    ds = VisiumPatchDataset(image_path, positions_csv, spot_ids, patch_size=patch_size)
    if len(ds) == 0:
        raise RuntimeError(f"{sample_name}: no valid spots")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    for patches, xs, ys, sids in tqdm(loader, desc=sample_name):
        feats = model(patches.to(device)).cpu().numpy()
        for i in range(len(patches)):
            rows.append(
                {
                    "sample_id": sample_name,
                    "spot_id": sids[i],
                    "x": int(xs[i].item()),
                    "y": int(ys[i].item()),
                    "feature_vector": json.dumps(feats[i].tolist()),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Extract CTransPath features")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config)
    p2s = cfg["path2space"]
    weights = resolve_project_path(cfg, p2s["ctranspath_weights"])
    patch_size = int(p2s.get("patch_size", 128))
    batch_size = int(p2s.get("batch_size", 64))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"ctranspath weights={weights}")
    model = CTransPathFeatureExtractor(weights, device).to(device)

    # Resolve label columns once
    label_cols: dict[str, dict[str, str]] = {}
    for cell_type, tcfg in cfg["targets"].items():
        label_cols[cell_type] = {}
        for sample in cfg["samples"]:
            deconv = Path(sample["deconv_csv"])
            col = resolve_label_column(
                deconv,
                tcfg["cell2location_column"],
                tcfg.get("cell2location_column_legacy"),
            )
            label_cols[cell_type][sample["paper_name"]] = col

    for sample in cfg["samples"]:
        paper = sample["paper_name"]
        deconv_paths = [Path(sample["deconv_csv"])]
        spot_ids = common_spot_ids(deconv_paths)
        # Prefer intersection of barcodes that have all target columns
        deconv = pd.read_csv(deconv_paths[0], index_col=0)
        deconv.index = deconv.index.astype(str)
        for cell_type in cfg["targets"]:
            col = label_cols[cell_type][paper]
            spot_ids = [s for s in spot_ids if s in deconv.index and col in deconv.columns]
        spot_ids = sorted(set(spot_ids))

        print(f"\n=== extract {paper} ({len(spot_ids)} candidate spots) ===")
        feat_df = extract_features(
            sample_name=paper,
            image_path=Path(sample["he_image"]),
            positions_csv=Path(sample["positions_csv"]),
            spot_ids=spot_ids,
            model=model,
            device=device,
            patch_size=patch_size,
            batch_size=batch_size,
        )

        for cell_type in cfg["targets"]:
            col = label_cols[cell_type][paper]
            labels = deconv.loc[feat_df["spot_id"].astype(str), col].astype(float).to_numpy()
            out = feat_df.copy()
            out["label"] = labels
            # column order
            out = out[["sample_id", "spot_id", "x", "y", "label", "feature_vector"]]
            out_dir = cfg["_output_dir"] / "ctranspath_features" / cell_type
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / f"{paper}_CTransPath_features.csv"
            out.to_csv(out_csv, index=False)
            print(f"  saved {out_csv} ({len(out)} spots)")


if __name__ == "__main__":
    main()
