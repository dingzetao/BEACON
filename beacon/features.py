"""Offline UNI feature extraction and cell2location label attachment / relabeling."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class TridentUniV1FeatureExtractor(nn.Module):
    """Frozen UNI_v1 encoder via TRIDENT (optional dependency)."""

    def __init__(self, weights_path: str | None = None):
        super().__init__()
        try:
            from trident.patch_encoder_models import encoder_factory
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Feature extraction requires the TRIDENT package "
                "(https://github.com/mahmoodlab/TRIDENT)."
            ) from e

        kwargs = {}
        if weights_path and Path(weights_path).is_file():
            kwargs["weights_path"] = weights_path
        self.encoder = encoder_factory("uni_v1", **kwargs)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._preprocess(x)).float()


class VisiumExtractionDataset(Dataset):
    """Center-crop Visium H&E patches and attach cell2location abundance labels."""

    def __init__(
        self,
        image_path: str | Path,
        positions_csv: str | Path,
        deconv_csv: str | Path,
        target_column: str,
        patch_size: int = 128,
    ):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        positions = pd.read_csv(positions_csv, header=None)
        positions = positions[positions[1] == 1]
        deconv = pd.read_csv(deconv_csv, index_col=0)
        if target_column not in deconv.columns:
            raise KeyError(
                f"Column {target_column!r} not found in {deconv_csv}. "
                f"Available (first 5): {list(deconv.columns[:5])}"
            )

        half = patch_size // 2
        border = int(patch_size * 1.5) // 2
        self.valid_data = []
        for _, row in positions.iterrows():
            spot_id = str(row[0])
            if spot_id not in deconv.index:
                continue
            px, py = int(row[5]), int(row[4])
            if not (
                py - border >= 0
                and py + border < self.image.shape[0]
                and px - border >= 0
                and px + border < self.image.shape[1]
            ):
                continue
            self.valid_data.append(
                {
                    "spot_id": spot_id,
                    "x": px,
                    "y": py,
                    "label": float(deconv.loc[spot_id, target_column]),
                    "half": half,
                }
            )

        self.transform = T.Compose(
            [T.ToPILImage(), T.CenterCrop(size=patch_size), T.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.valid_data)

    def __getitem__(self, idx: int):
        d = self.valid_data[idx]
        h = d["half"]
        patch = self.image[d["y"] - h : d["y"] + h, d["x"] - h : d["x"] + h]
        return self.transform(patch), d["label"], d["x"], d["y"], d["spot_id"]


@torch.no_grad()
def extract_uni_features_for_sample(
    sample_name: str,
    image_path: str | Path,
    positions_csv: str | Path,
    deconv_csv: str | Path,
    target_column: str,
    model: nn.Module,
    device: torch.device,
    output_dir: str | Path,
    batch_size: int = 64,
    patch_size: int = 128,
) -> Path:
    """Extract frozen UNI embeddings for one Visium slide and save a feature CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = VisiumExtractionDataset(
        image_path, positions_csv, deconv_csv, target_column, patch_size=patch_size
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    rows = []
    for patches, labels, xs, ys, spot_ids in tqdm(loader, desc=sample_name):
        feats = model(patches.to(device)).cpu().numpy()
        for i in range(len(patches)):
            rows.append(
                {
                    "sample_id": sample_name,
                    "spot_id": spot_ids[i],
                    "x": int(xs[i].item()),
                    "y": int(ys[i].item()),
                    "label": float(labels[i].item()),
                    "feature_vector": json.dumps(feats[i].tolist()),
                }
            )

    out_csv = output_dir / f"{sample_name}_UNI_features.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def relabel_feature_csv(
    source_csv: str | Path,
    deconv_csv: str | Path,
    target_column: str,
    output_csv: str | Path,
) -> Path:
    """Replace the `label` column using another cell2location abundance column."""
    feat_df = pd.read_csv(source_csv)
    deconv_df = pd.read_csv(deconv_csv, index_col=0)
    if target_column not in deconv_df.columns:
        raise KeyError(f"{target_column!r} not in {deconv_csv}")

    spot_ids = feat_df["spot_id"].astype(str)
    deconv_df = deconv_df.copy()
    deconv_df.index = deconv_df.index.astype(str)
    missing = spot_ids[~spot_ids.isin(deconv_df.index)]
    if len(missing) > 0:
        raise ValueError(f"{len(missing)} spots missing in deconvolution table")

    out_df = feat_df.copy()
    out_df["label"] = spot_ids.map(lambda sid: float(deconv_df.loc[sid, target_column])).values

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return output_csv
