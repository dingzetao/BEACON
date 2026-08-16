"""Dense sliding-window inference on Visium H&E or TMA cores."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.neighbors import kneighbors_graph
from torch.utils.data import DataLoader


class TridentUniV1InferenceEncoder(nn.Module):
    """Frozen UNI encoder used only during dense / TMA inference."""

    def __init__(self, weights_path: str | None = None):
        super().__init__()
        try:
            from trident.patch_encoder_models import encoder_factory
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Dense / TMA inference requires TRIDENT "
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


def get_tissue_pixel_mask(
    bgr_image: np.ndarray,
    pos_csv: str | Path | None = None,
    spot_radius: int = 80,
    gray_threshold: int = 215,
) -> np.ndarray:
    if pos_csv and Path(pos_csv).is_file():
        h, w = bgr_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pos = pd.read_csv(pos_csv, header=None)
        pos = pos[pos[1] == 1]
        for _, row in pos.iterrows():
            cv2.circle(mask, (int(row[5]), int(row[4])), radius=spot_radius, color=1, thickness=-1)
        return mask
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, gray_threshold, 255, cv2.THRESH_BINARY_INV)
    return (thresh > 0).astype(np.uint8)


@torch.no_grad()
def infer_dense_heatmap(
    encoder: nn.Module,
    head: nn.Module,
    image_path: str | Path,
    device: torch.device,
    pos_csv: str | Path | None = None,
    patch_size: int = 128,
    stride: int = 128,
    mask_min_fraction: float = 0.10,
    k_neighbors: int = 6,
    batch_size: int = 64,
) -> np.ndarray:
    """Return a dense abundance heatmap (NaN outside tissue)."""
    encoder.eval()
    head.eval()

    full_img = cv2.imread(str(image_path))
    if full_img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image_rgb = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape
    pixel_mask = get_tissue_pixel_mask(full_img, pos_csv=pos_csv)

    rows, cols = h // stride, w // stride
    heatmap = np.full((rows, cols), np.nan, dtype=np.float64)

    patch_list, grid_coords, spatial_coords = [], [], []
    to_tensor = T.Compose([T.ToTensor()])

    for i, y in enumerate(range(0, h - patch_size, stride)):
        if i >= rows:
            break
        for j, x in enumerate(range(0, w - patch_size, stride)):
            if j >= cols:
                break
            if np.mean(pixel_mask[y : y + patch_size, x : x + patch_size]) < mask_min_fraction:
                continue
            patch_list.append(to_tensor(image_rgb[y : y + patch_size, x : x + patch_size]))
            grid_coords.append((i, j))
            spatial_coords.append([x + patch_size // 2, y + patch_size // 2])

    if not patch_list:
        return heatmap

    feats = []
    loader = DataLoader(patch_list, batch_size=batch_size, shuffle=False)
    for batch in loader:
        feats.append(encoder(batch.to(device)).cpu())
    x_inf = torch.cat(feats, dim=0)
    spatial_np = np.asarray(spatial_coords)

    k = min(k_neighbors, len(spatial_coords) - 1)
    if k < 1:
        return heatmap

    knn = kneighbors_graph(spatial_np, n_neighbors=k, mode="connectivity", include_self=False)
    coo = knn.tocoo()
    edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long).to(device)

    preds_log = head(x_inf.to(device), edge_index).cpu().numpy().flatten()
    preds = np.expm1(preds_log)
    for idx, (gi, gj) in enumerate(grid_coords):
        heatmap[gi, gj] = preds[idx]
    return heatmap


def save_dense_heatmap(
    heatmap: np.ndarray,
    save_stem: str | Path,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    save_stem = Path(save_stem)
    save_stem.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(save_stem) + ".npy", heatmap)

    plt.figure(figsize=(10, 8))
    cmap = plt.cm.jet.copy()
    cmap.set_bad(color="white")
    im = plt.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Predicted dense spatial abundance")
    plt.title(title or save_stem.name)
    plt.savefig(str(save_stem) + "_dense_heatmap.png", dpi=150)
    plt.savefig(str(save_stem) + "_dense_heatmap.pdf")
    plt.close()
