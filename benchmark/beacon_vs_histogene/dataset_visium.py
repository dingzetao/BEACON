"""Visium → HisToGene tensors (flattened patches, grid positions, log1p labels)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


def load_positions_flexible(positions_csv: Path) -> pd.DataFrame:
    with open(positions_csv, "r", encoding="utf-8") as f:
        first = f.readline().strip().lower()
    if first.startswith("barcode") or "pxl_row" in first:
        df = pd.read_csv(positions_csv)
        colmap = {c.lower(): c for c in df.columns}

        def pick(*names):
            for n in names:
                if n in df.columns:
                    return df[n]
                if n.lower() in colmap:
                    return df[colmap[n.lower()]]
            raise KeyError(f"None of {names} in {positions_csv}")

        return pd.DataFrame(
            {
                "barcode": pick("barcode").astype(str),
                "in_tissue": pick("in_tissue").astype(int),
                "array_row": pick("array_row").astype(int),
                "array_col": pick("array_col").astype(int),
                "pxl_row": pick("pxl_row_in_fullres", "pxl_row").astype(int),
                "pxl_col": pick("pxl_col_in_fullres", "pxl_col").astype(int),
            }
        )
    raw = pd.read_csv(positions_csv, header=None)
    return pd.DataFrame(
        {
            "barcode": raw[0].astype(str),
            "in_tissue": raw[1].astype(int),
            "array_row": raw[2].astype(int),
            "array_col": raw[3].astype(int),
            "pxl_row": raw[4].astype(int),
            "pxl_col": raw[5].astype(int),
        }
    )


def _embed_grid(grid: np.ndarray, n_pos: int) -> np.ndarray:
    gmin = grid.min(axis=0)
    grid_emb = grid - gmin
    if grid_emb.max() >= n_pos:
        scale = max(1, int(np.ceil((grid_emb.max() + 1) / n_pos)))
        grid_emb = grid_emb // scale
    return np.clip(grid_emb, 0, n_pos - 1)


def convert_from_hist2st_pt(hist2st_pt: Path, out_pt: Path, n_pos: int = 128) -> dict:
    """Reuse Hist2ST-B prepared CHW patches → HisToGene flatten format."""
    d = torch.load(hist2st_pt, map_location="cpu", weights_only=False)
    patches = d["patches"]  # N,3,H,W in [0,1]
    n, c, h, w = patches.shape
    flat = patches.reshape(n, c * h * w)
    payload = {
        "paper_name": d["paper_name"],
        "spot_ids": d["spot_ids"],
        "patches_flat": flat.float(),
        "positions": d["positions"].long(),
        "coords_px": d["coords_px"].float(),
        "label": d["label"].float(),
        "label_log1p": d["label_log1p"].float(),
        "patch_size": int(h),
        "label_column": d.get("label_column", ""),
        "source": f"converted_from:{hist2st_pt}",
    }
    # Re-clip positions if n_pos differs
    pos = payload["positions"].numpy()
    if pos.max() >= n_pos:
        payload["positions"] = torch.tensor(_embed_grid(pos, n_pos), dtype=torch.long)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_pt)
    return {
        "paper_name": payload["paper_name"],
        "n_spots": n,
        "out_pt": str(out_pt),
        "source": "hist2st_convert",
    }


def prepare_one_sample(
    paper_name: str,
    he_image: Path,
    positions_csv: Path,
    deconv_csv: Path,
    label_column: str,
    out_pt: Path,
    patch_size: int = 112,
    n_pos: int = 128,
) -> dict:
    half = patch_size // 2
    img_bgr = cv2.imread(str(he_image))
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {he_image}")
    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    pos = load_positions_flexible(positions_csv)
    pos = pos[pos["in_tissue"] == 1].copy()
    deconv = pd.read_csv(deconv_csv, index_col=0)
    deconv.index = deconv.index.astype(str)
    if label_column not in deconv.columns:
        raise KeyError(f"{label_column} not in {deconv_csv}")

    spot_ids, flats, grid_xy, px_xy, labels = [], [], [], [], []
    for _, row in pos.iterrows():
        sid = str(row["barcode"])
        if sid not in deconv.index:
            continue
        px, py = int(row["pxl_col"]), int(row["pxl_row"])
        if py - half < 0 or py + half > h or px - half < 0 or px + half > w:
            continue
        patch = image[py - half : py + half, px - half : px + half]
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            continue
        # CHW [0,1] then flatten (HisToGene convention)
        patch_t = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        spot_ids.append(sid)
        flats.append(patch_t.reshape(-1))
        grid_xy.append([int(row["array_col"]), int(row["array_row"])])
        px_xy.append([px, py])
        labels.append(float(deconv.loc[sid, label_column]))

    if not flats:
        raise RuntimeError(f"{paper_name}: no valid spots")

    grid = np.asarray(grid_xy, dtype=np.int64)
    grid_emb = _embed_grid(grid, n_pos)
    y = np.asarray(labels, dtype=np.float32)
    y_log = np.log1p(np.clip(y, 0, None)).astype(np.float32)

    payload = {
        "paper_name": paper_name,
        "spot_ids": spot_ids,
        "patches_flat": torch.stack(flats, dim=0),
        "positions": torch.tensor(grid_emb, dtype=torch.long),
        "coords_px": torch.tensor(np.asarray(px_xy), dtype=torch.float32),
        "label": torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        "label_log1p": torch.tensor(y_log, dtype=torch.float32).unsqueeze(1),
        "patch_size": patch_size,
        "label_column": label_column,
        "source": "from_he",
    }
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_pt)
    return {
        "paper_name": paper_name,
        "n_spots": len(spot_ids),
        "out_pt": str(out_pt),
        "source": "from_he",
        "grid_emb_max": int(grid_emb.max()),
    }


class VisiumHisToGeneDataset(torch.utils.data.Dataset):
    """One item = one whole slide (HisToGene batch_size=1)."""

    def __init__(self, pt_paths: list[Path]):
        self.pt_paths = [Path(p) for p in pt_paths]
        for p in self.pt_paths:
            if not p.is_file():
                raise FileNotFoundError(p)

    def __len__(self) -> int:
        return len(self.pt_paths)

    def __getitem__(self, index: int):
        d = torch.load(self.pt_paths[index], map_location="cpu", weights_only=False)
        patches = d["patches_flat"].float()  # N, 3*H*W
        positions = d["positions"].long()  # N, 2
        exp = d["label_log1p"].float()  # N, 1
        return patches, positions, exp
