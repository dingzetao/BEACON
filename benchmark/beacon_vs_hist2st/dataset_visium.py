"""Visium → Hist2ST-format tensors (patches, grid positions, adj, log1p labels)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
HIST2ST_DIR = SCRIPT_DIR.parent / "hist2st"
sys.path.insert(0, str(HIST2ST_DIR))

from graph_construction import calcADJ  # noqa: E402


def load_positions_flexible(positions_csv: Path) -> pd.DataFrame:
    """
    Return columns:
      barcode, in_tissue, array_row, array_col, pxl_row, pxl_col
    """
    with open(positions_csv, "r", encoding="utf-8") as f:
        first = f.readline().strip().lower()
    if first.startswith("barcode") or "pxl_row" in first:
        df = pd.read_csv(positions_csv)
        # normalize common Space Ranger / GEO headers
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
    # Space Ranger list: barcode, in_tissue, array_row, array_col, pxl_row, pxl_col
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


def prepare_one_sample(
    paper_name: str,
    he_image: Path,
    positions_csv: Path,
    deconv_csv: Path,
    label_column: str,
    out_pt: Path,
    fig_size: int = 112,
    neighbor: int = 4,
    prune: str = "Grid",
    n_pos: int = 128,
) -> dict:
    half = fig_size // 2
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

    spot_ids, patches, grid_xy, px_xy, labels = [], [], [], [], []
    for _, row in pos.iterrows():
        sid = str(row["barcode"])
        if sid not in deconv.index:
            continue
        px, py = int(row["pxl_col"]), int(row["pxl_row"])
        if py - half < 0 or py + half > h or px - half < 0 or px + half > w:
            continue
        patch = image[py - half : py + half, px - half : px + half]
        if patch.shape[0] != fig_size or patch.shape[1] != fig_size:
            continue
        # CHW float in [0,1]
        patch_t = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        ar = int(row["array_row"])
        ac = int(row["array_col"])
        # Hist2ST Embedding index must be in [0, n_pos)
        if ar < 0 or ac < 0 or ar >= n_pos or ac >= n_pos:
            # shift into range while keeping relative grid for adj
            pass
        spot_ids.append(sid)
        patches.append(patch_t)
        grid_xy.append([ac, ar])  # (x,y) like Hist2ST loc
        px_xy.append([px, py])
        labels.append(float(deconv.loc[sid, label_column]))

    if not patches:
        raise RuntimeError(f"{paper_name}: no valid spots")

    grid = np.asarray(grid_xy, dtype=np.int64)
    # Remap grid to [0, n_pos) for embedding if needed
    gmin = grid.min(axis=0)
    grid_emb = grid - gmin
    if grid_emb.max() >= n_pos:
        # scale down rare oversized arrays
        scale = max(1, int(np.ceil((grid_emb.max() + 1) / n_pos)))
        grid_emb = grid_emb // scale
    grid_emb = np.clip(grid_emb, 0, n_pos - 1)

    adj = calcADJ(grid.astype(float), k=neighbor, pruneTag=prune)
    if torch.is_tensor(adj):
        adj = adj.float()
    else:
        adj = torch.tensor(adj, dtype=torch.float32)
    # Self-loops: Grid prune can leave degree-0 spots → GNN /0 → NaN
    n = adj.shape[0]
    adj = ((adj + torch.eye(n)) > 0).float()

    y = np.asarray(labels, dtype=np.float32)
    y_log = np.log1p(np.clip(y, 0, None)).astype(np.float32)

    payload = {
        "paper_name": paper_name,
        "spot_ids": spot_ids,
        "patches": torch.stack(patches, dim=0),  # N,3,H,W
        "positions": torch.tensor(grid_emb, dtype=torch.long),  # N,2 for embed
        "grid_raw": torch.tensor(grid, dtype=torch.long),
        "coords_px": torch.tensor(np.asarray(px_xy), dtype=torch.float32),
        "adj": adj.float() if torch.is_tensor(adj) else torch.tensor(adj, dtype=torch.float32),
        "label": torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        "label_log1p": torch.tensor(y_log, dtype=torch.float32).unsqueeze(1),
        "fig_size": fig_size,
        "label_column": label_column,
    }
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_pt)
    return {
        "paper_name": paper_name,
        "n_spots": len(spot_ids),
        "out_pt": str(out_pt),
        "grid_emb_max": int(grid_emb.max()),
    }


def sanitize_adj(adj: torch.Tensor) -> torch.Tensor:
    """Add self-loops and clean non-finite values (safe for existing prepared/*.pt)."""
    adj = adj.float()
    n = adj.shape[0]
    eye = torch.eye(n, dtype=adj.dtype, device=adj.device)
    adj = torch.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
    adj = ((adj + eye) > 0).float()
    return adj


class VisiumHist2STDataset(torch.utils.data.Dataset):
    """One item = one whole slide (Hist2ST batch_size=1 convention)."""

    def __init__(self, pt_paths: list[Path]):
        self.pt_paths = [Path(p) for p in pt_paths]
        for p in self.pt_paths:
            if not p.is_file():
                raise FileNotFoundError(p)

    def __len__(self) -> int:
        return len(self.pt_paths)

    def __getitem__(self, index: int):
        d = torch.load(self.pt_paths[index], map_location="cpu")
        patches = d["patches"]
        positions = d["positions"]
        exp = d["label_log1p"]  # N,1
        adj = sanitize_adj(d["adj"])
        n = patches.shape[0]
        oris = torch.zeros(n, 1)
        sfs = torch.ones(n)
        return patches, positions, exp, adj, oris, sfs
