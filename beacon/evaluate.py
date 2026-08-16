"""Spot-level evaluation metrics and spatial scatter plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import jaccard_score, mean_absolute_error, mean_squared_error, r2_score


@torch.no_grad()
def predict_abundance(model, graph, device) -> tuple[np.ndarray, np.ndarray]:
    """Return abundance predictions and ground truth on the original (expm1) scale."""
    model.eval()
    preds_log = model(graph.x.to(device), graph.edge_index.to(device)).cpu().numpy().flatten()
    preds = np.expm1(preds_log)
    labels = np.expm1(graph.y.numpy().flatten())
    return labels, preds


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    if len(np.unique(preds)) == 1 or len(np.unique(labels)) == 1:
        pearson_corr = spearman_corr = 0.0
    else:
        pearson_corr, _ = pearsonr(labels, preds)
        spearman_corr, _ = spearmanr(labels, preds)

    safe_labels = np.clip(labels, 0, None)
    safe_preds = np.clip(preds, 0, None)
    intersection = np.sum(np.minimum(safe_labels, safe_preds))
    union = np.sum(np.maximum(safe_labels, safe_preds))
    cont_jac = intersection / union if union > 0 else 0.0

    thresh_true = np.percentile(labels, 80)
    thresh_pred = np.percentile(preds, 80)
    bin_labels = (labels >= thresh_true).astype(int)
    bin_preds = (preds >= thresh_pred).astype(int)
    bin_jac = (
        jaccard_score(bin_labels, bin_preds)
        if (bin_labels.sum() + bin_preds.sum()) > 0
        else 0.0
    )

    return {
        "Pearson": float(pearson_corr),
        "Spearman": float(spearman_corr),
        "R2": float(r2_score(labels, preds)),
        "MSE": float(mean_squared_error(labels, preds)),
        "MAE": float(mean_absolute_error(labels, preds)),
        "Cont_Jac": float(cont_jac),
        "Bin_Jac": float(bin_jac),
    }


def plot_spatial_scatter(
    coords: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_name: str,
    save_dir: str | Path,
    vmin_true: float | None = None,
    vmax_true: float | None = None,
    vmin_pred: float | None = None,
    vmax_pred: float | None = None,
) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    xs, ys = coords[:, 0], -coords[:, 1]

    sc1 = axes[0].scatter(
        xs, ys, c=y_true, cmap="jet", s=15, alpha=0.8, vmin=vmin_true, vmax=vmax_true
    )
    axes[0].set_title(f"{sample_name} – Ground truth")
    axes[0].axis("off")
    plt.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04)

    sc2 = axes[1].scatter(
        xs,
        ys,
        c=y_pred,
        cmap="jet",
        marker="s",
        s=45,
        alpha=0.95,
        vmin=vmin_pred,
        vmax=vmax_pred,
    )
    axes[1].set_title(f"{sample_name} – BEACON prediction")
    axes[1].axis("off")
    plt.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = save_dir / f"{sample_name}_spatial_scatter.pdf"
    plt.savefig(out, dpi=150)
    plt.savefig(save_dir / f"{sample_name}_spatial_scatter.png", dpi=150)
    plt.close()
    return out
