"""Shared abundance metrics (aligned with other Visium benchmarks)."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import jaccard_score, mean_absolute_error, mean_squared_error, r2_score


def compute_abundance_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    mask = np.isfinite(labels) & np.isfinite(preds)
    labels, preds = labels[mask], preds[mask]
    if len(labels) < 2:
        return {
            "Pearson": 0.0,
            "Spearman": 0.0,
            "R2": 0.0,
            "MSE": 0.0,
            "MAE": 0.0,
            "Cont_Jac": 0.0,
            "Bin_Jac": 0.0,
            "n_spots": int(len(labels)),
        }
    if len(np.unique(preds)) == 1 or len(np.unique(labels)) == 1:
        pearson_corr = spearman_corr = 0.0
    else:
        pearson_corr, _ = pearsonr(labels, preds)
        spearman_corr, _ = spearmanr(labels, preds)

    safe_labels = np.clip(labels, 0, None)
    safe_preds = np.clip(preds, 0, None)
    inter = np.sum(np.minimum(safe_labels, safe_preds))
    union = np.sum(np.maximum(safe_labels, safe_preds))
    cont_jac = inter / union if union > 0 else 0.0
    bt = np.percentile(labels, 80)
    bp = np.percentile(preds, 80)
    bin_labels = (labels >= bt).astype(int)
    bin_preds = (preds >= bp).astype(int)
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
        "n_spots": int(len(labels)),
    }


def summarize_metrics(rows: list[dict], group_key: str) -> list[dict]:
    from collections import defaultdict

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row[group_key], row["method"], row["cell_type"])].append(row)
    summary = []
    for (split, method, cell_type), group in sorted(buckets.items()):
        out = {
            "split": split,
            "method": method,
            "cell_type": cell_type,
            "n_samples": len(group),
        }
        for key in ["Pearson", "Spearman", "R2", "MSE", "MAE", "Cont_Jac", "Bin_Jac"]:
            out[f"{key}_mean"] = float(np.mean([g[key] for g in group]))
        summary.append(out)
    return summary
