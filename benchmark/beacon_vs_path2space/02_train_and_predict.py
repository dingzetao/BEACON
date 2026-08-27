#!/usr/bin/env python3
"""
02 — Train Path2Space-B MLP abundance heads on 6 train slides; predict all 8 slides.

Training recipe (Path2Space-like):
  Adam lr=1e-4, dropout=0.2, MSE on log1p, max epochs=200
  Early stopping on train-internal spot holdout (NOT ST4/BC_A1),
  patience on Pearson; then retrain on all train spots for best_epoch.

Writes:
  {output_dir}/weights/{cell_type}_mlp.pth
  {output_dir}/path2space_spots/{paper}_{cell_type}.csv
    columns: spot_id, x, y, path2space_pred

Run on a compute node. Do not run on the login node.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config_io import load_benchmark_config
from models import AbundanceMLP


def set_seed(seed: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    feats = np.array([json.loads(v) for v in df["feature_vector"]], dtype=np.float32)
    df = df.copy()
    df["feat"] = list(feats)
    return df


def stack_features(dfs: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([f for df in dfs for f in df["feat"]], axis=0)
    y = np.concatenate([df["label"].to_numpy(dtype=np.float64) for df in dfs], axis=0)
    return X, y


@torch.no_grad()
def pearson_on_holdout(
    model: AbundanceMLP,
    X: torch.Tensor,
    y_raw: np.ndarray,
    device: torch.device,
) -> float:
    """Pearson between GT abundance and expm1(pred_log) on holdout spots."""
    model.eval()
    preds_log = model(X.to(device)).cpu().numpy().flatten()
    preds = np.expm1(preds_log)
    if len(np.unique(preds)) < 2 or len(np.unique(y_raw)) < 2:
        return 0.0
    r, _ = pearsonr(y_raw, preds)
    return float(r) if np.isfinite(r) else 0.0


def _run_epochs(
    model: AbundanceMLP,
    X_t: torch.Tensor,
    y_t: torch.Tensor,
    device: torch.device,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    log_every: int = 20,
    X_val: torch.Tensor | None = None,
    y_val_raw: np.ndarray | None = None,
    patience: int | None = None,
) -> tuple[AbundanceMLP, int, float]:
    """
    Train for up to n_epochs. If X_val given, early-stop on holdout Pearson.
    Returns (model_with_best_or_final_weights, best_epoch_1based, best_pearson).
    """
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    n = len(X_t)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_pearson = -np.inf
    stale = 0

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_t[idx].to(device)
            yb = y_t[idx].to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        mean_loss = epoch_loss / max(n_batches, 1)
        epoch_1 = epoch + 1

        if X_val is not None and y_val_raw is not None:
            r = pearson_on_holdout(model, X_val, y_val_raw, device)
            improved = r > best_pearson + 1e-6
            if improved:
                best_pearson = r
                best_epoch = epoch_1
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if epoch == 0 or epoch_1 % log_every == 0 or improved:
                print(
                    f"  epoch {epoch_1}/{n_epochs}  MSE(log1p)={mean_loss:.6f}  "
                    f"holdout_Pearson={r:.4f}  best={best_pearson:.4f}@ep{best_epoch}"
                )
            if patience is not None and stale >= patience and best_epoch > 0:
                print(
                    f"  early stop at epoch {epoch_1} "
                    f"(patience={patience}, best_epoch={best_epoch}, "
                    f"best_Pearson={best_pearson:.4f})"
                )
                break
        else:
            if epoch == 0 or epoch_1 % log_every == 0 or epoch_1 == n_epochs:
                print(f"  epoch {epoch_1}/{n_epochs}  MSE(log1p)={mean_loss:.6f}")
            best_epoch = epoch_1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_epoch, float(best_pearson if best_pearson > -np.inf else 0.0)


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    hidden_dims: tuple[int, ...],
    dropout: float,
    early_stop_frac: float = 0.2,
    early_stop_patience: int = 50,
    seed: int = 1,
) -> AbundanceMLP:
    """
    Path2Space-like: early-stop on train-internal holdout Pearson, then retrain
    on all train spots for the selected best_epoch (never uses paper val slides).
    """
    y_log = np.log1p(np.clip(y, 0, None)).astype(np.float32)
    n = len(X)
    rng = np.random.RandomState(seed)
    n_val = max(1, int(round(n * early_stop_frac)))
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    fit_idx = perm[n_val:]
    if len(fit_idx) < batch_size:
        # tiny data: skip holdout, train full
        fit_idx = np.arange(n)
        val_idx = np.array([], dtype=int)

    X_fit = torch.tensor(X[fit_idx], dtype=torch.float32)
    y_fit = torch.tensor(y_log[fit_idx], dtype=torch.float32).unsqueeze(1)

    print(
        f"  early-stop split: fit={len(fit_idx)} spots, "
        f"holdout={len(val_idx)} spots (train-internal only)"
    )

    model = AbundanceMLP(input_dim=X.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(
        device
    )

    if len(val_idx) > 0:
        X_val = torch.tensor(X[val_idx], dtype=torch.float32)
        y_val_raw = y[val_idx]
        print("  --- stage 1: find best_epoch on train-internal holdout ---")
        _, best_epoch, best_r = _run_epochs(
            model,
            X_fit,
            y_fit,
            device,
            n_epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            X_val=X_val,
            y_val_raw=y_val_raw,
            patience=early_stop_patience,
        )
        n_final = max(1, best_epoch)
        print(
            f"  --- stage 2: retrain on all {n} train spots for {n_final} epochs "
            f"(best holdout Pearson={best_r:.4f}) ---"
        )
    else:
        n_final = epochs
        print(f"  --- train all spots for {n_final} epochs (no holdout) ---")

    # Fresh init for full-data retrain
    model = AbundanceMLP(input_dim=X.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(
        device
    )
    X_all = torch.tensor(X, dtype=torch.float32)
    y_all = torch.tensor(y_log, dtype=torch.float32).unsqueeze(1)
    model, _, _ = _run_epochs(
        model,
        X_all,
        y_all,
        device,
        n_epochs=n_final,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        patience=None,
    )
    return model


@torch.no_grad()
def predict_df(model: AbundanceMLP, df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    model.eval()
    X = np.stack(df["feat"].to_list(), axis=0)
    preds_log = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy().flatten()
    preds = np.expm1(preds_log)
    return pd.DataFrame(
        {
            "spot_id": df["spot_id"].astype(str),
            "x": df["x"],
            "y": df["y"],
            "path2space_pred": preds,
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Train/predict Path2Space-B MLP")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config)
    p2s = cfg["path2space"]
    seed = int(p2s.get("seed", 1))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_names = list(cfg["train_samples"])
    all_names = [s["paper_name"] for s in cfg["samples"]]
    hidden = tuple(int(x) for x in p2s.get("mlp_hidden", [512, 256, 64]))
    epochs = int(p2s.get("epochs", 200))
    lr = float(p2s.get("lr", 1e-4))
    wd = float(p2s.get("weight_decay", 1e-5))
    batch_size = int(p2s.get("mlp_batch_size", 256))
    dropout = float(p2s.get("dropout", 0.2))
    early_stop_frac = float(p2s.get("early_stop_frac", 0.2))
    early_stop_patience = int(p2s.get("early_stop_patience", 50))

    print(
        f"hparams: lr={lr}, dropout={dropout}, epochs_max={epochs}, "
        f"early_stop_frac={early_stop_frac}, patience={early_stop_patience}"
    )

    weights_dir = cfg["_output_dir"] / "weights"
    pred_dir = cfg["_output_dir"] / "path2space_spots"
    weights_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    for cell_type in cfg["targets"]:
        feat_dir = cfg["_output_dir"] / "ctranspath_features" / cell_type
        print(f"\n=== {cell_type} ===")
        train_dfs = []
        for name in train_names:
            path = feat_dir / f"{name}_CTransPath_features.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            train_dfs.append(load_feature_table(path))
        X, y = stack_features(train_dfs)
        print(f"train spots={len(y)}, feat_dim={X.shape[1]}")

        model = train_mlp(
            X,
            y,
            device,
            epochs,
            lr,
            wd,
            batch_size,
            hidden,
            dropout,
            early_stop_frac=early_stop_frac,
            early_stop_patience=early_stop_patience,
            seed=seed,
        )
        wpath = weights_dir / f"{cell_type}_mlp.pth"
        torch.save(model.state_dict(), wpath)
        print(f"saved {wpath}")

        for name in all_names:
            path = feat_dir / f"{name}_CTransPath_features.csv"
            if not path.is_file():
                print(f"  [!] skip predict {name}: missing {path}")
                continue
            df = load_feature_table(path)
            out = predict_df(model, df, device)
            out_csv = pred_dir / f"{name}_{cell_type}.csv"
            out.to_csv(out_csv, index=False)
            split = "train" if name in train_names else "val"
            print(f"  predict [{split}] {name}: {len(out)} -> {out_csv}")


if __name__ == "__main__":
    main()
