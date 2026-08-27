#!/usr/bin/env python3
"""
02 — Train Hist2ST-B (log1p-MSE abundance) on the 6 train slides.

Two independent models: Tumor_EGFR and Mac_EREG.

Visium N can reach ~3800 spots; full Hist2ST bake graphs ×5 OOMs on 80GB.
Defaults: bake=5 with bake_no_grad, gradient checkpointing, bf16-mixed.

Writes:
  {output_dir}/weights/{cell_type}_hist2st.pt
  {output_dir}/weights/{cell_type}_train_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
HIST2ST_DIR = SCRIPT_DIR.parent / "hist2st"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(HIST2ST_DIR))

from config_io import load_benchmark_config
from dataset_visium import VisiumHist2STDataset
from HIST2ST import Hist2ST  # noqa: E402

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def resolve_precision(h: dict) -> str:
    pref = str(h.get("precision", "bf16-mixed"))
    if not torch.cuda.is_available():
        return "32-true"
    if pref in ("32", "32-true", "fp32"):
        return "32-true"
    if "bf16" in pref and torch.cuda.is_bf16_supported():
        return "bf16-mixed"
    if pref in ("16-mixed", "16", "bf16-mixed"):
        return "16-mixed"
    return pref


def build_model(h: dict) -> Hist2ST:
    return Hist2ST(
        n_genes=int(h.get("n_genes", 1)),
        learning_rate=float(h.get("lr", 1e-5)),
        fig_size=int(h.get("fig_size", 112)),
        kernel_size=int(h.get("kernel_size", 5)),
        patch_size=int(h.get("patch_size", 7)),
        depth1=int(h.get("depth1", 2)),
        depth2=int(h.get("depth2", 8)),
        depth3=int(h.get("depth3", 4)),
        heads=int(h.get("heads", 16)),
        channel=int(h.get("channel", 32)),
        dropout=float(h.get("dropout", 0.2)),
        n_pos=int(h.get("n_pos", 128)),
        zinb=float(h.get("zinb", 0.0)),
        nb=False,
        bake=int(h.get("bake", 5)),
        lamb=float(h.get("lamb", 0.5)),
        policy=str(h.get("policy", "mean")),
        label=None,
        use_checkpoint=bool(h.get("use_checkpoint", True)),
        bake_no_grad=bool(h.get("bake_no_grad", True)),
    )


class NonFiniteGuard(pl.Callback):
    """Stop early if weights become non-finite (do not save a poisoned ckpt)."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for p in pl_module.parameters():
            if p is not None and p.data is not None and not torch.isfinite(p.data).all():
                trainer.should_stop = True
                print("[fatal] non-finite weights detected; stopping training", flush=True)
                return


def train_one(cfg: dict, cell_type: str) -> Path:
    h = dict(cfg["hist2st"])
    seed = int(h.get("seed", 12000))
    set_seed(seed)

    prepared = cfg["_prepared_dir"] / cell_type
    train_ids = list(cfg["train_samples"])
    train_pts = [prepared / f"{sid}.pt" for sid in train_ids]
    for p in train_pts:
        if not p.is_file():
            raise FileNotFoundError(f"Missing prepared slide: {p} (run 01_prepare_visium.py)")

    train_ds = VisiumHist2STDataset(train_pts)
    val_ds = VisiumHist2STDataset([train_pts[-1]])
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(h)
    epochs = int(h.get("epochs", 350))
    precision = resolve_precision(h)
    grad_clip = float(h.get("gradient_clip_val", 1.0))

    trainer_kwargs = dict(
        max_epochs=epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        precision=precision,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        gradient_clip_val=grad_clip,
        callbacks=[NonFiniteGuard()],
    )
    if torch.cuda.is_available():
        try:
            trainer = pl.Trainer(accelerator="gpu", devices=1, **trainer_kwargs)
        except TypeError:
            trainer_kwargs.pop("precision", None)
            trainer = pl.Trainer(gpus=1, **trainer_kwargs)
    else:
        try:
            trainer = pl.Trainer(accelerator="cpu", **trainer_kwargs)
        except TypeError:
            trainer = pl.Trainer(**trainer_kwargs)

    print(
        f"[train] {cell_type}: {len(train_ids)} slides, epochs={epochs}, "
        f"lr={h.get('lr')}, bake={h.get('bake')}, lamb={h.get('lamb')}, "
        f"bake_no_grad={h.get('bake_no_grad', True)}, "
        f"checkpoint={h.get('use_checkpoint', True)}, precision={precision}, "
        f"grad_clip={grad_clip}",
        flush=True,
    )
    trainer.fit(model, train_loader, val_loader)

    # Refuse to save if weights are poisoned
    bad = False
    for p in model.parameters():
        if p is not None and not torch.isfinite(p.data).all():
            bad = True
            break
    if bad:
        raise RuntimeError(
            f"{cell_type}: model weights contain NaN/Inf after training. "
            "Not saving checkpoint. Check logs for nonfinite_loss / NonFiniteGuard."
        )

    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ckpt_dir = cfg["_output_dir"] / "weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / f"{cell_type}_hist2st.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cell_type": cell_type,
            "hist2st": {k: h[k] for k in h},
            "train_samples": train_ids,
            "seed": seed,
            "precision": precision,
        },
        out,
    )
    meta = {
        "cell_type": cell_type,
        "train_samples": train_ids,
        "checkpoint": str(out),
        "epochs": epochs,
        "seed": seed,
        "bake": int(h.get("bake", 0)),
        "lamb": float(h.get("lamb", 0.0)),
        "bake_no_grad": bool(h.get("bake_no_grad", True)),
        "use_checkpoint": bool(h.get("use_checkpoint", True)),
        "precision": precision,
        "gradient_clip_val": grad_clip,
    }
    (ckpt_dir / f"{cell_type}_train_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"[ok] saved {out}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--cell-type",
        choices=["Tumor_EGFR", "Mac_EREG", "both"],
        default="both",
    )
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config)
    targets = (
        list(cfg["targets"].keys())
        if args.cell_type == "both"
        else [args.cell_type]
    )
    for ct in targets:
        print(f"\n===== Hist2ST-B train: {ct} =====", flush=True)
        train_one(cfg, ct)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
