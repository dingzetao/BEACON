"""Thin adapter: call BEACON dense inference without copying GAT/UNI training code."""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .defaults import BEACON_PKG_ROOT, IMAGE_SUFFIXES, PROJECT_ROOT
from .state import ConfigState, RunState

LogFn = Callable[[str], None]
ProgressFn = Callable[[float, str], None]


def _ensure_beacon_on_path() -> None:
    root = str(BEACON_PKG_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _resolve_device(choice: str) -> torch.device:
    choice = (choice or "auto").lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_image_paths(cfg: ConfigState) -> list[Path]:
    p = Path(cfg.input_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    if cfg.input_mode == "file":
        if not p.is_file():
            raise FileNotFoundError(f"Expected an image file, got: {p}")
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(
                f"Unsupported extension {p.suffix}. "
                f"Supported: {sorted(IMAGE_SUFFIXES)}. "
                "For .svs/.ndpi, dearray cores first (see WSI/TMA_IHC_dearray.py)."
            )
        return [p]
    # directory
    if not p.is_dir():
        raise NotADirectoryError(p)
    files = sorted(
        [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES]
    )
    if not files:
        raise FileNotFoundError(f"No images in {p}")
    return files


def _save_thumbnail(image_path: Path, out_path: Path, max_side: int = 512) -> Path | None:
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return out_path


def run_inference_job(cfg: ConfigState, run: RunState) -> None:
    """Entry point for background thread. Mutates `run` in place."""
    run.status = "running"
    run.error = ""
    run.results.clear()
    run.progress = 0.0

    def log(msg: str) -> None:
        run.log(msg)

    def progress(frac: float, stage: str) -> None:
        run.progress = float(max(0.0, min(1.0, frac)))
        run.stage = stage

    try:
        _ensure_beacon_on_path()
        from beacon.infer import (  # noqa: WPS433
            TridentUniV1InferenceEncoder,
            infer_dense_heatmap,
            save_dense_heatmap,
        )
        from beacon.train import load_model  # noqa: WPS433

        targets = cfg.resolved_targets()
        if not targets:
            raise ValueError("Select at least one niche: Tumor_EGFR and/or Mac_EREG")

        images = collect_image_paths(cfg)
        device = _resolve_device(cfg.device)
        log(f"device={device}")
        log(f"project_root={PROJECT_ROOT}")
        log(f"n_images={len(images)}, targets={[t[0] for t in targets]}")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(cfg.output_dir).expanduser() / f"run_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run.run_dir = str(run_dir)

        # Persist config snapshot
        snap = {
            "input_mode": cfg.input_mode,
            "input_path": cfg.input_path,
            "device": str(device),
            "patch_size": cfg.patch_size,
            "stride": cfg.stride,
            "batch_size": cfg.batch_size,
            "k_neighbors": cfg.k_neighbors,
            "mask_min_fraction": cfg.mask_min_fraction,
            "targets": [{"name": n, "weights": str(w)} for n, w in targets],
            "images": [str(p) for p in images],
        }
        (run_dir / "run_config.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")

        for name, wpath in targets:
            if not wpath.is_file():
                raise FileNotFoundError(f"Missing checkpoint for {name}: {wpath}")

        progress(0.02, "Loading UNI encoder")
        log(f"python={sys.executable}")
        uni_kw = cfg.uni_weights_path.strip() or None
        if uni_kw and not Path(uni_kw).is_file():
            raise FileNotFoundError(f"UNI weights not found: {uni_kw}")
        log("Loading TridentUniV1InferenceEncoder (requires TRIDENT + UNI)…")
        try:
            encoder = TridentUniV1InferenceEncoder(weights_path=uni_kw).to(device)
        except ImportError as e:
            pth = (
                Path(sys.prefix)
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
                / "trident.pth"
            )
            hint = (
                "Cannot import `trident`. GUI uses whatever Python started it "
                f"(now: {sys.executable}).\n"
                "1) Start with the trident env, e.g.\n"
                "   .../envs/trident/bin/python -m gui\n"
                "2) On this cluster, trident was an editable install pointing at\n"
                "   .../biotech_learning/trident/TRIDENT-main — that folder is missing.\n"
                "   Re-clone Mahmoodlab/TRIDENT and `pip install -e .` inside the trident env."
            )
            if pth.is_file():
                hint += f"\n   Current trident.pth → {pth.read_text(encoding='utf-8').strip()}"
            raise ImportError(hint) from e

        heads: dict[str, torch.nn.Module] = {}
        for name, wpath in targets:
            progress(0.05, f"Loading GAT head {name}")
            log(f"Loading GAT weights: {wpath}")
            heads[name] = load_model(wpath, device, input_dim=1024)

        total_steps = max(len(images) * len(targets), 1)
        step = 0
        summary_rows: list[dict] = []

        for img_i, image_path in enumerate(images, start=1):
            stem = image_path.stem.replace("core_", "")
            img_out = run_dir / stem
            img_out.mkdir(parents=True, exist_ok=True)

            thumb = None
            if cfg.make_thumbnail:
                thumb = _save_thumbnail(image_path, img_out / "input_thumbnail.jpg")

            for name, _w in targets:
                step += 1
                stage = f"[{img_i}/{len(images)}] {stem} → {name}"
                progress(0.08 + 0.88 * (step - 1) / total_steps, stage)
                log(f"Infer dense heatmap: {stage}")
                try:
                    heatmap = infer_dense_heatmap(
                        encoder=encoder,
                        head=heads[name],
                        image_path=image_path,
                        device=device,
                        pos_csv=None,
                        patch_size=int(cfg.patch_size),
                        stride=int(cfg.stride),
                        mask_min_fraction=float(cfg.mask_min_fraction),
                        k_neighbors=int(cfg.k_neighbors),
                        batch_size=int(cfg.batch_size),
                    )
                except torch.cuda.OutOfMemoryError as e:
                    raise RuntimeError(
                        "CUDA OOM during inference. Try smaller batch_size or device=cpu."
                    ) from e

                n_valid = int(np.sum(~np.isnan(heatmap)))
                if n_valid == 0:
                    log(f"  [!] no tissue patches for {stem}/{name}; skipped save")
                    continue

                mean_ab = float(np.nanmean(heatmap))
                out_stem = img_out / name
                save_dense_heatmap(
                    heatmap,
                    out_stem,
                    title=f"{stem} · {name}",
                )
                png = Path(str(out_stem) + "_dense_heatmap.png")
                npy = Path(str(out_stem) + ".npy")
                log(f"  mean_abundance={mean_ab:.6f}, n_patches={n_valid}")
                log(f"  saved {png.name}, {npy.name}")

                result = {
                    "image_id": stem,
                    "image_path": str(image_path),
                    "target": name,
                    "mean_abundance": mean_ab,
                    "n_valid_patches": n_valid,
                    "heatmap_png": str(png) if png.is_file() else "",
                    "heatmap_npy": str(npy) if npy.is_file() else "",
                    "thumbnail": str(thumb) if thumb and Path(thumb).is_file() else "",
                }
                run.results.append(result)
                summary_rows.append(
                    {
                        "image_id": stem,
                        "target": name,
                        "mean_abundance": mean_ab,
                        "n_valid_patches": n_valid,
                        "heatmap_png": result["heatmap_png"],
                        "heatmap_npy": result["heatmap_npy"],
                    }
                )
                progress(0.08 + 0.88 * step / total_steps, stage)

        if summary_rows:
            csv_path = run_dir / "summary.csv"
            pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
            log(f"summary CSV: {csv_path}")

        # Combined gallery figure if both niches for same image
        _maybe_write_gallery(run_dir, run.results, log)

        progress(1.0, "Done")
        run.status = "done"
        log(f"Finished. Output directory: {run_dir}")
    except Exception as e:
        run.status = "error"
        run.error = str(e)
        log(f"ERROR: {e}")
        log(traceback.format_exc())
        progress(run.progress, "Error")


def _maybe_write_gallery(run_dir: Path, results: list[dict], log: LogFn) -> None:
    by_img: dict[str, dict[str, str]] = {}
    for r in results:
        by_img.setdefault(r["image_id"], {})[r["target"]] = r.get("heatmap_png") or ""
    for image_id, targets in by_img.items():
        paths = [(k, p) for k, p in targets.items() if p and Path(p).is_file()]
        if len(paths) < 1:
            continue
        fig, axes = plt.subplots(1, len(paths), figsize=(5 * len(paths), 4))
        if len(paths) == 1:
            axes = [axes]
        for ax, (name, png) in zip(axes, paths):
            img = plt.imread(png)
            ax.imshow(img)
            ax.set_title(name)
            ax.axis("off")
        fig.suptitle(image_id)
        out = run_dir / image_id / "gallery.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        log(f"gallery: {out}")
