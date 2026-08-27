"""BEACON NiceGUI inference workbench (research / lab use)."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEACON_PKG_ROOT = PROJECT_ROOT / "BEACON"

# Paper-facing names → existing gat_seed checkpoints in this monorepo
DEFAULT_CHECKPOINTS = {
    "Tumor_EGFR": PROJECT_ROOT
    / "decoder"
    / "gat_seed"
    / "model_weights"
    / "tumor_CD59"
    / "gat_abundance_head.pth",
    "Mac_EREG": PROJECT_ROOT
    / "decoder"
    / "gat_seed"
    / "model_weights"
    / "mac_sod2"
    / "gat_abundance_head.pth",
}

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "gui_runs"
DEFAULT_CORE_DIR = PROJECT_ROOT / "gui" / "WSI_data"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Prefer launching GUI with this interpreter (has TRIDENT / UNI):
#   /dssg/home/.../.conda/envs/mambaenv/envs/trident/bin/python -m gui
DEFAULT_TRIDENT_PYTHON = (
    Path.home() / ".conda/envs/mambaenv/envs/trident/bin/python"
)