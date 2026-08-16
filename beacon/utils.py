"""Small helpers for config loading and path resolution."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path: str | Path | None, base: str | Path | None = None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute() and base is not None:
        p = Path(base) / p
    return p
