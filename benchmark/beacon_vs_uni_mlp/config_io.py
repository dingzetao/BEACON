"""Load benchmark YAML and resolve project-relative paths."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_benchmark_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["project_root"])
    cfg["_config_dir"] = path.parent
    cfg["_project_root"] = root
    cfg["_output_dir"] = (root / cfg["output_dir"]).resolve()
    return cfg


def resolve_project_path(cfg: dict, rel_or_abs: str | Path | None) -> Path | None:
    if rel_or_abs is None:
        return None
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (cfg["_project_root"] / p).resolve()
