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
    cfg["_prepared_dir"] = (root / cfg.get("prepared_dir", "benchmark/beacon_vs_hist2st/prepared")).resolve()
    return cfg


def resolve_project_path(cfg: dict, rel_or_abs: str | Path | None) -> Path | None:
    if rel_or_abs is None:
        return None
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (cfg["_project_root"] / p).resolve()


def resolve_label_column(deconv_csv: Path, primary: str, legacy: str | None = None) -> str:
    import pandas as pd

    df = pd.read_csv(deconv_csv, index_col=0, nrows=0)
    cols = list(df.columns)
    if primary in cols:
        return primary
    if legacy and legacy in cols:
        return legacy
    raise KeyError(
        f"Neither {primary!r} nor {legacy!r} in {deconv_csv}. "
        f"Available (first 8): {cols[:8]}"
    )
