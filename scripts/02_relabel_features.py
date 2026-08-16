#!/usr/bin/env python3
"""Step 2: relabel existing UNI feature CSVs with another cell2location column.

Use this when UNI embeddings are shared across targets (e.g. Tumor_EGFR -> Mac_EREG).

Example:
  python scripts/02_relabel_features.py --config configs/example_Mac_EREG.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beacon.features import relabel_feature_csv
from beacon.utils import load_yaml, resolve_path


def main():
    parser = argparse.ArgumentParser(description="BEACON feature relabeling")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    rel = cfg["relabel"]
    source_dir = resolve_path(rel["source_feature_dir"], ROOT)
    output_dir = resolve_path(rel["output_feature_dir"], ROOT)
    target_column = rel["target_column"]

    for sample in rel["samples"]:
        name = sample["name"]
        src = source_dir / f"{name}_UNI_features.csv"
        out = output_dir / f"{name}_UNI_features.csv"
        path = relabel_feature_csv(
            source_csv=src,
            deconv_csv=sample["deconv"],
            target_column=target_column,
            output_csv=out,
        )
        print(f"  {name} -> {path}")


if __name__ == "__main__":
    main()
