#!/usr/bin/env python3
"""
01 — Prepare Visium slides as HisToGene tensors (flatten patches / coords / labels).

Prefer converting from Hist2ST-B prepared/*.pt when available (no TIF re-crop).
Fallback: crop from H&E.

Writes:
  {prepared_dir}/{cell_type}/{paper_name}.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config_io import load_benchmark_config, resolve_label_column
from dataset_visium import convert_from_hist2st_pt, prepare_one_sample


def main():
    parser = argparse.ArgumentParser(description="Prepare Visium for HisToGene-B")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--force-from-he",
        action="store_true",
        help="Ignore Hist2ST prepared cache; crop from H&E",
    )
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config)
    h = cfg["histogene"]
    patch_size = int(h.get("patch_size", 112))
    n_pos = int(h.get("n_pos", 128))
    prepared = cfg["_prepared_dir"]
    hist2st_root = cfg.get("_hist2st_prepared_dir")

    for cell_type, tcfg in cfg["targets"].items():
        print(f"\n=== prepare {cell_type} ===")
        out_dir = prepared / cell_type
        out_dir.mkdir(parents=True, exist_ok=True)
        for sample in cfg["samples"]:
            paper = sample["paper_name"]
            out_pt = out_dir / f"{paper}.pt"
            h2_pt = (
                hist2st_root / cell_type / f"{paper}.pt"
                if hist2st_root is not None
                else None
            )
            if (
                not args.force_from_he
                and h2_pt is not None
                and h2_pt.is_file()
            ):
                info = convert_from_hist2st_pt(h2_pt, out_pt, n_pos=n_pos)
                print(
                    f"  {paper}: n_spots={info['n_spots']}  "
                    f"[from Hist2ST] -> {out_pt}"
                )
                continue

            deconv = Path(sample["deconv_csv"])
            col = resolve_label_column(
                deconv,
                tcfg["cell2location_column"],
                tcfg.get("cell2location_column_legacy"),
            )
            info = prepare_one_sample(
                paper_name=paper,
                he_image=Path(sample["he_image"]),
                positions_csv=Path(sample["positions_csv"]),
                deconv_csv=deconv,
                label_column=col,
                out_pt=out_pt,
                patch_size=patch_size,
                n_pos=n_pos,
            )
            print(
                f"  {paper}: n_spots={info['n_spots']}  "
                f"grid_emb_max={info.get('grid_emb_max')}  -> {out_pt}"
            )


if __name__ == "__main__":
    main()
