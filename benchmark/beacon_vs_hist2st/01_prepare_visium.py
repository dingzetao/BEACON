#!/usr/bin/env python3
"""
01 — Prepare Visium slides as Hist2ST tensors (patches / coords / adj / labels).

Writes:
  {prepared_dir}/{cell_type}/{paper_name}.pt

Run on a compute node (I/O heavy). Do not run on the login node for all 8 large TIFFs if avoidable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config_io import load_benchmark_config, resolve_label_column
from dataset_visium import prepare_one_sample


def main():
    parser = argparse.ArgumentParser(description="Prepare Visium for Hist2ST-B")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config)
    h = cfg["hist2st"]
    fig_size = int(h.get("fig_size", 112))
    neighbor = int(h.get("neighbor", 4))
    prune = str(h.get("prune", "Grid"))
    n_pos = int(h.get("n_pos", 128))
    prepared = cfg["_prepared_dir"]

    for cell_type, tcfg in cfg["targets"].items():
        print(f"\n=== prepare {cell_type} ===")
        out_dir = prepared / cell_type
        out_dir.mkdir(parents=True, exist_ok=True)
        for sample in cfg["samples"]:
            paper = sample["paper_name"]
            deconv = Path(sample["deconv_csv"])
            col = resolve_label_column(
                deconv,
                tcfg["cell2location_column"],
                tcfg.get("cell2location_column_legacy"),
            )
            out_pt = out_dir / f"{paper}.pt"
            info = prepare_one_sample(
                paper_name=paper,
                he_image=Path(sample["he_image"]),
                positions_csv=Path(sample["positions_csv"]),
                deconv_csv=deconv,
                label_column=col,
                out_pt=out_pt,
                fig_size=fig_size,
                neighbor=neighbor,
                prune=prune,
                n_pos=n_pos,
            )
            print(
                f"  {paper}: n_spots={info['n_spots']}  "
                f"grid_emb_max={info['grid_emb_max']}  -> {out_pt}"
            )


if __name__ == "__main__":
    main()
