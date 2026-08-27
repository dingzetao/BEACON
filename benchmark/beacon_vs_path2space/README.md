# Path2Space-B vs BEACON (Protocol B)

Fair comparator: **CTransPath (frozen) + MLP** retrained on the same bladder Visium
split and cell2location Tumor_EGFR / Mac_EREG labels as BEACON. See `PROTOCOL_B.md`.

**Do not run these jobs on the login node** — use a GPU compute allocation.

## Setup

1. Download CTransPath weights (`ctranspath.pth`) from  
   https://doi.org/10.5281/zenodo.20174301  
   and place at (or edit YAML):

   `benchmark/beacon_vs_path2space/weights/ctranspath.pth`

2. Dependencies (same env as BEACON + `timm`):

   ```bash
   pip install timm opencv-python-headless torch torchvision torch-geometric scikit-learn scipy pyyaml tqdm pandas matplotlib
   ```

3. Config: `../configs/benchmark_path2space_b.yaml`

## Run order (compute node)

```bash
cd /path/to/pathospatial_model/benchmark/beacon_vs_path2space
CFG=../configs/benchmark_path2space_b.yaml

# 1) CTransPath features for all 8 slides (both niche labels)
# 在GPU上运行 python 01_extract_ctranspath_features.py --config "$CFG"
sbatch slurm_CTransPath.slurm

# 2) Train MLP on 6 train slides; predict train + val (all 8)
python 02_train_and_predict.py --config "$CFG"

# 3) BEACON spot predictions (existing UNI CSVs + GAT weights)
python 03_collect_beacon_spots.py --config "$CFG"

# 4) Metrics + scatter plots (primary = val: ST4, BC_A1)
python 04_run_benchmark.py --config "$CFG"
```

## Outputs

| Path | Content |
|---|---|
| `results/ctranspath_features/{Tumor_EGFR,Mac_EREG}/*.csv` | 768-d embeddings + labels |
| `results/weights/{Tumor_EGFR,Mac_EREG}_mlp.pth` | Path2Space-B heads |
| `results/path2space_spots/{sample}_{cell_type}.csv` | Path2Space-B predictions |
| `results/beacon_spots/{sample}_{cell_type}.csv` | BEACON predictions |
| `results/metrics_per_sample.csv` | Pearson / Spearman / … per slide |
| `results/metrics_summary.csv` | Means by split × method × niche |
| `results/scatter_plots/` | Spatial maps |

## Notes

- Patch geometry matches BEACON UNI (`128×128` spot crops → resize 224 for CTransPath).
- Loss: MSE on `log1p` abundance; inference uses `expm1` (same as BEACON).
- MLP recipe (Path2Space-like): `lr=1e-4`, `dropout=0.2`, `epochs≤200`, early stop on **train-internal** spot holdout Pearson (`patience=50`), then retrain on all train spots for `best_epoch`. Never uses `ST4` / `BC_A1` for tuning.
- Primary paper table: **val** rows only (`ST4`, `BC_A1`).
