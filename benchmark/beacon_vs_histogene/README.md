# HisToGene-B vs BEACON (Protocol B)

Fair comparator: **HisToGene backbone** (flatten 112 patch + coord embeds + ViT)
retrained on the same bladder Visium split and cell2location Tumor_EGFR / Mac_EREG
labels as BEACON. Endpoint = **log1p + MSE**. See `PROTOCOL.md`.

Call it **HisToGene-B** — do not claim “HisToGene as published.”

**Do not run heavy jobs on the login node** — use **a800** for train/predict.

## Setup

1. Vendor code: `../histogene/` ([maxpmx/HisToGene](https://github.com/maxpmx/HisToGene); checkpoint hook added).

2. Dependencies:

   ```bash
   pip install torch torchvision pytorch-lightning einops opencv-python-headless \
     scikit-learn scipy pyyaml pandas matplotlib torch-geometric
   ```

3. Config: `../configs/benchmark_histogene_b.yaml`

## Hyperparameters

| Item | Value |
|---|---|
| Patch | 112×112 (flattened) |
| n_layers / dim / dropout | 4 / 1024 / 0.1 |
| lr | 1e-5 |
| epochs | 350 |
| n_pos | 128 |
| precision | bf16-mixed |
| use_checkpoint | true |
| seed | 12000 |

## Run order

```bash
cd /path/to/pathospatial_model/benchmark/beacon_vs_histogene
CFG=../configs/benchmark_histogene_b.yaml

# 1) Prefer convert from Hist2ST-B prepared/*.pt (no TIF re-crop); else crop H&E
python 01_prepare_visium.py --config "$CFG"

# 2–3) GPU on a800
sbatch slurm_histogene_train.slurm
sbatch slurm_histogene_predict.slurm   # after train

# 4–5) CPU OK
python 04_collect_beacon_spots.py --config "$CFG"
python 05_run_benchmark.py --config "$CFG"
```

| Step | GPU? |
|---|---|
| `01_prepare_visium.py` | No |
| `02_train.py` / `slurm_histogene_train.slurm` | **Yes (a800)** |
| `03_predict.py` / `slurm_histogene_predict.slurm` | **Yes** |
| `04_collect_beacon_spots.py` | No |
| `05_run_benchmark.py` | No |

## Outputs

| Path | Content |
|---|---|
| `prepared/{Tumor_EGFR,Mac_EREG}/*.pt` | Flatten patches + coords + labels |
| `results/weights/{cell_type}_histogene.pt` | Checkpoints |
| `results/histogene_spots/*.csv` | HisToGene-B predictions (`expm1`) |
| `results/beacon_spots/*.csv` | BEACON predictions |
| `results/metrics_per_sample.csv` | Per-slide metrics |
| `results/metrics_summary.csv` | Means by split × method |
| `results/scatter_plots/` | Spatial maps |

Primary paper table: **val** only (`ST4`, `BC_A1`), methods `BEACON` vs `HisToGene-B`.
