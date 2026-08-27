# Hist2ST-B vs BEACON (Protocol B)

Fair comparator: **Hist2ST backbone** (CNN + Transformer + GNN) retrained on the
same bladder Visium split and cell2location Tumor_EGFR / Mac_EREG labels as BEACON.
Endpoint = **log1p + MSE** (ZINB coef = 0). See `PROTOCOL.md`.

Call it **Hist2ST-B** / Hist2ST-style — do not claim “Hist2ST as published.”

**Do not run heavy jobs on the login node** — use a GPU compute allocation.

## Setup

1. Upstream vendor lives at `../hist2st/` (biomed-AI/Hist2ST clone; thin wrappers only).

2. Dependencies (GPU env):

   ```bash
   pip install torch torchvision pytorch-lightning einops opencv-python-headless \
     scikit-learn scipy pyyaml pandas matplotlib scanpy anndata torch-geometric
   ```

3. Config: `../configs/benchmark_hist2st_b.yaml`

## Hyperparameters (Hist2ST defaults; not tuned on ST4 / BC_A1)

| Item | Value |
|---|---|
| Patch | 112×112 (Hist2ST default; BEACON uses 128) |
| lr | 1e-5 |
| epochs | 350 |
| dropout | 0.2 |
| depths / heads / channel | 2 / 8 / 4 / 16 / 32 |
| bake / lamb | **0 / 0**（Visium 防 NaN；原 HER2 可开 bake） |
| precision | **fp32**（`32-true`） |
| use_checkpoint | true |
| gradient_clip_val | 1.0 |
| zinb | 0.0 |
| seed | 12000 |
| n_pos | 128 (Visium array coords) |
| GPU | full **a800** / A100 (not debuga100 MIG 5GB) |

Two independent models (Tumor_EGFR, Mac_EREG).

## Run order

```bash
cd /path/to/pathospatial_model/benchmark/beacon_vs_hist2st
CFG=../configs/benchmark_hist2st_b.yaml

# 1) Patches + coords + adj + log1p labels (CPU / I/O; compute node OK)
python 01_prepare_visium.py --config "$CFG"

# 2–3) GPU on a800 (fp32 + bake=0 + adj self-loops; retrain after NaN fix)
sbatch slurm_hist2st_train.slurm
sbatch slurm_hist2st_predict.slurm   # after train finishes

# 4) BEACON spot predictions (CPU OK; GPU optional)
python 04_collect_beacon_spots.py --config "$CFG"

# 5) Metrics + scatter plots (CPU)
python 05_run_benchmark.py --config "$CFG"
```

| Step | GPU? |
|---|---|
| `01_prepare_visium.py` | No |
| `02_train.py` / `slurm_hist2st_train.slurm` | **Yes** |
| `03_predict.py` / `slurm_hist2st_predict.slurm` | **Yes** |
| `04_collect_beacon_spots.py` | No (optional CUDA) |
| `05_run_benchmark.py` | No |

Optional single niche: `--cell-type Tumor_EGFR` or `Mac_EREG` on steps 02–03.

## Outputs

| Path | Content |
|---|---|
| `prepared/{Tumor_EGFR,Mac_EREG}/*.pt` | Hist2ST tensors per slide |
| `results/weights/{cell_type}_hist2st.pt` | Hist2ST-B checkpoints |
| `results/hist2st_spots/*.csv` | Hist2ST-B spot predictions (`expm1`) |
| `results/beacon_spots/*.csv` | BEACON spot predictions |
| `results/metrics_per_sample.csv` | Pearson / Spearman / … |
| `results/metrics_summary.csv` | Means by split × method |
| `results/scatter_plots/` | Spatial maps |

Primary paper table: **val** only (`ST4`, `BC_A1`), methods `BEACON` vs `Hist2ST-B`.

## Fairness note

Patch size follows Hist2ST architecture default (112), not BEACON’s 128 — stated in
Methods; not tuned on the held-out val slides.
