# BEACON Visium benchmarks (Protocol B)

Cross-slide spot-level comparison of **BEACON** vs fair H&E → abundance baselines on the same Visium slides and cell2location ground truth (**Tumor_EGFR**, **Mac_EREG**).

## Methods

| Method | Folder | Config |
|---|---|---|
| **HisToGene-B** (ViT + coord embed) | [`beacon_vs_histogene/`](beacon_vs_histogene/) | [`configs/benchmark_histogene_b.yaml`](configs/benchmark_histogene_b.yaml) |
| **Hist2ST-B** (CNN + Transformer + GNN) | [`beacon_vs_hist2st/`](beacon_vs_hist2st/) | [`configs/benchmark_hist2st_b.yaml`](configs/benchmark_hist2st_b.yaml) |
| **Path2Space-B** (CTransPath + MLP) | [`beacon_vs_path2space/`](beacon_vs_path2space/) | [`configs/benchmark_path2space_b.yaml`](configs/benchmark_path2space_b.yaml) |
| **UNI+MLP** (same UNI features, no GAT) | [`beacon_vs_uni_mlp/`](beacon_vs_uni_mlp/) | [`configs/benchmark_uni_mlp.yaml`](configs/benchmark_uni_mlp.yaml) |

Each method folder has a `README.md` / `PROTOCOL*.md` with GPU notes and the `01`→`N` pipeline.

## Cohorts (manuscript)

- **Train:** ST1, ST2, ST3, BC_B1, BC_C1, BC_D1  
- **Val / test (primary table):** ST4, BC_A1  

## Before running

1. Edit `project_root` and sample paths (`he_image`, `positions_csv`, `deconv_csv`, BEACON weight / UNI feature dirs) in the YAML under `configs/`.
2. Run on a **GPU compute node** (not a login node). Hist2ST-B / HisToGene-B need substantial GPU memory.
3. Intermediate data (`prepared/`, `logs/`, heavy `results/`, `weights/`) stay local and are gitignored; only scripts, configs, and optional `metrics_*.csv` are meant for GitHub.

## Figures

- `gat_performance.R` / `gat_performance_eachsample.R` — bubble heatmaps from `metrics_summary.csv` / per-sample CSVs  
- Example PDF: `GAT_Performance.pdf` (and optional plots under `figures/`)

## Vendored baselines

- `hist2st/` — Hist2ST backbone used by Hist2ST-B  
- `histogene/` — HisToGene modules used by HisToGene-B  
