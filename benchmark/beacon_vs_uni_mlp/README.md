# BEACON vs UNI+MLP (ablation)

Same Visium split and cell2location Tumor_EGFR / Mac_EREG labels as BEACON.
**UNI features are reused** from `encoder/uni_v1/.../extracted_features_csv/`
(no re-extraction). Head = MLP (no GAT) to isolate the graph inductive bias.

**Do not run heavy jobs on the login node.**

## Fairness vs BEACON

| Item | BEACON | UNI+MLP |
|---|---|---|
| Features | offline UNI 1024-d | same CSVs |
| Split | 6 train / 2 val | same |
| Loss | MSE on log1p | same |
| Optimizer | Adam lr=1e-4, wd=1e-5 | same |
| Epochs | 300 | 300 |
| Seed | 1 | 1 |
| Head | GAT → MLP | MLP only (1024→128→32→1) |

## Run (compute node)

```bash
cd .../benchmark/beacon_vs_uni_mlp
CFG=../configs/benchmark_uni_mlp.yaml

python 01_train_and_predict.py --config "$CFG"
python 02_collect_beacon_spots.py --config "$CFG"
python 03_run_benchmark.py --config "$CFG"
```

## Outputs

| Path | Content |
|---|---|
| `results/weights/{Tumor_EGFR,Mac_EREG}_mlp.pth` | UNI+MLP heads |
| `results/uni_mlp_spots/*.csv` | UNI+MLP spot predictions |
| `results/beacon_spots/*.csv` | BEACON spot predictions |
| `results/metrics_per_sample.csv` | Pearson / Spearman / … |
| `results/metrics_summary.csv` | Means by split × method |
| `results/scatter_plots/` | Spatial maps |

Primary paper table: **val** only (`ST4`, `BC_A1`), method names `BEACON` vs `UNI-MLP`.
