# Data directory

This folder is a **local workspace** for inputs. Except for this README (and empty `.gitkeep` placeholders), contents under `data/` are **not** tracked by Git.

## Recommended layout

```
data/
├── features/
│   ├── Tumor_EGFR/     # {sample}_UNI_features.csv for Tumor_EGFR
│   └── Mac_EREG/       # relabeled CSVs for Mac_EREG
├── tma_cores/          # TMA core H&E images, e.g. core_A-1.jpg
└── toy/                # synthetic features from scripts/make_toy_demo.py
```

Paper Visium samples used in the example configs:

- **Train:** ST1, ST2, ST3, BC_B1, BC_C1, BC_D1  
- **Val:** ST4, BC_A1  

Feature files should be named `{sample}_UNI_features.csv` (for example `ST1_UNI_features.csv`).

## What to put here

| Content | How it is produced | Commit to GitHub? |
|---|---|---|
| Toy features (`data/toy/`) | `python scripts/make_toy_demo.py` | No |
| Real UNI feature CSVs | `scripts/01_extract_uni_features.py` / `02_relabel_features.py` | No |
| TMA core images | Your local de-arrayed H&E cores | No |
| Raw Visium H&E / positions / cell2location tables | Your study data | No (keep outside the repo or only in a private local config) |

The paths in `configs/example_Tumor_EGFR.yaml` and `configs/example_Mac_EREG.yaml` are placeholders (`/path/to/...`). Point them to your own files when running the full pipeline locally.

## Data availability

Raw clinical images, Visium inputs, and processed feature tables used in the manuscript are **not** distributed in this repository (patient privacy / institutional restrictions).

For academic collaboration or data access requests related to the study, please contact the corresponding authors:

- Zhixian Yao, Ph.D. — [yzxbrooklyn@sjtu.edu.cn](mailto:yzxbrooklyn@sjtu.edu.cn)
- Zhihong Liu, Ph.D., Prof. — [drzhihongliu@sjtu.edu.cn](mailto:drzhihongliu@sjtu.edu.cn)

Please include your name, affiliation, and intended use in the email.

## Privacy

Do **not** upload patient-identifiable whole-slide images, TMA cores, clinical tables, or raw spatial transcriptomic counts to the public repository.
