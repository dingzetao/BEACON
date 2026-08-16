# BEACON

**BEACON** (**B**ladder cancer **E**valuation via **A**ttention-based **C**ross-modal **O**nco-**N**iche) maps routine H&E histology to spatial abundances of Tumor_EGFR and Mac_EREG niche states for ICB risk stratification in muscle-invasive bladder cancer (MIBC).

This repository provides a clean demo of the BEACON computational pipeline described in our manuscript.

> **Manuscript status:** under review. Journal / DOI will be updated upon acceptance.

**Related manuscript**

*Integrative Single-cell and Spatial Analyses Define a Prognostic Myeloid-Tumor Niche that Confers Therapeutic Vulnerability in Immunotherapy-Resistant Muscle-Invasive Bladder Cancer*

Zetao Ding<sup>1,#</sup>, Renjie Wang<sup>1,#</sup>, Haodong Chi<sup>1,#</sup>, Zijie Xu<sup>1</sup>, Zheng Tang<sup>1</sup>, Jifu Ge<sup>1</sup>, Yin Yang<sup>1</sup>, Kaiying Chen<sup>1</sup>, Junru Lu<sup>1</sup>, Qi Pan<sup>1</sup>, Yigang Zeng<sup>1</sup>, Fang Zhang<sup>1</sup>, Zhixian Yao<sup>2,\*</sup>, Zhihong Liu<sup>1,\*</sup>

<sup>1</sup> Department of Urology, Shanghai General Hospital, Shanghai Jiao Tong University School of Medicine, Shanghai, China  
<sup>2</sup> Department of Urology, Institute of Molecular Medicine, Renji Hospital, Shanghai Jiao Tong University School of Medicine, Shanghai, China  

<sup>#</sup> These authors contributed equally.  
<sup>\*</sup> Corresponding authors.

**Corresponding authors**

- Zhixian Yao, Ph.D. — [yzxbrooklyn@sjtu.edu.cn](mailto:yzxbrooklyn@sjtu.edu.cn)  
- Zhihong Liu, Ph.D., Prof. — [drzhihongliu@sjtu.edu.cn](mailto:drzhihongliu@sjtu.edu.cn)

For data access requests, see [`data/README.md`](data/README.md).

## Overview

```
H&E Visium spots ──► frozen UNI encoder ──► 1024-d patch embeddings
                                              │
cell2location abundances ─────────────────────┘
                                              │
                         k-NN spatial graph + GAT–MLP head
                                              │
                    spot abundance / dense heatmap / TMA inference
```

Pipeline steps:

1. **Feature extraction** – crop Visium-centered H&E patches, encode with frozen UNI (via [TRIDENT](https://github.com/mahmoodlab/TRIDENT)), attach cell2location abundance labels  
2. **Relabeling (optional)** – reuse UNI embeddings and swap the abundance target (e.g. Tumor_EGFR → Mac_EREG)  
3. **GAT training** – train a graph-attention abundance head on offline feature CSVs  
4. **Visium dense inference** – sliding-window UNI + GAT heatmaps on whole H&E images  
5. **TMA / clinical inference** – apply the trained model to tissue-microarray cores

Example Visium split used in the manuscript configs:

- **Train:** ST1, ST2, ST3, BC_B1, BC_C1, BC_D1  
- **Val:** ST4, BC_A1  

## Repository layout

```
BEACON/
├── beacon/                 # core Python package
│   ├── models.py           # SpatialGATAbundanceHead
│   ├── features.py         # UNI extraction + relabel
│   ├── graph.py            # k-NN graph construction
│   ├── train.py            # training helpers
│   ├── evaluate.py         # metrics + scatter plots
│   └── infer.py            # dense / TMA inference
├── scripts/                # CLI entry points
├── configs/                # YAML examples
├── data/                   # user data (not tracked; see data/README.md)
└── outputs/                # run outputs (not tracked)
```

## Installation

```bash
git clone https://github.com/dingzetao/BEACON.git BEACON
cd BEACON
conda create -n beacon python=3.10 -y
conda activate beacon
pip install -r requirements.txt
```

Additional notes:

- **Training / toy demo** only needs PyTorch + PyTorch Geometric + scientific Python stack.  
- **Feature extraction / dense / TMA inference** additionally require [TRIDENT](https://github.com/mahmoodlab/TRIDENT) and UNI weights (follow TRIDENT docs for setup and license).

## Quick start (toy smoke test, no UNI required)

```bash
python scripts/make_toy_demo.py
python scripts/03_train_gat.py --config configs/toy_demo.yaml
```

This writes synthetic `*_UNI_features.csv` files, trains a small GAT head, and saves:

- `outputs/toy_demo/gat_abundance_head.pth`
- `outputs/toy_demo/metrics.csv`
- `outputs/toy_demo/evaluation_figures/`

## Full pipeline (real Visium + TMA)

Edit the placeholder paths in:

- `configs/example_Tumor_EGFR.yaml`
- `configs/example_Mac_EREG.yaml`

Then run:

```bash
# Tumor_EGFR
python scripts/01_extract_uni_features.py --config configs/example_Tumor_EGFR.yaml
python scripts/03_train_gat.py --config configs/example_Tumor_EGFR.yaml
python scripts/04_infer_visium_dense.py --config configs/example_Tumor_EGFR.yaml
python scripts/05_infer_tma.py --config configs/example_Tumor_EGFR.yaml

# Mac_EREG (reuse UNI embeddings; relabel abundances)
python scripts/02_relabel_features.py --config configs/example_Mac_EREG.yaml
python scripts/03_train_gat.py --config configs/example_Mac_EREG.yaml
python scripts/04_infer_visium_dense.py --config configs/example_Mac_EREG.yaml
python scripts/05_infer_tma.py --config configs/example_Mac_EREG.yaml
```

### Expected feature CSV format

Each `{sample}_UNI_features.csv` contains:

| column | description |
|---|---|
| `sample_id` | sample name |
| `spot_id` | Visium barcode |
| `x`, `y` | spot pixel coordinates |
| `label` | cell2location abundance for Tumor_EGFR or Mac_EREG (original scale) |
| `feature_vector` | JSON list of UNI embedding values |

In the YAML configs, set `target_column` to the abundance column name in your cell2location table (examples use `q05cell_abundance_w_sf_Tumor_EGFR` / `q05cell_abundance_w_sf_Mac_EREG`).

### Model defaults (manuscript)

| setting | value |
|---|---|
| Patch size / stride | 128 × 128 |
| Encoder | frozen UNI_v1 (1024-d) |
| Spatial graph | 6-NN on spot / window centers |
| GAT | 4-head → 128-d, then MLP → scalar |
| Target transform | `log1p` (inverse `expm1` at inference) |
| Loss / optimizer | MSE + Adam (`lr=1e-4`, `wd=1e-5`) |
| Epochs | 300 |

## Citation

If you use this code, please cite our manuscript (details will be updated after publication):

```bibtex
@article{Ding2026BEACON,
  title={Integrative Single-cell and Spatial Analyses Define a Prognostic Myeloid-Tumor Niche that Confers Therapeutic Vulnerability in Immunotherapy-Resistant Muscle-Invasive Bladder Cancer},
  author={Ding, Zetao and Wang, Renjie and Chi, Haodong and Xu, Zijie and Tang, Zheng and Ge, Jifu and Yang, Yin and Chen, Kaiying and Lu, Junru and Pan, Qi and Zeng, Yigang and Zhang, Fang and Yao, Zhixian and Liu, Zhihong},
  year={2026},
  note={Manuscript under review}
}
```

## License

This demo code is released for academic research. UNI / TRIDENT and any clinical images remain subject to their original licenses and institutional approvals. Patient-level images and clinical metadata are **not** included in this repository.

## Acknowledgements

- [UNI](https://github.com/mahmoodlab/UNI) / [TRIDENT](https://github.com/mahmoodlab/TRIDENT) pathology foundation models  
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)  
- cell2location for spatial cell-type deconvolution
