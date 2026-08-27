# Path2Space-style vs BEACON — Protocol B (recommended)

**Decision:** use **Protocol B** (direct niche-abundance regression), not Protocol A (predict genes → marker signature).

## Why B over A

| | A (gene → signature) | B (direct abundance) |
|---|---|---|
| Endpoint | Indirect; marker choice confounds the method | Same target as BEACON (cell2location Tumor_EGFR / Mac_EREG) |
| Fairness | Mixes “gene model quality” + “signature definition” | Isolates **H&E encoder + regressor** |
| Paper claim | Harder to interpret | Clean: Path2Space-style head vs UNI+GAT on identical labels |
| Compute | Heavy (~14k genes) | Light (2 scalar targets) |

Protocol A remains optional as a sensitivity analysis; **primary Path2Space benchmark = B**.

---

## What “Path2Space-style” means here

Reproduce Path2Space’s **learning setup**, not necessarily their breast-cancer pretrained gene ensemble:

1. **Tile / spot patch** centered on each Visium spot (same physical scale as Visium; Path2Space uses spot-aligned tiles).  
2. **Pathology encoder** → patch embedding (Path2Space paper: **CTransPath**; fixed or lightly adapted).  
3. **MLP regressor** from embedding → abundance (Path2Space cell-fraction stage is MLP on features; here the target is niche abundance, not cancer/TIL/stroma).  
4. **Cross-slide training**; at inference on held-out slides, **H&E only**.

Do **not** use breast Path2Space gene weights as the primary baseline (domain shift). Retrain / train the MLP (and optionally freeze CTransPath) on **your bladder Visium**.

---

## Data & split (match BEACON)

| Role | Samples |
|---|---|
| Train | ST1 (BCB1_0128), ST2 (11.30T1), ST3 (121T2), BC_B1, BC_C1, BC_D1 |
| Val (held-out) | ST4 (113BC), BC_A1 |

**Labels (identical to BEACON):**

- Tumor_EGFR ← `q05cell_abundance_w_sf_tumor_CD59` (or renamed Tumor_EGFR column)  
- Mac_EREG ← `q05cell_abundance_w_sf_Mφ_SOD2` (or renamed Mac_EREG column)  

Train **two independent heads** (same as BEACON), or one multi-task MLP with two outputs — prefer **two heads** for a 1:1 comparison with `gat_tumor_EGFR` / `gat_mac_EREG`.

**Patches:**

- Prefer the **same 128×128 spot crops** already used for UNI/BEACON (or Path2Space default tile size if you stick strictly to their preprocessing; then state the difference in Methods).  
- Recommendation for fairness: **reuse BEACON/UNI patch geometry** so only encoder + head differ.

---

## Training flow (Path2Space-B)

```text
For each train Visium spot:
  H&E patch → CTransPath (frozen) → 768-d embedding
  → MLP → scalar abundance (log1p target, same as BEACON)

Loss: MSE in log1p space
Optimizer: Adam, lr=1e-4 (Path2Space-like)
Dropout: 0.2
Epochs: max 200; early-stop on train-internal spot holdout Pearson (patience=50),
        then retrain on all train spots for best_epoch
        (do NOT tune on ST4 / BC_A1)
```

**Concrete steps:**

1. **Feature extraction (once)**  
   Extract CTransPath embeddings for all spots on all 8 slides → `{sample}_CTransPath_features.csv`  
   (same CSV schema as UNI features: `spot_id, x, y, label, feature_vector`).

2. **Train MLP abundance heads**  
   - Input: 6 train slides’ embeddings + cell2location labels  
   - Architecture sketch (Path2Space-like): Linear→ReLU→Dropout→…→1, or a small 2–3 layer MLP  
   - **No graph / GAT** (that is BEACON’s inductive bias)  
   - Save: `path2spaceB_Tumor_EGFR.pth`, `path2spaceB_Mac_EREG.pth`

3. **Optional ablation (supplementary)**  
   Same MLP but with **UNI** embeddings instead of CTransPath → isolates GAT vs MLP given UNI.

---

## Evaluation flow (fair vs BEACON)

```text
Held-out slides ST4, BC_A1 (and optionally report train slides separately):

  BEACON:     UNI + GAT   → spot abundance
  Path2Space-B: CTransPath + MLP → spot abundance

  Ground truth: cell2location spot abundance

  Metrics (same as BEACON paper / istar benchmark):
    Pearson r, Spearman ρ (primary)
    Cont_Jac, Bin_Jac (secondary)
```

**Rules:**

1. **Primary table:** val slides only (ST4, BC_A1), H&E-only inference for both.  
2. **Secondary table:** train-slide in-sample reconstruction (optional; label clearly).  
3. Do **not** mix iStar into this primary table; iStar stays in a separate “within-slide imputation” track.  
4. Same spot barcodes, same GT column, same metric code (`metrics.py`).

---

## What to report in the paper

1. **Main Visium method comparison:** BEACON vs Path2Space-B on ST4 & BC_A1 (Pearson / Spearman).  
2. **Clinical track (unchanged):** BEACON on TMA vs IHC / ICB (Path2Space-B can be applied to TMA later if desired).  
3. **iStar track (separate):** within-slide baseline; not claimed as same setting.  
4. Methods sentence: Path2Space-B was retrained on the same bladder Visium split and cell2location niche labels; breast pretrained gene models were not used as the primary comparator.

---

## Implementation checklist

- [x] CTransPath loader + feature extract (`01_extract_ctranspath_features.py`)  
- [x] Feature CSVs for 8 samples × Tumor_EGFR / Mac_EREG  
- [x] Train MLP on 6 slides + predict all 8 (`02_train_and_predict.py`)  
- [x] Collect BEACON spots (`03_collect_beacon_spots.py`)  
- [x] Benchmark table method = `Path2Space-B` (`04_run_benchmark.py`)  
- [ ] (Optional) UNI+MLP ablation  
- [ ] Place `ctranspath.pth` under `weights/` (Zenodo 10.5281/zenodo.20174301)

---

## Out of scope for Protocol B primary

- Breast Path2Space zero-shot gene prediction on bladder  
- Comparing Path2Space cancer/TIL/stroma fractions to Tumor_EGFR/Mac_EREG  
- Claiming Path2Space “as published” without bladder retraining  

---

## One-line summary

**Train a CTransPath + MLP abundance model on the same 6 bladder slides and cell2location Tumor_EGFR / Mac_EREG labels as BEACON; evaluate both with H&E only on ST4 and BC_A1 using Pearson/Spearman.** That is the fair Path2Space benchmark.
