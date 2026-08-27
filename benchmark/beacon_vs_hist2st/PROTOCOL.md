# Hist2ST-B vs BEACON — Protocol B

## Decision

**Primary fair comparator:** retrain **Hist2ST backbone** on bladder Visium for the **same niche abundances** as BEACON (Tumor_EGFR / Mac_EREG), same 6/2 split, H&E-only inference on held-out slides.

Call the method **Hist2ST-B** (Hist2ST-style), not “Hist2ST as published.”

## What stays Hist2ST vs what is adapted

| Kept from Hist2ST | Adapted for BEACON fairness |
|---|---|
| Spot H&E patches (112×112) | Target = niche abundance (not 785 genes) |
| CNN + Transformer + GNN | Loss = **log1p + MSE** (ZINB coef = 0) |
| Within-slide adjacency + coord embeds | Fixed split ST1–3, BC_B1/C1/D1 → train; ST4, BC_A1 → val |
| Optional self-distillation (`bake=5`; Visium uses `bake_no_grad` so graphs fit) | Two independent heads (one per niche) |

## Reviewer wording

> We evaluate all methods on the same H&E → Tumor_EGFR / Mac_EREG abundance task and the same train/val slides. Hist2ST originally predicts gene counts with ZINB; we retain its vision–Transformer–GNN backbone but replace the output/loss with log1p-MSE abundance regression so differences reflect architecture, not mismatched endpoints. Path2Space-B and UNI+MLP use the same endpoint for the same reason.

## Evaluation

- Metrics: Pearson, Spearman (same `metrics.py` as other Visium benchmarks)
- **Primary table:** val only (ST4, BC_A1)
- Secondary: train in-sample (label clearly)

## Out of scope (primary)

- HER2 pretrained Hist2ST zero-shot on bladder
- Gene prediction → marker signature as the main table
- Claiming bit-identical Hist2ST paper settings
