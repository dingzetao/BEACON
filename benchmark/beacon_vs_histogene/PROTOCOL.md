# HisToGene-B vs BEACON — Protocol B

## Decision

**Primary fair comparator:** retrain **HisToGene backbone** (flatten patch + spatial coord embeddings + ViT) on bladder Visium for the **same niche abundances** as BEACON (Tumor_EGFR / Mac_EREG), same 6/2 split, H&E-only inference on held-out slides.

Call the method **HisToGene-B** (HisToGene-style), not “HisToGene as published.”

## What stays HisToGene vs what is adapted

| Kept from HisToGene | Adapted for BEACON fairness |
|---|---|
| Spot H&E patches (112×112), flattened | Target = niche abundance (`n_genes=1`) |
| Linear patch embed + (x,y) Embedding + ViT | Labels = **log1p(abundance)**; MSE (native form) |
| Whole-slide Transformer batching | Fixed split ST1–3, BC_B1/C1/D1 → train; ST4, BC_A1 → val |
| Defaults: lr=1e-5, dropout 0.1, depth 4 | `n_pos=128`; Visium bf16 + gradient checkpointing |

## Reviewer wording

> We evaluate all methods on the same H&E → Tumor_EGFR / Mac_EREG abundance task and the same train/val slides. HisToGene originally predicts gene expression with MSE via a Vision Transformer with spatial embeddings; we retain that backbone but set the output to log1p-MSE niche abundance so differences reflect architecture, not mismatched endpoints. Path2Space-B, UNI+MLP, and Hist2ST-B use the same endpoint for the same reason.

## Evaluation

- Metrics: Pearson, Spearman (same as other Visium benchmarks)
- **Primary table:** val only (ST4, BC_A1)
- Secondary: train in-sample (label clearly)

## Out of scope (primary)

- HER2 pretrained HisToGene zero-shot on bladder
- Super-resolution denser-grid maps as primary metrics
- Claiming bit-identical HisToGene paper settings
