# ==============================================================================
# Per-sample Pearson bubble plots (same style as original gat_performance.R)
#
# For each of 4 methods (BEACON, Path2Space-B, HisToGene-B, Hist2ST-B):
#   x = 8 slides  Train_* (6) + Val_* (2)
#   y = Tumor Pearson | Mac Pearson
#   one PDF per method  (no interactive Rplot / print)
#
# CSV split "val" kept as Val_ on the x-axis (same naming as the original figure).
#
# Usage:
#   cd .../pathospatial_model/benchmark
#   Rscript gat_performance_eachsample.R
# ==============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(readr)
library(stringr)

# ------------------------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------------------------
ROOT <- "/dssg/home/acct-drzhihongliu/drzhihongliu-user2/script/spatial_transcriptome/pathospatial_model/benchmark"

paths <- list(
  path2space = file.path(ROOT, "beacon_vs_path2space/results/metrics_per_sample.csv"),
  histogene  = file.path(ROOT, "beacon_vs_histogene/results/metrics_per_sample.csv"),
  hist2st    = file.path(ROOT, "beacon_vs_hist2st/results/metrics_per_sample.csv")
)

# Optional: map paper_name → display id (closer to original Train_GEO_* labels)
sample_display <- c(
  ST1   = "ST1",
  ST2   = "ST2",
  ST3   = "ST3",
  ST4   = "ST4",
  BC_B1 = "BC_B1",
  BC_C1 = "BC_C1",
  BC_D1 = "BC_D1",
  BC_A1 = "BC_A1"
)

sample_order <- c(
  "Train_ST1", "Train_ST2", "Train_ST3",
  "Train_BC_B1", "Train_BC_C1", "Train_BC_D1",
  "Val_ST4", "Val_BC_A1"
)

method_specs <- list(
  list(name = "BEACON",       file_tag = "BEACON"),
  list(name = "Path2Space-B", file_tag = "Path2Space-B"),
  list(name = "HisToGene-B",  file_tag = "HisToGene-B"),
  list(name = "Hist2ST-B",    file_tag = "Hist2ST-B")
)

# ------------------------------------------------------------------------------
# 2. Load per-sample metrics (BEACON once from path2space file)
# ------------------------------------------------------------------------------
all_rows <- bind_rows(
  read_csv(paths$path2space, show_col_types = FALSE),
  read_csv(paths$histogene,  show_col_types = FALSE) %>% filter(method != "BEACON"),
  read_csv(paths$hist2st,    show_col_types = FALSE) %>% filter(method != "BEACON")
) %>%
  distinct(sample, split, cell_type, method, .keep_all = TRUE)

# ------------------------------------------------------------------------------
# 3. Plot builder (original bubble aesthetic; no print())
# ------------------------------------------------------------------------------
make_bubble_plot <- function(df_method) {
  df_long <- df_method %>%
    mutate(
      Group_Clean = ifelse(split == "train", "Train", "Val"),
      Sample = paste0(Group_Clean, "_", sample_display[sample]),
      Metric = case_when(
        cell_type == "Tumor_EGFR" ~ "Tumor Pearson",
        cell_type == "Mac_EREG"   ~ "Mac Pearson",
        TRUE ~ as.character(cell_type)
      ),
      Value = Pearson
    ) %>%
    select(Sample, Metric, Value)

  df_long$Sample <- factor(df_long$Sample, levels = sample_order)
  df_long$Metric <- factor(df_long$Metric, levels = rev(c("Tumor Pearson", "Mac Pearson")))

  ggplot(df_long, aes(x = Sample, y = Metric)) +
    geom_hline(
      yintercept = seq_len(n_distinct(df_long$Metric)),
      linetype = "dotted", color = "darkgray", size = 0.5
    ) +
    geom_point(aes(fill = Value), shape = 21, size = 16, color = "transparent") +
    geom_text(
      aes(label = sprintf("%.2f", Value)),
      color = "black", size = 4.5, fontface = "bold"
    ) +
    scale_fill_gradient2(
      low = "#7A4E9E",
      mid = "#F7F7F7",
      high = "#2E8B57",
      midpoint = 0.45,
      limits = c(0, 1),
      oob = scales::squish,
      name = "Model\nPerformance"
    ) +
    scale_x_discrete(position = "top") +
    labs(x = NULL, y = NULL) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(
        size = 11, face = "bold", color = "black",
        angle = 15, hjust = 0
      ),
      axis.text.y = element_text(size = 12, face = "bold", color = "black"),
      panel.grid = element_blank(),
      legend.title = element_text(size = 10, face = "bold"),
      legend.text = element_text(size = 9),
      plot.margin = margin(25, 25, 20, 25)
    )
}

# ------------------------------------------------------------------------------
# 4. One PDF per method (ggsave only — no print → no Rplots.pdf)
# ------------------------------------------------------------------------------
for (spec in method_specs) {
  mname <- spec$name
  df_m <- all_rows %>% filter(method == mname)
  if (nrow(df_m) == 0) {
    warning("No rows for method: ", mname, " — skip")
    next
  }
  # expect 8 samples × 2 niches
  n_expect <- 16L
  if (nrow(df_m) != n_expect) {
    warning(mname, ": got ", nrow(df_m), " rows (expected ", n_expect, ")")
  }

  p <- make_bubble_plot(df_m)
  out_pdf <- file.path(ROOT, paste0("GAT_Performance_", spec$file_tag, ".pdf"))
  ggsave(out_pdf, plot = p, width = 9.5, height = 2.75)
  message("Wrote: ", out_pdf)
}

message("Done. Four PDFs under: ", ROOT)
