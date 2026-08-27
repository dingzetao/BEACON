# ==============================================================================
# Protocol B method comparison — same visual language as original gat_performance.R
# (bubble heatmap: fill = Pearson r, text = value)
#
# Layout:
#   x = 4 methods × {Train, Test}   (8 columns)
#   y = Tumor Pearson | Mac Pearson (2 rows)
#
# CSV split "val" = held-out ST4 + BC_A1 (no tuning) → labeled Test on the figure.
#
# Usage:
#   cd .../pathospatial_model/benchmark
#   Rscript gat_performance.R
# ==============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(readr)

# ------------------------------------------------------------------------------
# 1. Paths (edit ROOT if needed)
# ------------------------------------------------------------------------------
ROOT <- "/dssg/home/acct-drzhihongliu/drzhihongliu-user2/script/spatial_transcriptome/pathospatial_model/benchmark"

paths <- list(
  path2space = file.path(ROOT, "beacon_vs_path2space/results/metrics_summary.csv"),
  histogene  = file.path(ROOT, "beacon_vs_histogene/results/metrics_summary.csv"),
  hist2st    = file.path(ROOT, "beacon_vs_hist2st/results/metrics_summary.csv")
)

OUT_PDF <- file.path(ROOT, "GAT_Performance.pdf")

# ------------------------------------------------------------------------------
# 2. Collect BEACON + three Protocol-B baselines (Pearson means only)
# ------------------------------------------------------------------------------
# BEACON is duplicated in each pairwise summary → take once
beacon <- read_csv(paths$path2space, show_col_types = FALSE) %>%
  filter(method == "BEACON")

dat <- bind_rows(
  beacon,
  read_csv(paths$path2space, show_col_types = FALSE) %>% filter(method == "Path2Space-B"),
  read_csv(paths$histogene,  show_col_types = FALSE) %>% filter(method == "HisToGene-B"),
  read_csv(paths$hist2st,    show_col_types = FALSE) %>% filter(method == "Hist2ST-B")
)

# ------------------------------------------------------------------------------
# 3. Build Sample-like column labels: Method_Train / Method_Test
# ------------------------------------------------------------------------------
method_levels <- c("BEACON", "Path2Space-B", "HisToGene-B", "Hist2ST-B")

df_long <- dat %>%
  mutate(
    Split = ifelse(split == "train", "Train", "Test"),
    Method = factor(method, levels = method_levels),
    # one x-tick per method×split (Train then Test under each method)
    Sample = paste0(as.character(Method), "_", Split),
    Metric = case_when(
      cell_type == "Tumor_EGFR" ~ "Tumor Pearson",
      cell_type == "Mac_EREG"   ~ "Mac Pearson",
      TRUE ~ cell_type
    ),
    Value = Pearson_mean
  ) %>%
  select(Sample, Metric, Value, Method, Split)

sample_order <- as.vector(rbind(
  paste0(method_levels, "_Train"),
  paste0(method_levels, "_Test")
))

df_long$Sample <- factor(df_long$Sample, levels = sample_order)
# Top row Tumor, bottom row Mac (same rev trick as original)
df_long$Metric <- factor(df_long$Metric, levels = rev(c("Tumor Pearson", "Mac Pearson")))

# ------------------------------------------------------------------------------
# 4. Bubble heatmap (match original theme / palette)
# ------------------------------------------------------------------------------
p <- ggplot(df_long, aes(x = Sample, y = Metric)) +
  geom_hline(
    yintercept = 1:length(unique(df_long$Metric)),
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
    name = "Model\nPerformance"
  ) +
  scale_x_discrete(position = "top") +
  labs(x = NULL, y = NULL) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(
      size = 10, face = "bold", color = "black",
      angle = 20, hjust = 0
    ),
    axis.text.y = element_text(size = 12, face = "bold", color = "black"),
    panel.grid = element_blank(),
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 9),
    plot.margin = margin(25, 25, 20, 25)
  )

# ------------------------------------------------------------------------------
# 5. Export
# ------------------------------------------------------------------------------
print(p)
ggsave(OUT_PDF, plot = p, width = 9.5, height = 2.75)
message("Wrote: ", OUT_PDF)
