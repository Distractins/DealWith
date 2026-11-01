# -*- coding: utf-8 -*-
"""
restore_config.py - 一键恢复被写坏的上游配置文件 (干净UTF-8, 无BOM, 无空字节)。

用法:
    python restore_config.py
"""

import os

CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "wsi_patch_selection_benchmark", "configs", "default_config.yaml",
)

CONTENT = """# ============================================================
# WSI Patch Selection Benchmark - Default Configuration
# ============================================================
# Copy this file and edit paths for your environment.
# CLI arguments override values in this file.

# ---------------------------------------------------------------------------
# Paths (REQUIRED: edit before running)
# ---------------------------------------------------------------------------
wsi_root: "D:/xcjhhh/TCGA_READ_WSI"
output_root: "outputs_read"

# ---------------------------------------------------------------------------
# WSI filtering
# ---------------------------------------------------------------------------
only_dx1: true

# ---------------------------------------------------------------------------
# Patch extraction parameters
# ---------------------------------------------------------------------------
patch_size: 1024
patches_per_case: 6
stride: 1024
ds_mask: 32

# ---------------------------------------------------------------------------
# Candidate pool parameters
# ---------------------------------------------------------------------------
preselect_topk: 96
min_tissue_ratio_preselect: 0.35
candidate_pool_size: 72
max_tries: 5000

# ---------------------------------------------------------------------------
# Scoring parameters
# ---------------------------------------------------------------------------
innovation_weight: 0.25
enable_gland_irregularity: true

# ---------------------------------------------------------------------------
# Diversity parameters
# ---------------------------------------------------------------------------
min_center_distance_ratio: 0.65
min_feature_distance: 0.10

# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------
seed: 2025

# ---------------------------------------------------------------------------
# Algorithms to run (comment out to disable)
# ---------------------------------------------------------------------------
enabled_samplers:
  - random
  - grid
  - largest_tissue
  - stratified
  - kmeans
  - yottixel
  - splice
  - sdm
  - sentinel

# ---------------------------------------------------------------------------
# Algorithm-specific parameters
# ---------------------------------------------------------------------------
stratified:
  n_bins_x: 3
  n_bins_y: 3

splice:
  redundancy_lambda: 2.0

yottixel:
  color_clusters: 8
  spatial_ratio_per_cluster: 0.25
  max_total_candidates: 400

sdm:
  morphology_seed_quality_weight: 0.25
  quality_tumor_weight: 0.45

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
visualization:
  enabled: true
  figures_dir: "figures"
  dpi: 300
  formats:
    - png
    - pdf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging:
  level: "INFO"
  file: "logs/benchmark.log"
  format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
"""

with open(CONFIG, "w", encoding="utf-8") as f:
    f.write(CONTENT)

print(f"[恢复] 配置文件已恢复为干净UTF-8: {CONFIG}")

# 验证
with open(CONFIG, "rb") as f:
    head = f.read(16)
print(f"[验证] 前16字节hex: {head.hex()}")
print(f"[验证] 应是以 23203d3d3d3d 开头 (= ' # ===')")
