# WSI Patch Selection Benchmark

A unified, modular, and extensible benchmark framework for **Representative Patch Selection** in Whole Slide Images (WSI).

## Overview

This project compares 9 patch selection algorithms on histopathology WSIs under a **fair, unified protocol**:

- **Same WSI** for all algorithms
- **Same Patch Size** (1024×1024 px)
- **Same K** (number of patches per case)
- **Same Candidate Pool** (no re-scanning per algorithm)
- **Same Evaluation Metrics**

## Algorithms

| # | Algorithm | Sampler | Description |
|---|-----------|---------|-------------|
| 1 | Random | `random` | Uniform random selection (baseline) |
| 2 | Grid | `grid` | Quality top-K from grid scan + diversity |
| 3 | Largest Tissue | `largest_tissue` | Restricted to largest tissue component |
| 4 | Stratified Spatial | `stratified` | Per-bin best + cross-bin diversity |
| 5 | K-Means | `kmeans` | Feature-space clustering + medoids |
| 6 | Yottixel-inspired | `yottixel` | Color + spatial two-level mosaic |
| 7 | SPLICE-inspired | `splice` | Greedy with cosine redundancy penalty |
| 8 | SDM | `sdm` | Distinct Morphology: seed + medoid |
| 9 | Sentinel (SAPS) | `sentinel` | Random candidates + 3-tier QC + diversity |

## Project Structure

```
wsi_patch_selection_benchmark/
├── run_patch_selection.py      # Main entry point
├── configs/                    # YAML config + loader
├── datasets/                   # WSI file discovery & reading
├── common/                     # Data structures (dataclasses, enums, constants)
├── utils/                      # Logger, seed, file I/O, math, timer
├── core/                       # Tissue mask, features, scoring, diversity, pool
├── samplers/                   # 9 patch selection algorithms
├── metrics/                    # Spatial coverage, diversity, redundancy
├── evaluation/                 # Batch stats, CSV, report generation
├── visualization/              # Charts, overlays, paper figures
├── reports/                    # Auto-generated reports
└── outputs/                    # Patches, CSVs, figures
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Paths

Edit `configs/default_config.yaml` to set your WSI root directory:

```yaml
wsi_root: "E:/gdc/TCGA_COAD_WSI"   # Path to your .svs files
output_root: "outputs"               # Where results go
```

### 3. Run the Benchmark

```bash
python run_patch_selection.py
```

Or with CLI overrides:

```bash
python run_patch_selection.py \
    --wsi-root /data/TCGA_COAD_WSI \
    --output-root ./results \
    --patches-per-case 6 \
    --patch-size 1024 \
    --seed 2025
```

## Configuration

All parameters are in `configs/default_config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_size` | 1024 | Patch side length (px) |
| `patches_per_case` | 6 | Target patches per patient (K) |
| `seed` | 2025 | Global random seed |
| `stride` | 1024 | Grid scan stride |
| `ds_mask` | 32 | Tissue mask downsampling |
| `innovation_weight` | 0.25 | Innovation score weight |
| `min_center_distance_ratio` | 0.65 | Spatial diversity threshold |
| `min_feature_distance` | 0.10 | Feature diversity threshold |

## Output Files

After running, `outputs/` contains:

```
outputs/
├── patches/
│   ├── random/          # Saved PNG patches per algorithm
│   ├── grid/
│   └── ...
├── csv/
│   ├── patch_metrics.csv         # Per-patch quality & scores
│   ├── slide_metrics.csv         # Per-slide summary
│   ├── method_summary.csv        # Algorithm comparison
│   └── paper_tables.csv          # Publication-ready table
├── figures/
│   ├── bar_*.png                 # Bar charts per metric
│   ├── radar_all.png             # Multi-metric radar
│   ├── heatmap.png               # Method × metric heatmap
│   └── paper_figures/            # Publication-quality figures
└── reports/
    └── report.md                 # Auto-generated experiment report
```

## Adding a New Sampler

1. Create `samplers/your_sampler.py`:

```python
from samplers.base_sampler import BaseSampler
from samplers import register_sampler

@register_sampler
class YourSampler(BaseSampler):
    name = "Your Algorithm"

    @staticmethod
    def algorithm_name() -> str:
        return "your_algo"

    def select_patches(self, candidate_pool, num_patches):
        # Your selection logic here
        return selected[:num_patches]
```

2. Add `your_algo` to `configs/default_config.yaml` under `enabled_samplers`.

## Metrics

Four representative metrics evaluate patch selection quality:

| Metric | Description |
|--------|-------------|
| Spatial Coverage | Convex hull area ratio of selected patch centers |
| Feature Diversity | Mean pairwise Euclidean distance in feature space |
| Redundancy Rate | Fraction of patch pairs below similarity threshold |
| Covered Region Count | Number of spatial bins containing ≥1 patch |

## Design Principles

1. **Separation of Concerns**: Datasets read WSIs. Core computes features. Samplers select patches. Metrics evaluate. Each module does one thing.
2. **No Hardcoded Paths**: All paths from YAML config or CLI.
3. **Unified Candidate Pool**: WSI scanned once, shared across all samplers.
4. **PEP8 Compliance**: Type annotations, docstrings, logging throughout.
5. **Reproducibility**: Deterministic seeds per slide, atomic case commits.

## References

This project is based on the TCGA-COAD WSI dataset. The sentinel algorithm (SAPS) introduces tumor-biased selection for diagnostically relevant patch extraction.

## License

Research use only. For academic purposes, please cite the original papers for each algorithm.
