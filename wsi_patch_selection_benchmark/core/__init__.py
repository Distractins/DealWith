# -*- coding: utf-8 -*-
"""WSI 图像块选择基准测试的核心处理模块。

所有采样器所依赖的基础能力均位于此处：
- 组织掩膜（Tissue Mask）：build_hybrid_tissue_mask、extract_largest_connected_component
- 特征提取（Feature Extraction）：14+ 项质量/肿瘤形态学特征
- 评分（Scoring）：QC 过滤、基线/创新/最终评分的计算
- 多样性（Diversity）：空间与特征空间的多样性选择
- 候选池（Candidate Pool）：所有采样器共享的统一候选池构建器
- 图像块 IO（Patch IO）：原子化的病例级图像块保存
"""

from core.tissue_mask import (
    build_hybrid_tissue_mask,
    extract_largest_connected_component,
)
from core.feature_extraction import (
    tissue_ratio_in_patch,
    white_ratio_in_patch,
    dark_ratio_in_patch,
    blur_score,
    mean_saturation,
    colorfulness_score,
    entropy_score,
    edge_density_score,
    stain_balance_penalty,
    nuclear_like_density,
    heterogeneity_score_crc,
    nuclear_edge_density,
    gland_irregularity_score,
    compute_tumor_bias,
    patch_quality_metrics,
    patch_feature_vector,
    morphology_feature_vector,
    rgb_hist_24bins,
)
from core.scoring import (
    patch_passes_qc_by_level,
    compute_baseline_score,
    compute_innovation_score,
    compute_final_score,
    normalize_scores,
)
from core.diversity import (
    far_enough,
    feature_far_enough,
    select_diverse_topk,
)
from core.candidate_pool import (
    CandidatePool,
    CandidatePoolBuilder,
    generate_grid_positions,
    merge_candidates_by_priority,
)
from core.patch_io import (
    commit_case_patches_atomic,
    save_patch_image,
)

__all__ = [
    # 组织掩膜
    "build_hybrid_tissue_mask",
    "extract_largest_connected_component",
    # 特征提取
    "tissue_ratio_in_patch",
    "white_ratio_in_patch",
    "dark_ratio_in_patch",
    "blur_score",
    "mean_saturation",
    "colorfulness_score",
    "entropy_score",
    "edge_density_score",
    "stain_balance_penalty",
    "nuclear_like_density",
    "heterogeneity_score_crc",
    "nuclear_edge_density",
    "gland_irregularity_score",
    "compute_tumor_bias",
    "patch_quality_metrics",
    "patch_feature_vector",
    "morphology_feature_vector",
    "rgb_hist_24bins",
    # 评分
    "patch_passes_qc_by_level",
    "compute_baseline_score",
    "compute_innovation_score",
    "compute_final_score",
    "normalize_scores",
    # 多样性
    "far_enough",
    "feature_far_enough",
    "select_diverse_topk",
    # 候选池
    "CandidatePool",
    "CandidatePoolBuilder",
    "generate_grid_positions",
    "merge_candidates_by_priority",
    # 图像块 IO
    "commit_case_patches_atomic",
    "save_patch_image",
]
