# -*- coding: utf-8 -*-
"""WSI补丁选择基准测试的共享常量。

原始脚本中的所有魔数均提取于此。
"""

# ---------------------------------------------------------------------------
# 各级别的QC阈值
# ---------------------------------------------------------------------------
QC_THRESHOLDS = {
    "strict": {
        "tissue_ratio": 0.50,
        "white_ratio": 0.45,
        "dark_ratio": 0.20,
        "blur_score": 35.0,
        "mean_sat": 12.0,
        "entropy": 4.0,
    },
    "medium": {
        "tissue_ratio": 0.38,
        "white_ratio": 0.62,
        "dark_ratio": 0.25,
        "blur_score": 20.0,
        "mean_sat": 8.0,
        "entropy": 3.4,
    },
    "relaxed": {
        "tissue_ratio": 0.25,
        "white_ratio": 0.78,
        "dark_ratio": 0.35,
        "blur_score": 8.0,
        "mean_sat": 4.0,
        "entropy": 2.8,
    },
}

# ---------------------------------------------------------------------------
# 基线评分权重（compute_baseline_score）
# ---------------------------------------------------------------------------
BASELINE_SCORE_WEIGHTS = {
    "tissue_ratio": 3.0,
    "white_ratio_penalty": 1.6,
    "blur_score": 0.0022,
    "blur_score_cap": 1000.0,
    "mean_sat": 0.010,
    "mean_sat_cap": 120.0,
    "colorfulness": 0.010,
    "colorfulness_cap": 120.0,
    "entropy": 0.12,
    "edge_density": 1.0,
    "dark_ratio_penalty": 1.0,
    "stain_balance_penalty": 1.2,
}

# ---------------------------------------------------------------------------
# 创新评分权重（compute_innovation_score）
# ---------------------------------------------------------------------------
INNOVATION_SCORE_WEIGHTS = {
    "tumor_bias": 1.0,
    "gland_irregularity": 0.35,
    "heterogeneity_crc": 0.15,
}

# ---------------------------------------------------------------------------
# 肿瘤偏倚权重（compute_tumor_bias）
# ---------------------------------------------------------------------------
TUMOR_BIAS_WEIGHTS = {
    "nuclear_like_density": 1.8,
    "heterogeneity_crc": 1.0,
    "nuclear_edge_density": 1.4,
    "gland_irregularity": 0.8,
}

# ---------------------------------------------------------------------------
# QC质量归一化权重（normalize_scores）
# ---------------------------------------------------------------------------
QC_QUALITY_WEIGHTS = {
    "tissue_norm": 0.22,
    "white_good_norm": 0.16,
    "dark_good_norm": 0.08,
    "blur_norm": 0.14,
    "sat_norm": 0.12,
    "color_norm": 0.08,
    "entropy_norm": 0.10,
    "edge_norm": 0.10,
}

# ---------------------------------------------------------------------------
# 肿瘤形态归一化权重（normalize_scores）
# ---------------------------------------------------------------------------
TUMOR_MORPH_WEIGHTS = {
    "tumor_bias_norm": 0.60,
    "gland_irregularity_norm": 0.25,
    "heterogeneity_norm": 0.15,
}

# ---------------------------------------------------------------------------
# 评分归一化范围（minmax_scale: [xmin, xmax]）
# ---------------------------------------------------------------------------
NORMALIZE_SCORE_RANGES = {
    "baseline_score": (0.0, 10.0),
    "innovation_score": (0.0, 5.5),
    "final_score": (0.0, 11.5),
    "blur_score": (0.0, 800.0),
    "mean_sat": (0.0, 120.0),
    "colorfulness": (0.0, 120.0),
    "entropy": (0.0, 8.0),
    "edge_density_divisor": 0.25,
    "tumor_bias": (0.0, 5.0),
    "dark_ratio_divisor": 0.35,
}

# ---------------------------------------------------------------------------
# 特征向量索引（patch_feature_vector）
# ---------------------------------------------------------------------------
FEATURE_VECTOR_INDICES = {
    "tissue_ratio": 0,
    "white_ratio": 1,
    "blur_score": 2,
    "mean_sat": 3,
    "colorfulness": 4,
    "entropy": 5,
    "edge_density": 6,
    "tumor_bias": 7,
}

FEATURE_VECTOR_DIVISORS = {
    "tissue_ratio": 1.0,
    "white_ratio": 1.0,
    "blur_score": 1000.0,
    "mean_sat": 255.0,
    "colorfulness": 100.0,
    "entropy": 8.0,
    "edge_density": 1.0,
    "tumor_bias": 10.0,
}

# ---------------------------------------------------------------------------
# 方法颜色，用于一致的视觉呈现
# ---------------------------------------------------------------------------
METHOD_COLORS = {
    "random": "#E41A1C",
    "grid": "#377EB8",
    "largest_tissue": "#4DAF4A",
    "stratified": "#984EA3",
    "kmeans": "#FF7F00",
    "yottixel": "#FFFF33",
    "splice": "#A65628",
    "sdm": "#F781BF",
    "sentinel": "#66C2A5",
}
