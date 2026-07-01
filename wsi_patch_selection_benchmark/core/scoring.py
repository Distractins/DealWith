# -*- coding: utf-8 -*-
"""质量控制过滤与综合评分。

从原始脚本的 [B] 和 [E] 章节迁移而来。

提供：
- 三级QC过滤（严格 / 中等 / 宽松）
- 基线质量评分（质量指标的加权求和）
- 创新评分（肿瘤形态学）
- 最终评分 = 基线评分 + 创新权重 * 创新评分
- 0-1归一化，用于图块间 / 切片间的比较
"""

import logging
from typing import Dict

from common.constants import (
    QC_THRESHOLDS,
    BASELINE_SCORE_WEIGHTS,
    INNOVATION_SCORE_WEIGHTS,
    QC_QUALITY_WEIGHTS,
    TUMOR_MORPH_WEIGHTS,
    NORMALIZE_SCORE_RANGES,
)
from utils.math_utils import clip01, minmax_scale

logger = logging.getLogger(__name__)


# ============================================================
# 质量控制过滤
# ============================================================

def patch_passes_qc_by_level(metrics: Dict[str, float], level: str) -> bool:
    """检查图块是否通过指定严格程度的QC过滤。

    三级QC过滤，从最严格到最宽松：
    - strict：高组织含量、低白/暗像素、锐利、色彩丰富、高熵
    - medium：中等阈值
    - relaxed：宽松阈值（接受大多数非背景图块）

    Args:
        metrics: 来自 patch_quality_metrics() 的字典。
        level: 可选值为 'strict'、'medium'、'relaxed' 之一。

    Returns:
        如果图块通过指定等级的所有阈值则返回 True。
    """
    thresholds = QC_THRESHOLDS.get(level)
    if thresholds is None:
        logger.warning(f"Unknown QC level: {level}")
        return False

    if metrics["tissue_ratio"] < thresholds["tissue_ratio"]:
        return False
    if metrics["white_ratio"] > thresholds["white_ratio"]:
        return False
    if metrics["dark_ratio"] > thresholds["dark_ratio"]:
        return False
    if metrics["blur_score"] < thresholds["blur_score"]:
        return False
    if metrics["mean_sat"] < thresholds["mean_sat"]:
        return False
    if metrics["entropy"] < thresholds["entropy"]:
        return False

    return True


# ============================================================
# 评分计算
# ============================================================

def compute_baseline_score(metrics: Dict[str, float]) -> float:
    """根据图块指标计算基线质量评分。

    基线评分越高 = 图像质量越好（组织含量、清晰度、
    色彩、结构丰富度）。

    负贡献项：dark_ratio、stain_balance_penalty。

    Args:
        metrics: 来自 patch_quality_metrics() 的字典。

    Returns:
        基线质量评分（通常在 0~10 范围内）。
    """
    score = 0.0
    score += BASELINE_SCORE_WEIGHTS["tissue_ratio"] * metrics["tissue_ratio"]
    score += (
        BASELINE_SCORE_WEIGHTS["white_ratio_penalty"]
        * (1.0 - metrics["white_ratio"])
    )
    score += (
        BASELINE_SCORE_WEIGHTS["blur_score"]
        * min(metrics["blur_score"], BASELINE_SCORE_WEIGHTS["blur_score_cap"])
    )
    score += (
        BASELINE_SCORE_WEIGHTS["mean_sat"]
        * min(metrics["mean_sat"], BASELINE_SCORE_WEIGHTS["mean_sat_cap"])
    )
    score += (
        BASELINE_SCORE_WEIGHTS["colorfulness"]
        * min(metrics["colorfulness"], BASELINE_SCORE_WEIGHTS["colorfulness_cap"])
    )
    score += BASELINE_SCORE_WEIGHTS["entropy"] * metrics["entropy"]
    score += BASELINE_SCORE_WEIGHTS["edge_density"] * metrics["edge_density"]

    score -= BASELINE_SCORE_WEIGHTS["dark_ratio_penalty"] * metrics["dark_ratio"]
    score -= (
        BASELINE_SCORE_WEIGHTS["stain_balance_penalty"]
        * metrics["stain_balance_penalty"]
    )

    return float(score)


def compute_innovation_score(metrics: Dict[str, float]) -> float:
    """计算创新（肿瘤形态学）评分。

    创新评分越高 = 诊断学上更具相关性的形态学特征：
    肿瘤偏向性、腺体不规则性、组织异质性。

    Args:
        metrics: 来自 patch_quality_metrics() 的字典。

    Returns:
        创新评分（通常在 0~5.5 范围内）。
    """
    score = 0.0
    score += INNOVATION_SCORE_WEIGHTS["tumor_bias"] * metrics["tumor_bias"]
    score += (
        INNOVATION_SCORE_WEIGHTS["gland_irregularity"]
        * metrics["gland_irregularity"]
    )
    score += (
        INNOVATION_SCORE_WEIGHTS["heterogeneity_crc"]
        * metrics["heterogeneity_crc"]
    )
    return float(score)


def compute_final_score(
    metrics: Dict[str, float],
    innovation_weight: float = 0.25,
) -> Dict[str, float]:
    """计算最终评分 = 基线评分 + 创新权重 * 创新评分。

    同时计算所有评分的 0-1 归一化版本，以便进行切片间比较。

    Args:
        metrics: 来自 patch_quality_metrics() 的字典。
        innovation_weight: 创新评分在最终评分中的权重。
            对于不使用形态学的方法（如 Grid、Largest Tissue、
            Stratified），可设为 0.0。

    Returns:
        包含原始评分和归一化评分的字典。
    """
    baseline_score = compute_baseline_score(metrics)
    innovation_score = compute_innovation_score(metrics)
    final_score = baseline_score + innovation_weight * innovation_score

    score_norm_pack = normalize_scores(
        baseline_score=baseline_score,
        innovation_score=innovation_score,
        final_score=final_score,
        metrics=metrics,
    )

    return {
        "baseline_score": float(baseline_score),
        "innovation_score": float(innovation_score),
        "final_score": float(final_score),
        **score_norm_pack,
    }


# ============================================================
# 评分归一化
# ============================================================

def normalize_scores(
    baseline_score: float,
    innovation_score: float,
    final_score: float,
    metrics: Dict[str, float],
) -> Dict[str, float]:
    """使用固定的可解释范围将所有评分归一化到 [0, 1] 区间。

    使用固定范围（而非逐切片 min/max），使得来自
    不同切片和病例的评分保持直接可比性。

    Args:
        baseline_score: 原始基线质量评分。
        innovation_score: 原始创新评分。
        final_score: 原始最终评分。
        metrics: 来自 patch_quality_metrics() 的字典。

    Returns:
        归一化评分的字典。
    """
    ranges = NORMALIZE_SCORE_RANGES

    # 原始分数归一化
    baseline_norm = minmax_scale(
        baseline_score, ranges["baseline_score"][0], ranges["baseline_score"][1]
    )
    innovation_norm = minmax_scale(
        innovation_score, ranges["innovation_score"][0], ranges["innovation_score"][1]
    )
    final_norm = minmax_scale(
        final_score, ranges["final_score"][0], ranges["final_score"][1]
    )

    # 逐指标归一化
    tissue_norm = clip01(metrics.get("tissue_ratio", 0.0))
    white_good_norm = clip01(1.0 - metrics.get("white_ratio", 0.0))
    dark_good_norm = clip01(
        1.0 - min(metrics.get("dark_ratio", 0.0) / ranges["dark_ratio_divisor"], 1.0)
    )
    blur_norm = minmax_scale(
        metrics.get("blur_score", 0.0), ranges["blur_score"][0], ranges["blur_score"][1]
    )
    sat_norm = minmax_scale(
        metrics.get("mean_sat", 0.0), ranges["mean_sat"][0], ranges["mean_sat"][1]
    )
    color_norm = minmax_scale(
        metrics.get("colorfulness", 0.0),
        ranges["colorfulness"][0],
        ranges["colorfulness"][1],
    )
    entropy_norm = minmax_scale(
        metrics.get("entropy", 0.0), ranges["entropy"][0], ranges["entropy"][1]
    )
    edge_norm = clip01(metrics.get("edge_density", 0.0) / ranges["edge_density_divisor"])
    tumor_bias_norm = minmax_scale(
        metrics.get("tumor_bias", 0.0),
        ranges["tumor_bias"][0],
        ranges["tumor_bias"][1],
    )
    gland_irregularity_norm = clip01(metrics.get("gland_irregularity", 0.0))
    heterogeneity_norm = clip01(metrics.get("heterogeneity_crc", 0.0))

    # 综合归一化得分
    qc_quality_norm = clip01(
        QC_QUALITY_WEIGHTS["tissue_norm"] * tissue_norm
        + QC_QUALITY_WEIGHTS["white_good_norm"] * white_good_norm
        + QC_QUALITY_WEIGHTS["dark_good_norm"] * dark_good_norm
        + QC_QUALITY_WEIGHTS["blur_norm"] * blur_norm
        + QC_QUALITY_WEIGHTS["sat_norm"] * sat_norm
        + QC_QUALITY_WEIGHTS["color_norm"] * color_norm
        + QC_QUALITY_WEIGHTS["entropy_norm"] * entropy_norm
        + QC_QUALITY_WEIGHTS["edge_norm"] * edge_norm
    )

    tumor_morph_norm = clip01(
        TUMOR_MORPH_WEIGHTS["tumor_bias_norm"] * tumor_bias_norm
        + TUMOR_MORPH_WEIGHTS["gland_irregularity_norm"] * gland_irregularity_norm
        + TUMOR_MORPH_WEIGHTS["heterogeneity_norm"] * heterogeneity_norm
    )

    return {
        "baseline_score_norm": baseline_norm,
        "innovation_score_norm": innovation_norm,
        "final_score_norm": final_norm,
        "tissue_norm": tissue_norm,
        "white_good_norm": white_good_norm,
        "dark_good_norm": dark_good_norm,
        "blur_norm": blur_norm,
        "sat_norm": sat_norm,
        "color_norm": color_norm,
        "entropy_norm": entropy_norm,
        "edge_norm": edge_norm,
        "tumor_bias_norm": tumor_bias_norm,
        "gland_irregularity_norm": gland_irregularity_norm,
        "heterogeneity_norm": heterogeneity_norm,
        "qc_quality_norm": qc_quality_norm,
        "tumor_morph_norm": tumor_morph_norm,
    }
