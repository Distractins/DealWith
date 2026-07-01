# -*- coding: utf-8 -*-
"""图像块特征提取函数。

包含所有质量指标、肿瘤形态学特征以及特征向量构建例程，
均从原始脚本的第 [D] 节中提取。

涵盖 14+ 个特征函数：
- 基础质量：tissue_ratio、white_ratio、dark_ratio
- 清晰度：blur_score（拉普拉斯方差）
- 颜色：mean_saturation、colorfulness_score、stain_balance_penalty
- 信息量：entropy_score（香农熵）
- 结构：edge_density_score（Canny）、nuclear_edge_density
- 肿瘤形态学：nuclear_like_density、heterogeneity_score_crc、
  gland_irregularity_score、compute_tumor_bias
"""

import logging
from typing import Dict

import numpy as np
import cv2

from common.constants import (
    TUMOR_BIAS_WEIGHTS,
    FEATURE_VECTOR_DIVISORS,
)

logger = logging.getLogger(__name__)


# ============================================================
# 基础质量指标
# ============================================================

def tissue_ratio_in_patch(
    mask: np.ndarray,
    x0: int,
    y0: int,
    patch_size: int,
    ds: int,
) -> float:
    """计算图像块边界框内组织像素的占比。

    使用降采样后的组织掩膜以提高计算效率。

    Args:
        mask: 组织掩膜（二值，uint8）。
        x0: 第0层坐标下的左上角 X。
        y0: 第0层坐标下的左上角 Y。
        patch_size: 第0层像素单位下的图像块边长。
        ds: 掩膜的降采样倍率。

    Returns:
        组织占比，取值范围 [0, 1]。
    """
    x1, y1 = x0 + patch_size, y0 + patch_size
    mx0, my0 = x0 // ds, y0 // ds
    mx1, my1 = x1 // ds, y1 // ds

    mx0 = max(0, mx0)
    my0 = max(0, my0)
    mx1 = min(mask.shape[1], mx1)
    my1 = min(mask.shape[0], my1)

    if mx1 <= mx0 or my1 <= my0:
        return 0.0

    region = mask[my0:my1, mx0:mx1]
    return float(region.mean())


def white_ratio_in_patch(rgb_patch: np.ndarray, white_thresh: int = 220) -> float:
    """超过亮度阈值的像素占比（近白色/背景区域）。

    Args:
        rgb_patch: RGB 图像数组。
        white_thresh: 用于判定白色像素的灰度值阈值。

    Returns:
        白色像素占比，取值范围 [0, 1]。
    """
    gray = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2GRAY)
    return float((gray >= white_thresh).mean())


def dark_ratio_in_patch(rgb_patch: np.ndarray, dark_thresh: int = 25) -> float:
    """低于暗度阈值的像素占比（过度染色/伪影区域）。

    Args:
        rgb_patch: RGB 图像数组。
        dark_thresh: 用于判定暗像素的灰度值阈值。

    Returns:
        暗像素占比，取值范围 [0, 1]。
    """
    gray = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2GRAY)
    return float((gray <= dark_thresh).mean())


# ============================================================
# 清晰度
# ============================================================

def blur_score(rgb_patch: np.ndarray) -> float:
    """通过拉普拉斯方差计算模糊度/清晰度。

    值越高 = 越清晰（模糊程度越低）。

    Args:
        rgb_patch: RGB 图像数组。

    Returns:
        拉普拉斯方差。
    """
    gray = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ============================================================
# 颜色指标
# ============================================================

def mean_saturation(rgb_patch: np.ndarray) -> float:
    """HSV 颜色空间中的平均饱和度。

    Args:
        rgb_patch: RGB 图像数组。

    Returns:
        平均饱和度值。
    """
    hsv = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean())


def colorfulness_score(rgb_patch: np.ndarray) -> float:
    """计算色彩丰富度指标（Hasler & Süsstrunk, 2003）。

    基于对立色彩空间统计量（rg 与 yb 通道）。

    Args:
        rgb_patch: RGB 图像数组。

    Returns:
        色彩丰富度得分（值越高 = 色彩越丰富）。
    """
    patch = rgb_patch.astype(np.float32)
    rg = np.abs(patch[:, :, 0] - patch[:, :, 1])
    yb = np.abs(0.5 * (patch[:, :, 0] + patch[:, :, 1]) - patch[:, :, 2])

    std_rg, mean_rg = float(np.std(rg)), float(np.mean(rg))
    std_yb, mean_yb = float(np.std(yb)), float(np.mean(yb))

    return float(
        np.sqrt(std_rg ** 2 + std_yb ** 2)
        + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)
    )


def stain_balance_penalty(rgb_patch: np.ndarray) -> float:
    """染色不均衡的惩罚项（RGB 各通道均值方差较大时产生惩罚）。

    Args:
        rgb_patch: RGB 图像数组。

    Returns:
        染色均衡惩罚值（值越低 = 越均衡）。
    """
    mean_rgb = rgb_patch.reshape(-1, 3).mean(axis=0)
    penalty = np.std(mean_rgb) / (np.mean(mean_rgb) + 1e-6)
    return float(penalty)


# ============================================================
# 信息量
# ============================================================

def entropy_score(gray_patch: np.ndarray) -> float:
    """计算灰度直方图的香农熵。

    Args:
        gray_patch: 灰度图像数组。

    Returns:
        熵值，单位为比特。
    """
    hist = cv2.calcHist([gray_patch], [0], None, [256], [0, 256]).flatten()
    prob = hist / (hist.sum() + 1e-8)
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


# ============================================================
# 边缘/结构指标
# ============================================================

def edge_density_score(gray_patch: np.ndarray) -> float:
    """Canny 边缘检测器检测到的边缘像素占比。

    Args:
        gray_patch: 灰度图像数组。

    Returns:
        边缘密度，取值范围 [0, 1]。
    """
    edges = cv2.Canny(gray_patch, 50, 150)
    return float((edges > 0).mean())


def nuclear_edge_density(rgb_patch: np.ndarray) -> float:
    """使用 CLAHE 增强后的边缘密度。

    CLAHE 可增强细胞核边界，因此该指标可作为图像块中
    细胞核密度的近似代理指标。

    Args:
        rgb_patch: RGB 图像数组。

    Returns:
        增强后的边缘密度，取值范围 [0, 1]。
    """
    gray = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    edges = cv2.Canny(gray_enhanced, 80, 200)
    return float((edges > 0).mean())


# ============================================================
# 肿瘤形态学特征
# ============================================================

def nuclear_like_density(rgb_patch: np.ndarray) -> float:
    """使用 HSV 颜色阈值法估算细胞核密度。

    目标为苏木精染色的细胞核：色调 100-170°（蓝紫色），
    中等饱和度，以及较低的明度（较暗）。

    Args:
        rgb_patch: RGB 图像数组。

    Returns:
        类细胞核像素占比，取值范围 [0, 1]。
    """
    hsv = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    nucleus_mask = (
        (h >= 100) & (h <= 170)
        & (s >= 25)
        & (v <= 170)
    )
    return float(nucleus_mask.mean())


def heterogeneity_score_crc(rgb_patch: np.ndarray) -> float:
    """通过灰度标准差评估组织异质性。

    标准差越高 = 组织纹理变化越大（常见于结直肠癌 CRC）。

    Args:
        rgb_patch: RGB 图像数组。

    Returns:
        异质性得分，取值范围 [0, ~1]（上限为 std/64）。
    """
    gray = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2GRAY)
    std_val = float(np.std(gray))
    return min(std_val / 64.0, 1.0)


def gland_irregularity_score(rgb_patch: np.ndarray) -> float:
    """通过结构张量相干性评估腺体结构不规则程度。

    低相干性 = 腺体边界不规则（常见于结直肠癌 CRC）。
    高相干性 = 组织排列规则有序（例如正常结肠组织）。

    为提高计算效率，图像块会被缩小至 256x256。

    Args:
        rgb_patch: RGB 图像数组。

    Returns:
        不规则性得分，取值范围 [0, 1]（值越高 = 越不规则）。
    """
    h, w = rgb_patch.shape[:2]
    if h > 256 and w > 256:
        small = cv2.resize(rgb_patch, (256, 256), interpolation=cv2.INTER_AREA)
    else:
        small = rgb_patch

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 通过 Sobel 梯度计算结构张量分量
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    j11 = float(np.sum(gx * gx))
    j22 = float(np.sum(gy * gy))
    j12 = float(np.sum(gx * gy))

    denom = j11 + j22 + 1e-6
    coherence = np.sqrt((j11 - j22) ** 2 + 4.0 * (j12 ** 2)) / denom

    irregularity = 1.0 - float(np.clip(coherence, 0.0, 1.0))
    return irregularity


def compute_tumor_bias(
    rgb_patch: np.ndarray,
    enable_gland_irregularity: bool = True,
) -> Dict[str, float]:
    """从形态学特征计算肿瘤偏向性得分。

    将以下特征组合为一个单一的肿瘤相关性得分：
    细胞核密度、异质性、核边缘密度，以及可选的腺体不规则性。

    Args:
        rgb_patch: RGB 图像数组。
        enable_gland_irregularity: 若为 False，则跳过腺体不规则性计算
            （将其置为 0）。对于非形态学采样器可提升运行速度。

    Returns:
        包含各特征值以及组合后的 'tumor_bias' 的字典。
    """
    nld = nuclear_like_density(rgb_patch)
    het = heterogeneity_score_crc(rgb_patch)
    ned = nuclear_edge_density(rgb_patch)

    if enable_gland_irregularity:
        gir = gland_irregularity_score(rgb_patch)
    else:
        gir = 0.0

    tumor_bias = (
        TUMOR_BIAS_WEIGHTS["nuclear_like_density"] * nld
        + TUMOR_BIAS_WEIGHTS["heterogeneity_crc"] * het
        + TUMOR_BIAS_WEIGHTS["nuclear_edge_density"] * ned
        + (TUMOR_BIAS_WEIGHTS["gland_irregularity"] * gir
           if enable_gland_irregularity else 0.0)
    )

    return {
        "nuclear_like_density": nld,
        "heterogeneity_crc": het,
        "nuclear_edge_density": ned,
        "gland_irregularity": gir,
        "tumor_bias": float(tumor_bias),
    }


# ============================================================
# 聚合指标
# ============================================================

def patch_quality_metrics(
    rgb_patch: np.ndarray,
    tissue_ratio: float,
    enable_gland_irregularity: bool = True,
) -> Dict[str, float]:
    """计算单个图像块的所有质量和形态学指标。

    这是对单个图像块进行特征提取的主入口函数。
    在候选池构建（第 2 阶段）期间被调用。

    Args:
        rgb_patch: RGB 图像数组（patch_size x patch_size x 3）。
        tissue_ratio: 从掩膜中预先计算得到的组织占比。
        enable_gland_irregularity: 若为 False，则跳过计算开销较高的腺体计算。

    Returns:
        包含所有指标（质量 + 肿瘤形态学）的字典。
    """
    gray = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2GRAY)

    metrics = {
        "tissue_ratio": tissue_ratio,
        "white_ratio": white_ratio_in_patch(rgb_patch),
        "dark_ratio": dark_ratio_in_patch(rgb_patch),
        "blur_score": blur_score(rgb_patch),
        "mean_sat": mean_saturation(rgb_patch),
        "colorfulness": colorfulness_score(rgb_patch),
        "entropy": entropy_score(gray),
        "edge_density": edge_density_score(gray),
        "stain_balance_penalty": stain_balance_penalty(rgb_patch),
    }
    metrics.update(
        compute_tumor_bias(
            rgb_patch,
            enable_gland_irregularity=enable_gland_irregularity,
        )
    )
    return metrics


# ============================================================
# 特征向量
# ============================================================

def patch_feature_vector(metrics: Dict[str, float]) -> np.ndarray:
    """从指标字典构建标准的 8 维特征向量。

    供 KMeans、SPLICE、多样性选择以及通用特征空间操作使用。

    各分量：
        0: tissue_ratio
        1: white_ratio
        2: blur_score / 1000
        3: mean_sat / 255
        4: colorfulness / 100
        5: entropy / 8
        6: edge_density
        7: tumor_bias / 10

    Args:
        metrics: 来自 patch_quality_metrics() 的字典。

    Returns:
        归一化后的 8 维 float32 特征向量。
    """
    return np.array(
        [
            metrics["tissue_ratio"],
            metrics["white_ratio"],
            metrics["blur_score"] / FEATURE_VECTOR_DIVISORS["blur_score"],
            metrics["mean_sat"] / FEATURE_VECTOR_DIVISORS["mean_sat"],
            metrics["colorfulness"] / FEATURE_VECTOR_DIVISORS["colorfulness"],
            metrics["entropy"] / FEATURE_VECTOR_DIVISORS["entropy"],
            metrics["edge_density"],
            metrics["tumor_bias"] / FEATURE_VECTOR_DIVISORS["tumor_bias"],
        ],
        dtype=np.float32,
    )


def morphology_feature_vector(metrics: Dict[str, float]) -> np.ndarray:
    """构建扩展的 12 维形态学特征向量。

    供 SDM 采样器用于基于形态学的种子点选择与分组。
    包含所有标准特征，外加细胞核密度和边缘密度。

    Args:
        metrics: 来自 patch_quality_metrics() 的字典。

    Returns:
        12 维 float32 特征向量。
    """
    return np.array(
        [
            metrics["tissue_ratio"],
            metrics["white_ratio"],
            metrics["blur_score"] / FEATURE_VECTOR_DIVISORS["blur_score"],
            metrics["mean_sat"] / FEATURE_VECTOR_DIVISORS["mean_sat"],
            metrics["colorfulness"] / FEATURE_VECTOR_DIVISORS["colorfulness"],
            metrics["entropy"] / FEATURE_VECTOR_DIVISORS["entropy"],
            metrics["edge_density"],
            metrics["tumor_bias"] / FEATURE_VECTOR_DIVISORS["tumor_bias"],
            metrics["nuclear_like_density"],
            metrics["heterogeneity_crc"],
            metrics["nuclear_edge_density"],
            metrics["gland_irregularity"],
        ],
        dtype=np.float32,
    )


def rgb_hist_24bins(rgb_patch: np.ndarray) -> np.ndarray:
    """计算 24 区间的 RGB 直方图（每通道 8 个区间），经 L1 归一化。

    供受 Yottixel 启发的采样器用于基于颜色的聚类。

    Args:
        rgb_patch: RGB 图像数组。

    Returns:
        24 维 float32 直方图向量，经 L1 归一化。
    """
    hist = np.zeros(24, dtype=np.float32)
    for c in range(3):
        channel_hist = cv2.calcHist(
            [rgb_patch], [c], None, [8], [0, 256]
        ).flatten()
        hist[c * 8: (c + 1) * 8] = channel_hist

    # L1 归一化
    total = hist.sum()
    if total > 0:
        hist /= total

    return hist
