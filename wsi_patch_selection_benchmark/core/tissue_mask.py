# -*- coding: utf-8 -*-
"""WSI切片的组织掩膜生成。

通过结合饱和度（HSV）、光密度（OD）和亮度阈值处理，
并进行形态学清理，构建混合组织掩膜。

从原始脚本的第[C]部分迁移而来。
"""

import logging
from typing import Tuple, Optional

import numpy as np
import cv2
import openslide

from utils.image_utils import rgb_to_od

logger = logging.getLogger(__name__)


def build_hybrid_tissue_mask(
    slide: openslide.OpenSlide,
    downsample: int = 32,
) -> Tuple[np.ndarray, int]:
    """为WSI切片构建混合组织掩膜。

    结合三种信号：
    1. 饱和度（HSV）—— 大津阈值
    2. 光密度（OD）—— 大津阈值
    3. 亮度掩膜（灰度 < 235）

    然后应用形态学开/闭运算，并移除小于图像面积0.05%的小连通区域。

    Args:
        slide: 一个openslide.OpenSlide对象。
        downsample: 掩膜的下采样因子（值越大越快，但越粗糙）。

    Returns:
        (二值掩膜uint8数组, 下采样因子) 组成的元组。
        mask[y, x] 对于组织区域为1，对于背景区域为0。
    """
    w0, h0 = slide.dimensions
    tw, th = max(1, w0 // downsample), max(1, h0 // downsample)

    # 获取下采样缩略图
    thumb = slide.get_thumbnail((tw, th)).convert("RGB")
    img = np.array(thumb)

    # ---- 饱和度通道 ----
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    sat_blur = cv2.GaussianBlur(sat, (5, 5), 0)
    _, sat_mask = cv2.threshold(sat_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ---- 光密度 ----
    od = rgb_to_od(img)
    od_mean = od.mean(axis=2)
    od_norm = cv2.normalize(od_mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    od_blur = cv2.GaussianBlur(od_norm, (5, 5), 0)
    _, od_mask = cv2.threshold(od_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ---- 亮度掩膜 ----
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bright_mask = (gray < 235).astype(np.uint8) * 255

    # ---- 融合 ----
    fused = cv2.bitwise_and(cv2.bitwise_or(sat_mask, od_mask), bright_mask)

    # ---- 形态学清理 ----
    kernel1 = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((7, 7), np.uint8)
    fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, kernel1, iterations=1)
    fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, kernel2, iterations=2)

    # ---- 移除小连通区域 ----
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (fused > 0).astype(np.uint8), connectivity=8
    )
    cleaned = np.zeros_like(fused, dtype=np.uint8)
    min_area = max(16, int(0.0005 * fused.shape[0] * fused.shape[1]))

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 1

    tissue_pct = 100.0 * cleaned.sum() / cleaned.size
    logger.debug(
        f"Tissue mask: {cleaned.sum()} tissue pixels "
        f"({tissue_pct:.1f}% of thumbnail), "
        f"downsample={downsample}"
    )

    return cleaned.astype(np.uint8), downsample


def extract_largest_connected_component(
    mask: np.ndarray,
) -> Optional[np.ndarray]:
    """从二值掩膜中提取最大连通区域。

    由最大组织成分采样器使用，以将候选区域生成限制在主要组织区域。

    Args:
        mask: uint8 ndarray类型的二值掩膜（0=背景，1=组织）。

    Returns:
        仅包含最大连通区域的二值掩膜，
        如果输入掩膜为空则返回None。
    """
    if mask.sum() == 0:
        logger.warning("Input mask is empty, cannot extract largest component")
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    if num_labels <= 1:
        return mask.copy()

    # 找到最大连通区域（跳过标签0 = 背景）
    areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num_labels)]
    if not areas:
        return mask.copy()

    largest_label = max(areas, key=lambda x: x[0])[1]
    largest_mask = np.zeros_like(mask, dtype=np.uint8)
    largest_mask[labels == largest_label] = 1

    logger.debug(
        f"Largest component: {largest_mask.sum()} pixels "
        f"(out of {mask.sum()} total tissue)"
    )

    return largest_mask
