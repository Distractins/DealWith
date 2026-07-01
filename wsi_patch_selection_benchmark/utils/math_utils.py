# -*- coding: utf-8 -*-
"""用于评分和相似度计算的数学工具函数。

从原始脚本的归一化和距离函数迁移而来。
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def clip01(x: float) -> float:
    """将值截断到 [0, 1] 范围内。

    Args:
        x: 输入值。

    Returns:
        被限制在 [0, 1] 范围内的值。
    """
    return float(np.clip(x, 0.0, 1.0))


def minmax_scale(x: float, xmin: float, xmax: float) -> float:
    """对值进行最大-最小归一化，映射到 [0, 1] 范围内。

    Args:
        x: 输入值。
        xmin: 预期范围的最小值。
        xmax: 预期范围的最大值。

    Returns:
        [0, 1] 范围内的归一化值。如果 xmax <= xmin，则返回 0。
    """
    if xmax <= xmin:
        return 0.0
    return clip01((x - xmin) / (xmax - xmin))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量之间的余弦相似度。

    Args:
        a: 第一个向量。
        b: 第二个向量。

    Returns:
        [-1, 1] 范围内的余弦相似度。
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def pairwise_l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量之间的 L2（欧几里得）距离。

    Args:
        a: 第一个向量。
        b: 第二个向量。

    Returns:
        欧几里得距离。
    """
    return float(np.linalg.norm(a - b))
