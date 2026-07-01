# -*- coding: utf-8 -*-
"""确定性随机种子工具。

通过管理全局种子并提供每个切片的确定性随机数生成器，
确保跨运行的可复现结果。
"""

import hashlib
import random
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """为 Python、NumPy 和 OpenCV 设置全局随机种子。

    Args:
        seed: 整型种子值。
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import cv2
        cv2.setRNGSeed(seed)
    except Exception:
        pass  # cv2.setRNGSeed 可能并非在所有构建中都可用

    logger.info(f"Global random seed set to {seed}")


def stable_int_hash(text: str, mod: int = 10 ** 9) -> int:
    """从文本字符串生成稳定的整型哈希值。

    用于推导每个切片/病例的确定性种子。

    Args:
        text: 输入字符串（例如，切片基础名称）。
        mod: 输出的模数。

    Returns:
        在 [0, mod) 范围内的整型哈希值。
    """
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:12], 16) % mod


def deterministic_rng_for_slide(
    slide_base: str,
    base_seed: int,
    mod: int = 10 ** 6,
) -> random.Random:
    """为特定切片创建一个确定性的 random.Random 实例。

    这确保同一张切片始终生成相同的随机候选，
    无论处理顺序如何。

    Args:
        slide_base: 切片基础名称（用于哈希计算）。
        base_seed: 全局基础种子。
        mod: 哈希的模数。

    Returns:
        一个使用种子初始化过的 random.Random 实例。
    """
    slide_seed = base_seed + stable_int_hash(slide_base, mod=mod)
    return random.Random(slide_seed)


def deterministic_numpy_rng(
    slide_base: str,
    base_seed: int,
    mod: int = 10 ** 6,
) -> np.random.Generator:
    """为特定切片创建一个确定性的 NumPy 随机数生成器。

    Args:
        slide_base: 切片基础名称。
        base_seed: 全局基础种子。
        mod: 哈希的模数。

    Returns:
        一个使用种子初始化过的 numpy.random.Generator 实例。
    """
    slide_seed = base_seed + stable_int_hash(slide_base, mod=mod)
    return np.random.default_rng(slide_seed)
