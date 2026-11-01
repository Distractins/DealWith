# -*- coding: utf-8 -*-
"""随机采样器：从候选池中均匀随机地选择图像块。

这是最简单的基线方法。无质量、多样性或空间约束。
"""

import logging
import random
from typing import List

import numpy as np

from common.dataclasses import CandidatePatch, ConfigBundle
from core.candidate_pool import CandidatePool
from samplers.base_sampler import BaseSampler
from samplers import register_sampler

logger = logging.getLogger(__name__)


@register_sampler
class RandomSampler(BaseSampler):
    """从候选池中均匀随机选择图像块。

    为可复现性使用确定性的每切片随机数生成器。
    """

    name = "Random"

    @staticmethod
    def algorithm_name() -> str:
        return "random"

    def select_patches(
        self,
        candidate_pool: CandidatePool,
        num_patches: int,
    ) -> List[CandidatePatch]:
        """从候选池中随机选择图像块。

        如果候选池中的候选数量少于 num_patches，则返回全部。

        参数:
            candidate_pool: 预构建的候选池。
            num_patches: 目标数量（K）。

        返回:
            随机选择的候选图像块。
        """
        if len(candidate_pool) == 0:
            logger.warning(
                f"[{candidate_pool.slide_base}] Random: empty pool"
            )
            return []

        # 使用确定性的每切片随机数生成器
        rng = random.Random(
            self.config.seed + hash(candidate_pool.slide_base) % (10 ** 6)
        )

        candidates = candidate_pool.candidates
        k = min(num_patches, len(candidates))
        selected = rng.sample(candidates, k)

        logger.info(
            f"[{candidate_pool.slide_base}] Random: "
            f"selected {len(selected)}/{num_patches} patches"
        )
        return selected
