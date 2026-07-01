# -*- coding: utf-8 -*-
"""网格采样器：从网格扫描的候选池中，按基线质量分数选择 top-K 图像块，同时兼顾空间与特征多样性。

这是最简单的质量驱动基线。
"""

import logging
from typing import List

from common.dataclasses import CandidatePatch, ConfigBundle
from common.enums import QCLevel
from core.candidate_pool import CandidatePool
from core.diversity import select_diverse_topk
from samplers.base_sampler import BaseSampler
from samplers import register_sampler

logger = logging.getLogger(__name__)


@register_sampler
class GridSampler(BaseSampler):
    """基于网格的质量驱动图像块选择。

    策略：
    1. 按 baseline_score 对候选块排序（不含创新性/形态学）
    2. 施加空间与特征多样性约束
    3. 选择 top-K 多样化图像块
    """

    name = "Grid (Quality Top-K)"

    @staticmethod
    def algorithm_name() -> str:
        return "grid"

    def select_patches(
        self,
        candidate_pool: CandidatePool,
        num_patches: int,
    ) -> List[CandidatePatch]:
        """选择带有多样性约束的 top-K 高质量图像块。

        创新性权重设为 0.0，以实现纯质量驱动的选择。

        参数:
            candidate_pool: 预先构建的候选池。
            num_patches: 目标数量 (K)。

        返回:
            选中的多样化高质量图像块。
        """
        if len(candidate_pool) == 0:
            logger.warning(f"[{candidate_pool.slide_base}] Grid: empty pool")
            return []

        # 按 baseline_score 排序（仅看质量，不含形态学）
        sorted_candidates = sorted(
            candidate_pool.candidates,
            key=lambda c: c.scores.baseline_score,
            reverse=True,
        )

        logger.debug(
            f"[{candidate_pool.slide_base}] Grid: "
            f"{len(sorted_candidates)} candidates sorted by baseline_score"
        )

        # 应用多样性选择
        selected = select_diverse_topk(
            candidates=sorted_candidates,
            topk=num_patches,
            patch_size=self.config.patch_size,
            min_center_distance_ratio=self.config.min_center_distance_ratio,
            min_feature_distance=self.config.min_feature_distance,
        )

        logger.info(
            f"[{candidate_pool.slide_base}] Grid: "
            f"selected {len(selected)}/{num_patches} patches"
        )
        return selected
