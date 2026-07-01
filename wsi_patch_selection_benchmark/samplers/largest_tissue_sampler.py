# -*- coding: utf-8 -*-
"""最大组织成分采样器：将候选区域限制在组织掩码的最大连通分量内，
然后按质量+多样性进行选择。

迁移自：dataprocessing/wsi_compare/1Largest Tissue Component.py
"""

import logging
from typing import List

from common.dataclasses import CandidatePatch, ConfigBundle
from core.candidate_pool import CandidatePool
from core.diversity import select_diverse_topk
from samplers.base_sampler import BaseSampler
from samplers import register_sampler

logger = logging.getLogger(__name__)


@register_sampler
class LargestTissueSampler(BaseSampler):
    """选择限制在最大组织成分内。

    策略：
    1. 过滤候选区域，仅保留中心位于组织掩码最大连通分量内的候选
    2. 按 baseline_score 排序
    3. 施加空间+特征多样性约束
    """

    name = "Largest Tissue Component"

    @staticmethod
    def algorithm_name() -> str:
        return "largest_tissue"

    def select_patches(
        self,
        candidate_pool: CandidatePool,
        num_patches: int,
    ) -> List[CandidatePatch]:
        """从最大组织成分中选择图块。

        参数:
            candidate_pool: 预构建的候选池，包含 largest_mask。
            num_patches: 目标数量 (K)。

        返回:
            从最大组织区域中选择的图块。
        """
        if len(candidate_pool) == 0:
            logger.warning(
                f"[{candidate_pool.slide_base}] LargestTissue: empty pool"
            )
            return []

        # 过滤到最大连通分量
        if candidate_pool.largest_mask is not None:
            filtered = candidate_pool.filter_by_mask_region(
                candidate_pool.largest_mask, candidate_pool.ds
            )
        else:
            filtered = candidate_pool.candidates

        if len(filtered) == 0:
            logger.warning(
                f"[{candidate_pool.slide_base}] LargestTissue: "
                f"no candidates in largest component"
            )
            return []

        logger.debug(
            f"[{candidate_pool.slide_base}] LargestTissue: "
            f"{len(filtered)} candidates in largest component "
            f"(from {len(candidate_pool)} total)"
        )

        # 按 baseline_score 排序
        sorted_candidates = sorted(
            filtered,
            key=lambda c: c.scores.baseline_score,
            reverse=True,
        )

        # 施加多样性选择
        selected = select_diverse_topk(
            candidates=sorted_candidates,
            topk=num_patches,
            patch_size=self.config.patch_size,
            min_center_distance_ratio=self.config.min_center_distance_ratio,
            min_feature_distance=self.config.min_feature_distance,
        )

        logger.info(
            f"[{candidate_pool.slide_base}] LargestTissue: "
            f"selected {len(selected)}/{num_patches} patches"
        )
        return selected
