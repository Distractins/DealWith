# -*- coding: utf-8 -*-
"""SPLICE 启发的采样器：带余弦相似度冗余惩罚的贪婪顺序选择。

策略：在每一步中，选择最大化以下值的候选块：
    value = baseline_score - lambda * max_cosine_similarity_to_selected

迁移自：dataprocessing/wsi_compare/4SPLICE_inspired.py
"""

import logging
from typing import List

import numpy as np

from common.dataclasses import CandidatePatch, ConfigBundle
from core.candidate_pool import CandidatePool
from core.diversity import far_enough
from utils.math_utils import cosine_similarity
from samplers.base_sampler import BaseSampler
from samplers import register_sampler

logger = logging.getLogger(__name__)


@register_sampler
class SpliceSampler(BaseSampler):
    """SPLICE 启发的顺序式非冗余选择。

    策略：
    1. 从 baseline_score 最高的候选块开始
    2. 迭代选择满足以下条件的候选块：
       max(baseline_score - lambda * max_cosine_similarity_to_selected)
    3. 将空间多样性作为硬性预过滤条件
    4. 为每个已选候选块记录 redundancy_penalty
    """

    name = "SPLICE-inspired"

    @staticmethod
    def algorithm_name() -> str:
        return "splice"

    def select_patches(
        self,
        candidate_pool: CandidatePool,
        num_patches: int,
    ) -> List[CandidatePatch]:
        """通过 SPLICE 带冗余惩罚的贪婪选择方式选择候选块。

        Args:
            candidate_pool: 预构建的候选池。
            num_patches: 目标数量（K）。

        Returns:
            已选的非冗余候选块列表。
        """
        candidates = candidate_pool.candidates
        if len(candidates) == 0:
            logger.warning(f"[{candidate_pool.slide_base}] SPLICE: empty pool")
            return []

        redundancy_lambda = self.config.redundancy_lambda
        min_dist = self.config.patch_size * self.config.min_center_distance_ratio
        slide_base = candidate_pool.slide_base

        # 在按 baseline_score 排序的副本上操作
        pool = sorted(
            candidates,
            key=lambda c: c.scores.baseline_score,
            reverse=True,
        )

        if len(pool) <= num_patches:
            return pool

        selected = []
        selected_feats = []

        # 贪婪顺序选择
        remaining = list(range(len(pool)))
        k = min(num_patches, len(pool))

        for _ in range(k):
            best_idx = None
            best_value = -float("inf")

            for idx in remaining:
                item = pool[idx]

                # 空间预过滤
                if selected:
                    ok_spatial = far_enough(
                        item.patch.cx,
                        item.patch.cy,
                        [(s.patch.cx, s.patch.cy) for s in selected],
                        min_dist,
                    )
                    if not ok_spatial:
                        continue

                # 计算选择价值
                baseline = item.scores.baseline_score
                if len(selected_feats) > 0:
                    max_sim = max(
                        cosine_similarity(item.feature.values, sf)
                        for sf in selected_feats
                    )
                    redundancy_penalty = redundancy_lambda * max_sim
                else:
                    max_sim = 0.0
                    redundancy_penalty = 0.0

                value = baseline - redundancy_penalty

                if value > best_value:
                    best_value = value
                    best_idx = idx
                    # 存储以备后续使用
                    item.selection_score = float(value)
                    item.redundancy_penalty = float(redundancy_penalty)

            if best_idx is not None:
                best_item = pool[best_idx]
                selected.append(best_item)
                selected_feats.append(best_item.feature.values)
                remaining.remove(best_idx)
            else:
                # 未找到满足空间多样性的候选块，选 baseline 最高的
                if remaining:
                    best_idx = max(
                        remaining,
                        key=lambda i: pool[i].scores.baseline_score,
                    )
                    selected.append(pool[best_idx])
                    selected_feats.append(pool[best_idx].feature.values)
                    remaining.remove(best_idx)

        # 第一个入选的候选块无惩罚
        if len(selected) > 0:
            selected[0].redundancy_penalty = 0.0
            selected[0].selection_score = float(selected[0].scores.baseline_score)

        logger.info(
            f"[{slide_base}] SPLICE: "
            f"selected {len(selected)}/{num_patches} patches"
        )
        return selected[:num_patches]
