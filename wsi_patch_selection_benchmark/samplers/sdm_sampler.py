# -*- coding: utf-8 -*-
"""SDM（独特形态选择，Distinct Morphology Selection）采样器。

两阶段形态感知选择：
1. 种子选取：识别具有高质量+多样化形态的独特种子
2. 分组分配+代表性选择：将候选块分配到最近的种子，
   按质量偏好在每组中选取 medoid

迁移自：dataprocessing/wsi_compare/6SDM：Distinct Morphology Selection.py
"""

import logging
from typing import List, Dict

import numpy as np

from common.dataclasses import CandidatePatch, ConfigBundle, FeatureVector
from core.candidate_pool import CandidatePool
from core.feature_extraction import morphology_feature_vector
from samplers.base_sampler import BaseSampler
from samplers import register_sampler

logger = logging.getLogger(__name__)


@register_sampler
class SDMSampler(BaseSampler):
    """SDM 启发的独特形态选择。

    策略：
    1. 为所有候选块计算12维形态特征向量
    2. 选择 K 个独特种子：最大化
       0.75 * min_morph_dist + 0.25 * quality_score
    3. 将每个候选块分配到最近的种子（硬聚类）
    4. 每组内：选择最小化以下指标的 medoid
       mean_pairwise_morph_dist - 0.25*qc_quality - 0.20*tumor_morph
    5. 按复合质量-肿瘤分数填充剩余名额
    """

    name = "SDM (Distinct Morphology)"

    @staticmethod
    def algorithm_name() -> str:
        return "sdm"

    def select_patches(
        self,
        candidate_pool: CandidatePool,
        num_patches: int,
    ) -> List[CandidatePatch]:
        """通过独特形态选择方式选择候选块。

        Args:
            candidate_pool: 预构建的候选池。
            num_patches: 目标数量（K）。

        Returns:
            已选的形态多样性候选块列表。
        """
        candidates = candidate_pool.candidates
        if len(candidates) == 0:
            logger.warning(f"[{candidate_pool.slide_base}] SDM: empty pool")
            return []

        k = min(num_patches, len(candidates))

        # 计算形态特征向量
        for c in candidates:
            if c.morph_feature is None:
                c.morph_feature = FeatureVector(
                    values=morphology_feature_vector(c.metrics.to_dict())
                )

        morph_feats = np.stack([c.morph_feature.values for c in candidates])

        # ---- 阶段1：选择独特形态种子 ----
        seed_indices = self._choose_distinct_seeds(candidates, morph_feats, k)
        selected_indices = set(seed_indices)

        # ---- 阶段2：分配到种子 + 选择代表 ----
        if len(seed_indices) < k:
            # 将剩余候选块分配到最近的种子
            groups = self._assign_to_seeds(morph_feats, seed_indices)
            # 在每组中选择代表
            more_indices = self._choose_group_representatives(
                candidates, morph_feats, groups, k - len(seed_indices)
            )
            for idx in more_indices:
                if idx not in selected_indices:
                    selected_indices.add(idx)
                    if len(selected_indices) >= k:
                        break

        # ---- 阶段3：按复合分数填充剩余名额 ----
        if len(selected_indices) < k:
            composite_scores = [
                0.55 * c.scores.qc_quality_norm + 0.45 * c.scores.tumor_morph_norm
                for c in candidates
            ]
            remaining_order = sorted(
                [i for i in range(len(candidates)) if i not in selected_indices],
                key=lambda i: composite_scores[i],
                reverse=True,
            )
            for idx in remaining_order:
                if len(selected_indices) >= k:
                    break
                selected_indices.add(idx)

        selected = [candidates[i] for i in sorted(selected_indices)[:k]]

        logger.info(
            f"[{candidate_pool.slide_base}] SDM: "
            f"selected {len(selected)}/{num_patches} patches "
            f"({len(seed_indices)} seeds)"
        )
        return selected

    # ------------------------------------------------------------------
    # 种子选择
    # ------------------------------------------------------------------

    @staticmethod
    def _choose_distinct_seeds(
        candidates: List[CandidatePatch],
        morph_feats: np.ndarray,
        k: int,
    ) -> List[int]:
        """选择 K 个种子，最大化 min_morph_dist + quality。

        第一个种子：max(0.6*qc_quality_norm + 0.4*tumor_morph_norm)
        后续种子：max(0.75*min_dist_to_existing + 0.25*quality_score)

        Args:
            candidates: 候选块列表。
            morph_feats: (N, D) 形态特征矩阵。
            k: 种子数量。

        Returns:
            种子候选块索引列表。
        """
        n = len(candidates)
        if k >= n:
            return list(range(n))

        seeds = []
        remaining = set(range(n))

        # 第一个种子：复合质量最高者
        quality_scores = np.array([
            0.6 * c.scores.qc_quality_norm + 0.4 * c.scores.tumor_morph_norm
            for c in candidates
        ])
        first = int(np.argmax(quality_scores))
        seeds.append(first)
        remaining.discard(first)

        # 后续种子
        for _ in range(1, k):
            if not remaining:
                break

            selected_morph = morph_feats[seeds]

            best_idx = None
            best_score = -float("inf")

            for idx in remaining:
                feat = morph_feats[idx]
                dists = np.linalg.norm(selected_morph - feat, axis=1)
                min_dist = float(np.min(dists)) if len(dists) > 0 else 0.0

                quality = quality_scores[idx]
                score = 0.75 * min_dist + 0.25 * quality

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                seeds.append(best_idx)
                remaining.discard(best_idx)

        return seeds

    # ------------------------------------------------------------------
    # 分配
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_to_seeds(
        morph_feats: np.ndarray,
        seed_indices: List[int],
    ) -> Dict[int, List[int]]:
        """将每个候选块索引分配到离其最近的种子（硬聚类）。

        Args:
            morph_feats: (N, D) 形态特征矩阵。
            seed_indices: 种子候选块的索引列表。

        Returns:
            字典：seed_index -> 候选块索引列表。
        """
        groups: Dict[int, List[int]] = {s: [] for s in seed_indices}
        seed_morph = morph_feats[seed_indices]

        for i in range(len(morph_feats)):
            if i in seed_indices:
                groups[i].append(i)
                continue
            dists = np.linalg.norm(seed_morph - morph_feats[i], axis=1)
            nearest = seed_indices[int(np.argmin(dists))]
            groups[nearest].append(i)

        return groups

    # ------------------------------------------------------------------
    # 组内代表选择
    # ------------------------------------------------------------------

    @staticmethod
    def _choose_group_representatives(
        candidates: List[CandidatePatch],
        morph_feats: np.ndarray,
        groups: Dict[int, List[int]],
        target_k: int,
    ) -> List[int]:
        """从各组中选取除种子外的代表候选块。

        对每个组，找到最小化以下指标的 medoid：
        mean_pairwise_morph_dist - 0.25*qc_quality_norm - 0.20*tumor_morph_norm

        Args:
            candidates: 候选块列表。
            morph_feats: (N, D) 特征矩阵。
            groups: seed_index -> 成员索引。
            target_k: 要选取的额外代表数量。

        Returns:
            已选候选块索引列表。
        """
        selected = []
        quality_scores = np.array([
            0.25 * c.scores.qc_quality_norm + 0.20 * c.scores.tumor_morph_norm
            for c in candidates
        ])

        # 为每个非种子成员评分
        scored_indices = []
        for seed_idx, members in groups.items():
            if len(members) <= 1:
                continue
            non_seed_members = [m for m in members if m != seed_idx]
            for m in non_seed_members:
                # 到组内各成员的平均成对距离
                group_feats = morph_feats[members]
                dists = np.linalg.norm(group_feats - morph_feats[m], axis=1)
                mean_dist = float(np.mean(dists))

                score = mean_dist - quality_scores[m]
                scored_indices.append((score, m))

        # 按分数升序排列（分数越低 = 代表性越好）
        scored_indices.sort(key=lambda x: x[0])

        return [idx for _, idx in scored_indices[:target_k]]
