# -*- coding: utf-8 -*-
"""KMeans 采样器：在特征空间中对候选块进行聚类，选择各簇的中心点（medoid）。

使用自定义轻量级 k-means 实现，采用最远点初始化以保证簇中心的多样性。

迁移自：dataprocessing/wsi_compare/7kmeans.py
"""

import logging
from typing import List, Optional

import numpy as np

from common.dataclasses import CandidatePatch, ConfigBundle
from core.candidate_pool import CandidatePool
from samplers.base_sampler import BaseSampler
from samplers import register_sampler
from utils.seed import stable_int_hash

logger = logging.getLogger(__name__)


@register_sampler
class KMeansSampler(BaseSampler):
    """基于特征空间的 K-means 聚类与 medoid 选择。

    策略：
    1. 从候选池中提取特征矩阵
    2. 通过最远点采样初始化 K 个聚类中心
    3. 运行 k-means 聚类（最多8轮迭代）
    4. 选择离每个聚类中心最近的真实候选块作为 medoid
    5. 按 final_score 从高到低填充剩余名额
    """

    name = "K-Means Clustering"

    @staticmethod
    def algorithm_name() -> str:
        return "kmeans"

    def select_patches(
        self,
        candidate_pool: CandidatePool,
        num_patches: int,
    ) -> List[CandidatePatch]:
        """通过对特征向量进行 k-means 聚类来选择候选块。

        Args:
            candidate_pool: 预构建的候选池。
            num_patches: 目标数量（K）。

        Returns:
            已选的 medoid 候选块列表。
        """
        candidates = candidate_pool.candidates
        if len(candidates) == 0:
            logger.warning(f"[{candidate_pool.slide_base}] KMeans: empty pool")
            return []

        k = min(num_patches, len(candidates))

        # 构建特征矩阵
        feats = np.stack([c.feature.values for c in candidates])

        # 自定义 k-means
        labels, centers = self._simple_kmeans(
            feats,
            k,
            candidate_pool.slide_base,
        )

        # 每个聚类选一个 medoid
        selected = []
        used_indices = set()

        for cluster_id in range(k):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue

            # Medoid：与聚类中心距离最小的候选块
            cluster_feats = feats[cluster_indices]
            dists = np.linalg.norm(cluster_feats - centers[cluster_id], axis=1)
            medoid_relative_idx = int(np.argmin(dists))
            medoid_idx = int(cluster_indices[medoid_relative_idx])

            candidate = candidates[medoid_idx]
            candidate.cluster_id = cluster_id
            selected.append(candidate)
            used_indices.add(medoid_idx)

        # 按 final_score 从高到低填充剩余名额
        if len(selected) < k:
            remaining = sorted(
                [c for i, c in enumerate(candidates) if i not in used_indices],
                key=lambda c: c.scores.final_score,
                reverse=True,
            )
            for c in remaining:
                if len(selected) >= k:
                    break
                selected.append(c)

        logger.info(
            f"[{candidate_pool.slide_base}] KMeans: "
            f"selected {len(selected)}/{num_patches} patches "
            f"from {len(candidates)} candidates"
        )
        return selected[:num_patches]

    # ------------------------------------------------------------------
    # 轻量级自定义 k-means
    # ------------------------------------------------------------------

    def _simple_kmeans(
        self,
        X: np.ndarray,
        k: int,
        slide_base: str,
        max_iter: int = 8,
    ) -> tuple:
        """带最远点初始化的自定义 k-means。

        使用确定性 NumPy RNG，确保每张切片的可复现性。

        Args:
            X: 特征矩阵 (N, D)。
            k: 聚类数。
            slide_base: 切片基名，用于确定性随机种子。
            max_iter: 最大迭代次数。

        Returns:
            (labels: np.ndarray, centers: np.ndarray) 的元组。
        """
        n, d = X.shape
        if k >= n:
            # 聚类数超过点数：每个点自成一个聚类
            return np.arange(n), X.copy()

        # 确定性随机数生成器
        rng = np.random.default_rng(
            self.config.seed + stable_int_hash(slide_base, mod=10 ** 6)
        )

        # 最远点初始化
        centers = np.zeros((k, d), dtype=np.float32)
        # 从 baseline_score 前50%的候选中选取第一个中心
        top_half = max(1, n // 2)
        centers[0] = X[rng.integers(0, top_half)].copy()

        for i in range(1, k):
            dists = np.min(
                np.linalg.norm(X[:, None, :] - centers[None, :i, :], axis=2),
                axis=1,
            )
            centers[i] = X[int(np.argmax(dists))].copy()

        # 迭代优化
        labels = np.zeros(n, dtype=np.int32)
        for _ in range(max_iter):
            # 分配到最近的中心
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)

            # 更新聚类中心
            new_centers = np.zeros_like(centers)
            for j in range(k):
                mask = new_labels == j
                if mask.sum() > 0:
                    new_centers[j] = X[mask].mean(axis=0)
                else:
                    # 空聚类：随机选取一个点
                    new_centers[j] = X[rng.integers(0, n)]

            if np.array_equal(new_labels, labels):
                labels = new_labels
                centers = new_centers
                break

            labels = new_labels
            centers = new_centers

        return labels, centers
