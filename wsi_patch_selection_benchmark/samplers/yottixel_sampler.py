# -*- coding: utf-8 -*-
"""Yottixel 启发的采样器：先按颜色聚类，再在空间上进行子聚类。

两级"马赛克"方法：
1. 颜色聚类：对24-bin RGB直方图进行 K-means 聚类
2. 空间子聚类：在每个颜色簇内，对 (cx, cy) 坐标进行 K-means 聚类
3. 选择离空间聚类中心最近的候选块
4. 对最终选择施加空间多样性约束

迁移自：dataprocessing/wsi_compare/8Yottixel-inspired.py
"""

import logging
from typing import List

import numpy as np

from common.dataclasses import CandidatePatch, ConfigBundle, FeatureVector
from core.candidate_pool import CandidatePool
from core.feature_extraction import rgb_hist_24bins
from core.diversity import far_enough
from samplers.base_sampler import BaseSampler
from samplers import register_sampler
from utils.seed import stable_int_hash

logger = logging.getLogger(__name__)


@register_sampler
class YottixelSampler(BaseSampler):
    """Yottixel 启发的马赛克式候选块选择。

    策略：
    1. 为所有含组织的候选块计算24-bin RGB直方图
    2. 按颜色直方图进行 K-means 聚类 → COLOR_CLUSTERS 个分组
    3. 在每个颜色组内，按空间坐标进行子聚类
       → 选择离每个空间聚类中心最近的 medoid
    4. 去重，按分数排序，施加空间多样性约束
    """

    name = "Yottixel-inspired Mosaic"

    @staticmethod
    def algorithm_name() -> str:
        return "yottixel"

    def select_patches(
        self,
        candidate_pool: CandidatePool,
        num_patches: int,
    ) -> List[CandidatePatch]:
        """通过两级颜色-空间马赛克方式选择候选块。

        Args:
            candidate_pool: 预构建的候选池。
            num_patches: 目标数量（K）。

        Returns:
            已选的马赛克候选块列表。
        """
        candidates = candidate_pool.candidates
        if len(candidates) == 0:
            logger.warning(f"[{candidate_pool.slide_base}] Yottixel: empty pool")
            return []

        color_clusters = self.config.color_clusters
        spatial_ratio = self.config.spatial_ratio_per_cluster
        max_total = self.config.max_total_candidates
        slide_base = candidate_pool.slide_base

        # 阶段1：计算颜色直方图
        # 仅使用富含组织的候选块
        tissue_candidates = [
            c for c in candidates
            if c.metrics.tissue_ratio >= self.config.min_tissue_ratio_preselect
        ]
        if len(tissue_candidates) == 0:
            tissue_candidates = candidates

        color_hists = []
        for c in tissue_candidates:
            hist = rgb_hist_24bins(c.patch.patch_np)
            color_hists.append(hist)
            c.color_hist = hist

        color_hists = np.stack(color_hists)

        # 阶段2：颜色聚类
        n_colors = min(color_clusters, len(tissue_candidates))
        rng = np.random.default_rng(
            self.config.seed + stable_int_hash(slide_base, mod=10 ** 6)
        )

        color_labels, color_centers = self._simple_kmeans_np(
            color_hists, n_colors, rng
        )

        # 阶段3：在每个颜色组内进行空间子聚类
        mosaic_candidates = []
        for color_id in range(n_colors):
            group_indices = np.where(color_labels == color_id)[0]
            if len(group_indices) == 0:
                continue

            group_candidates = [tissue_candidates[int(i)] for i in group_indices]

            # 空间聚类
            n_spatial = max(1, int(len(group_candidates) * spatial_ratio))
            n_spatial = min(n_spatial, len(group_candidates))

            if n_spatial == 1:
                # 只有一个空间聚类：按分数选最优
                best = max(group_candidates, key=lambda c: c.scores.final_score)
                best.cluster_id = color_id
                best.spatial_cluster_id = 0
                mosaic_candidates.append(best)
                continue

            # 构建空间坐标矩阵
            coords = np.array(
                [[c.patch.cx, c.patch.cy] for c in group_candidates],
                dtype=np.float32,
            )
            coords[:, 0] /= (candidate_pool.tissue_mask.shape[1] * candidate_pool.ds)
            coords[:, 1] /= (candidate_pool.tissue_mask.shape[0] * candidate_pool.ds)

            spatial_labels, spatial_centers = self._simple_kmeans_np(
                coords, n_spatial, rng
            )

            # 为每个空间聚类选取 medoid
            for sid in range(n_spatial):
                s_indices = np.where(spatial_labels == sid)[0]
                if len(s_indices) == 0:
                    continue
                s_coords = coords[s_indices]
                dists = np.linalg.norm(s_coords - spatial_centers[sid], axis=1)
                best_idx = int(s_indices[int(np.argmin(dists))])
                cand = group_candidates[best_idx]
                cand.cluster_id = color_id
                cand.spatial_cluster_id = sid
                mosaic_candidates.append(cand)

        # 限制总数上限
        mosaic_candidates = sorted(
            mosaic_candidates,
            key=lambda c: c.scores.final_score,
            reverse=True,
        )[:max_total]

        # 按 (x0, y0) 去重
        seen = set()
        deduped = []
        for c in mosaic_candidates:
            key = (c.patch.x0, c.patch.y0)
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        logger.debug(
            f"[{slide_base}] Yottixel: "
            f"{len(deduped)} mosaic candidates "
            f"({n_colors} color clusters)"
        )

        # 最终选择，施加空间多样性
        selected = self._select_spatial_diverse(
            deduped,
            num_patches,
            self.config.patch_size * self.config.min_center_distance_ratio,
        )

        logger.info(
            f"[{slide_base}] Yottixel: "
            f"selected {len(selected)}/{num_patches} patches"
        )
        return selected

    # ------------------------------------------------------------------
    # 辅助函数
    # ------------------------------------------------------------------

    @staticmethod
    def _simple_kmeans_np(
        X: np.ndarray,
        k: int,
        rng: np.random.Generator,
        max_iter: int = 10,
    ) -> tuple:
        """用于颜色直方图和空间坐标的轻量级 k-means。

        Args:
            X: 数据矩阵 (N, D)。
            k: 聚类数。
            rng: NumPy 随机数生成器。
            max_iter: 最大迭代次数。

        Returns:
            (labels, centers) 的元组。
        """
        n, d = X.shape
        if k >= n:
            return np.arange(n), X.copy()

        # 随机初始化
        idx = rng.choice(n, k, replace=False)
        centers = X[idx].copy().astype(np.float32)

        labels = np.zeros(n, dtype=np.int32)
        for _ in range(max_iter):
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)

            new_centers = np.zeros_like(centers)
            for j in range(k):
                mask = new_labels == j
                if mask.sum() > 0:
                    new_centers[j] = X[mask].mean(axis=0)
                else:
                    new_centers[j] = X[rng.integers(0, n)]

            if np.array_equal(new_labels, labels):
                labels = new_labels
                centers = new_centers
                break

            labels = new_labels
            centers = new_centers

        return labels, centers

    @staticmethod
    def _select_spatial_diverse(
        candidates: List[CandidatePatch],
        topk: int,
        min_dist: float,
    ) -> List[CandidatePatch]:
        """带空间多样性的贪婪选择。

        Args:
            candidates: 已按分数降序排列的候选块列表。
            topk: 目标数量。
            min_dist: 最小中心距离。

        Returns:
            已选的多样性候选块列表。
        """
        selected = []
        selected_centers = []

        for c in candidates:
            if len(selected) >= topk:
                break
            if far_enough(c.patch.cx, c.patch.cy, selected_centers, min_dist):
                selected.append(c)
                selected_centers.append((c.patch.cx, c.patch.cy))

        # 填充剩余名额
        if len(selected) < topk:
            for c in candidates:
                if len(selected) >= topk:
                    break
                if c not in selected:
                    selected.append(c)

        return selected[:topk]
