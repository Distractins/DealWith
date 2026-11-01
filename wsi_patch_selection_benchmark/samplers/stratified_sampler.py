# -*- coding: utf-8 -*-
"""分层空间采样（Stratified Spatial Sampling）：将WSI划分为空间网格单元，从每个
单元中挑选最佳候选，然后用多样性策略填充剩余名额。

迁移自：dataprocessing/wsi_compare/3Stratified Spatial Sampling.py
"""

import logging
from typing import Dict, List, Tuple

from common.dataclasses import CandidatePatch, ConfigBundle
from core.candidate_pool import CandidatePool
from core.diversity import far_enough, feature_far_enough
from samplers.base_sampler import BaseSampler
from samplers import register_sampler

logger = logging.getLogger(__name__)


@register_sampler
class StratifiedSampler(BaseSampler):
    """带多样性约束的分层空间采样。

    策略：
    1. 将WSI划分为 N_BINS_X * N_BINS_Y 个空间网格单元
    2. 根据每个候选块的中心所在单元进行分组
    3. 在每个单元内按 baseline_score 排序
    4. 多轮渐进式选择，逐步放宽约束：
       - 第1轮：每个单元选1个最优候选（空间+特征多样性）
       - 第2轮：从剩余候选中填充（空间+特征多样性）
       - 第3轮：从剩余候选中填充（仅空间多样性）
       - 第4轮：无条件填充
    """

    name = "Stratified Spatial Sampling"

    @staticmethod
    def algorithm_name() -> str:
        return "stratified"

    def select_patches(
        self,
        candidate_pool: CandidatePool,
        num_patches: int,
    ) -> List[CandidatePatch]:
        """按分层空间覆盖选择候选块。

        Args:
            candidate_pool: 预构建的候选池。
            num_patches: 目标数量（K）。

        Returns:
            空间覆盖良好的已选候选块列表。
        """
        if len(candidate_pool) == 0:
            logger.warning(
                f"[{candidate_pool.slide_base}] Stratified: empty pool"
            )
            return []

        n_bins_x = self.config.n_bins_x
        n_bins_y = self.config.n_bins_y

        # 从掩码推算WSI的大致尺寸
        mask = candidate_pool.tissue_mask
        ds = candidate_pool.ds
        w0, h0 = mask.shape[1] * ds, mask.shape[0] * ds

        # 将候选块分配到各网格单元
        bin_to_candidates: Dict[Tuple[int, int], List[CandidatePatch]] = {}
        for c in candidate_pool.candidates:
            bx = min(int(c.patch.cx / (w0 / n_bins_x)), n_bins_x - 1)
            by = min(int(c.patch.cy / (h0 / n_bins_y)), n_bins_y - 1)
            bin_key = (bx, by)
            bin_to_candidates.setdefault(bin_key, []).append(c)

        # 在每个网格单元内按 baseline_score 排序
        for bin_key in bin_to_candidates:
            bin_to_candidates[bin_key] = sorted(
                bin_to_candidates[bin_key],
                key=lambda c: c.scores.baseline_score,
                reverse=True,
            )

        logger.debug(
            f"[{candidate_pool.slide_base}] Stratified: "
            f"{len(bin_to_candidates)} non-empty bins "
            f"(grid: {n_bins_x}x{n_bins_y})"
        )

        selection = self._select_stratified(
            bin_to_candidates=bin_to_candidates,
            topk=num_patches,
            patch_size=self.config.patch_size,
            min_center_distance_ratio=self.config.min_center_distance_ratio,
            min_feature_distance=self.config.min_feature_distance,
        )

        logger.info(
            f"[{candidate_pool.slide_base}] Stratified: "
            f"selected {len(selection)}/{num_patches} patches"
        )
        return selection

    # ------------------------------------------------------------------
    # 多轮分层选择
    # ------------------------------------------------------------------

    @staticmethod
    def _select_stratified(
        bin_to_candidates: Dict[Tuple[int, int], List[CandidatePatch]],
        topk: int,
        patch_size: int,
        min_center_distance_ratio: float,
        min_feature_distance: float,
    ) -> List[CandidatePatch]:
        """跨网格单元的多轮贪婪选择。

        Args:
            bin_to_candidates: 包含已排序候选块的空间网格单元。
            topk: 目标数量 K。
            patch_size: 用于计算空间阈值的候选块尺寸。
            min_center_distance_ratio: 空间多样性阈值比率。
            min_feature_distance: 特征多样性阈值。

        Returns:
            已选候选块列表。
        """
        selected = []
        selected_centers = []
        selected_feats = []
        min_dist = patch_size * min_center_distance_ratio

        # 构建所有候选块的扁平索引，便于高效访问
        all_items = []
        bin_membership = {}
        idx = 0
        for bin_key, items in bin_to_candidates.items():
            for i, item in enumerate(items):
                all_items.append(item)
                bin_membership[idx] = (bin_key, i)
                idx += 1

        remaining = set(range(len(all_items)))

        # ---- 第1轮：每个网格单元选最优候选（空间+特征多样性） ----
        bin_order = sorted(
            bin_to_candidates.keys(),
            key=lambda bk: bin_to_candidates[bk][0].scores.baseline_score
            if bin_to_candidates[bk] else 0.0,
            reverse=True,
        )

        for bin_key in bin_order:
            if len(selected) >= topk:
                break
            items = bin_to_candidates[bin_key]
            # 从该网格单元中挑选最佳剩余候选
            for item in items:
                idx = all_items.index(item)
                if idx not in remaining:
                    continue
                if far_enough(item.patch.cx, item.patch.cy, selected_centers, min_dist):
                    if feature_far_enough(item.feature.values, selected_feats, min_feature_distance):
                        selected.append(item)
                        selected_centers.append((item.patch.cx, item.patch.cy))
                        selected_feats.append(item.feature.values)
                        remaining.discard(idx)
                        break

        # ---- 第2轮：从顶部剩余候选中填充（空间+特征多样性） ----
        if len(selected) < topk:
            remaining_items = [all_items[i] for i in sorted(remaining)]
            remaining_items.sort(key=lambda c: c.scores.baseline_score, reverse=True)
            for item in remaining_items:
                if len(selected) >= topk:
                    break
                if far_enough(item.patch.cx, item.patch.cy, selected_centers, min_dist):
                    if feature_far_enough(item.feature.values, selected_feats, min_feature_distance):
                        selected.append(item)
                        selected_centers.append((item.patch.cx, item.patch.cy))
                        selected_feats.append(item.feature.values)

        # ---- 第3轮：仅空间多样性 ----
        if len(selected) < topk:
            remaining_items = [c for c in all_items if c not in selected]
            remaining_items.sort(key=lambda c: c.scores.baseline_score, reverse=True)
            for item in remaining_items:
                if len(selected) >= topk:
                    break
                if far_enough(item.patch.cx, item.patch.cy, selected_centers, min_dist):
                    selected.append(item)
                    selected_centers.append((item.patch.cx, item.patch.cy))

        # ---- 第4轮：无约束 ----
        if len(selected) < topk:
            remaining_items = [c for c in all_items if c not in selected]
            remaining_items.sort(key=lambda c: c.scores.baseline_score, reverse=True)
            for item in remaining_items:
                if len(selected) >= topk:
                    break
                selected.append(item)

        return selected[:topk]
