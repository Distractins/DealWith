# -*- coding: utf-8 -*-
"""多样性选择：空间与特征空间约束。

从原始脚本的 [G] 章节迁移而来。

确保所选图块在以下两方面不冗余：
- 空间位置（图块中心之间的欧几里得距离）
- 特征空间（特征向量之间的欧几里得距离）
"""

import logging
from typing import List, Tuple

import numpy as np

from common.dataclasses import CandidatePatch

logger = logging.getLogger(__name__)


def far_enough(
    x: int,
    y: int,
    selected_centers: List[Tuple[int, int]],
    min_dist: float,
) -> bool:
    """检查新图块中心在空间上是否与所有已选图块足够远。

    Args:
        x: 候选中心的 X 坐标。
        y: 候选中心的 Y 坐标。
        selected_centers: 已选图块的 (cx, cy) 列表。
        min_dist: 允许的最小欧几里得距离。

    Returns:
        如果候选与所有已选图块都足够远则返回 True。
    """
    for sx, sy in selected_centers:
        if (x - sx) ** 2 + (y - sy) ** 2 < min_dist ** 2:
            return False
    return True


def feature_far_enough(
    feat: np.ndarray,
    selected_feats: List[np.ndarray],
    min_feat_dist: float,
) -> bool:
    """检查特征向量在特征空间中是否与所有已选向量足够远。

    使用归一化特征空间中的 L2（欧几里得）距离。

    Args:
        feat: 候选的特征向量（np.ndarray）。
        selected_feats: 已选图块的特征向量列表。
        min_feat_dist: 允许的最小欧几里得距离。

    Returns:
        如果候选与所有已选图块都足够多样则返回 True。
    """
    if len(selected_feats) == 0:
        return True
    for sf in selected_feats:
        dist = np.linalg.norm(feat - sf)
        if dist < min_feat_dist:
            return False
    return True


def select_diverse_topk(
    candidates: List[CandidatePatch],
    topk: int,
    patch_size: int,
    min_center_distance_ratio: float = 0.65,
    min_feature_distance: float = 0.10,
) -> List[CandidatePatch]:
    """选择具有空间与特征多样性约束的 Top-K 候选。

    这是一个贪心选择，按分数降序遍历候选。

    三次遍历，约束逐次放宽：
    1. 完全多样性：同时满足空间与特征约束
    2. 仅空间：仅空间约束（放松特征约束）
    3. 无约束：取剩余的任何候选

    Args:
        candidates: 按分数降序排序的 CandidatePatch 列表。
        topk: 要选择的图块数量。
        patch_size: 图块边长（用于计算空间阈值）。
        min_center_distance_ratio: 最小空间距离 = ratio * patch_size。
        min_feature_distance: 最小特征空间距离（L2 范数）。

    Returns:
        最多 `topk` 个选中的 CandidatePatch 对象。
    """
    if len(candidates) == 0 or topk <= 0:
        return []

    selected: List[CandidatePatch] = []
    selected_centers: List[Tuple[int, int]] = []
    selected_feats: List[np.ndarray] = []

    min_center_distance = patch_size * min_center_distance_ratio

    # 第 1 轮：完全多样性（空间 + 特征）
    for item in candidates:
        if len(selected) >= topk:
            break

        ok_spatial = far_enough(item.patch.cx, item.patch.cy, selected_centers, min_center_distance)
        ok_feature = feature_far_enough(item.feature.values, selected_feats, min_feature_distance)

        if ok_spatial and ok_feature:
            selected.append(item)
            selected_centers.append((item.patch.cx, item.patch.cy))
            selected_feats.append(item.feature.values)

    # 第 2 轮：仅空间多样性
    if len(selected) < topk:
        for item in candidates:
            if len(selected) >= topk:
                break
            if item in selected:
                continue

            ok_spatial = far_enough(item.patch.cx, item.patch.cy, selected_centers, min_center_distance)
            if ok_spatial:
                selected.append(item)
                selected_centers.append((item.patch.cx, item.patch.cy))
                # 不添加到 selected_feats（特征约束已放宽）

    # 第 3 轮：无约束（按分数取前几个）
    if len(selected) < topk:
        for item in candidates:
            if len(selected) >= topk:
                break
            if item not in selected:
                selected.append(item)

    return selected[:topk]
