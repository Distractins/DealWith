# -*- coding: utf-8 -*-
"""空间覆盖率指标。

衡量选定贴片对组织区域的覆盖程度：
- 凸包面积比：convex_hull_area / total_tissue_area
- 贴片中心之间的平均最近邻距离
- 最近邻距离的标准差（覆盖的均匀性）
"""

import logging
from typing import Dict, List

import numpy as np
from scipy.spatial import ConvexHull

from common.dataclasses import CandidatePatch
from metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)


class SpatialCoverageMetric(BaseMetric):
    """空间覆盖率和分布指标。

    计算：
    - spatial_coverage_hull_ratio：凸包面积 / 边界框面积
    - spatial_mean_nn_dist：平均最近邻距离
    - spatial_std_nn_dist：最近邻距离的标准差
    """

    name = "Spatial Coverage"

    @staticmethod
    def metric_name() -> str:
        return "spatial_coverage"

    def evaluate(
        self,
        selected_patches: List[CandidatePatch],
    ) -> Dict[str, float]:
        """计算空间覆盖率指标。

        Args:
            selected_patches: 选定的 CandidatePatch 对象。

        Returns:
            空间覆盖率指标的字典。
        """
        if len(selected_patches) < 3:
            logger.debug("Spatial coverage: need >= 3 patches for convex hull")
            return {
                "spatial_coverage_hull_ratio": 0.0,
                "spatial_mean_nn_dist": 0.0,
                "spatial_std_nn_dist": 0.0,
            }

        # 提取中心坐标
        centers = np.array([
            [c.patch.cx, c.patch.cy] for c in selected_patches
        ], dtype=np.float64)

        # ---- 凸包面积比 ----
        try:
            hull = ConvexHull(centers)
            hull_area = hull.volume  # 在二维中，volume 即面积

            # 所有中心的边界框
            bbox_area = (
                (centers[:, 0].max() - centers[:, 0].min())
                * (centers[:, 1].max() - centers[:, 1].min())
            )
            if bbox_area > 0:
                hull_ratio = float(hull_area / bbox_area)
            else:
                hull_ratio = 0.0
        except Exception:
            hull_ratio = 0.0

        # ---- 最近邻距离 ----
        nn_dists = []
        for i in range(len(centers)):
            dists = np.linalg.norm(centers - centers[i], axis=1)
            dists[i] = np.inf  # 排除自身
            nn_dists.append(float(np.min(dists)))

        mean_nn = float(np.mean(nn_dists)) if nn_dists else 0.0
        std_nn = float(np.std(nn_dists)) if nn_dists else 0.0

        return {
            "spatial_coverage": hull_ratio,  # 主指标，与 metric_name 对应
            "spatial_coverage_hull_ratio": hull_ratio,
            "spatial_mean_nn_dist": mean_nn,
            "spatial_std_nn_dist": std_nn,
        }
