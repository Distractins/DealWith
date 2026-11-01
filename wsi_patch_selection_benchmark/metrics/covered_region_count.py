# -*- coding: utf-8 -*-
"""覆盖区域计数指标。

统计有多少个空间网格单元包含至少一个被选中的图像块。
值越高 = 组织区域的空间覆盖越好。
"""

import logging
from typing import Dict, List

import numpy as np

from common.dataclasses import CandidatePatch
from metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)

#: 空间分箱的默认网格大小
DEFAULT_N_BINS = 5


class CoveredRegionCountMetric(BaseMetric):
    """空间区域覆盖指标。

    计算:
    - covered_region_count: n_bins * n_bins 个网格单元中包含至少一个
        被选中图像块的单元数量
    - covered_region_ratio: covered_count / total_bins
    """

    name = "Covered Region Count"

    def __init__(self, n_bins: int = DEFAULT_N_BINS):
        """初始化指标。

        Args:
            n_bins: 每个维度的分箱数（默认: 5，共 25 个分箱）。
        """
        self.n_bins = n_bins

    @staticmethod
    def metric_name() -> str:
        return "covered_region_count"

    def evaluate(
        self,
        selected_patches: List[CandidatePatch],
    ) -> Dict[str, float]:
        """计算覆盖区域计数。

        Args:
            selected_patches: 被选中的 CandidatePatch 对象列表。

        Returns:
            包含 covered_region_count 和 ratio 的字典。
        """
        if len(selected_patches) == 0:
            return {
                "covered_region_count": 0,
                "covered_region_ratio": 0.0,
                "n_bins": self.n_bins,
            }

        # 确定空间范围
        centers = np.array([
            [c.patch.cx, c.patch.cy] for c in selected_patches
        ])

        x_min, y_min = centers.min(axis=0)
        x_max, y_max = centers.max(axis=0)

        # 略微扩展范围以避免零宽度分箱
        if x_max == x_min:
            x_max = x_min + 1
        if y_max == y_min:
            y_max = y_min + 1

        # 统计被占用的分箱
        occupied = set()
        for cx, cy in centers:
            bx = min(int((cx - x_min) / (x_max - x_min) * self.n_bins), self.n_bins - 1)
            by = min(int((cy - y_min) / (y_max - y_min) * self.n_bins), self.n_bins - 1)
            occupied.add((bx, by))

        total_bins = self.n_bins * self.n_bins
        count = len(occupied)
        ratio = count / total_bins

        return {
            "covered_region_count": count,
            "covered_region_ratio": ratio,
            "n_bins": self.n_bins,
        }
