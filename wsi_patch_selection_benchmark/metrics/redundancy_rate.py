# -*- coding: utf-8 -*-
"""冗余率指标。

衡量过于相似的patch对所占的比例：
- redundancy_rate：特征距离小于阈值的patch对占比
- 使用可配置的相似度阈值
"""

import logging
from typing import Dict, List

import numpy as np

from common.dataclasses import CandidatePatch
from metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)

#: 判定两个patch冗余的默认特征距离阈值
DEFAULT_REDUNDANCY_THRESHOLD = 0.15


class RedundancyRateMetric(BaseMetric):
    """冗余率指标。

    计算：
    - redundancy_rate：低于相似度阈值的patch对占比
    - redundancy_threshold：所使用的阈值
    """

    name = "Redundancy Rate"

    def __init__(self, threshold: float = DEFAULT_REDUNDANCY_THRESHOLD):
        """初始化指标。

        Args:
            threshold: 特征距离阈值，两个patch的特征距离低于该值时
                视为冗余（默认值：0.15）。
        """
        self.threshold = threshold

    @staticmethod
    def metric_name() -> str:
        return "redundancy_rate"

    def evaluate(
        self,
        selected_patches: List[CandidatePatch],
    ) -> Dict[str, float]:
        """计算冗余率。

        Args:
            selected_patches: 选中的 CandidatePatch 对象列表。

        Returns:
            包含 redundancy_rate 和 threshold 的字典。
        """
        n = len(selected_patches)
        if n < 2:
            logger.debug("Redundancy rate: need >= 2 patches")
            return {
                "redundancy_rate": 0.0,
                "redundancy_threshold": self.threshold,
            }

        # 统计相似patch对的数量
        feats = np.stack([c.feature.values for c in selected_patches])
        total_pairs = n * (n - 1) / 2
        similar_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(feats[i] - feats[j]))
                if d < self.threshold:
                    similar_pairs += 1

        rate = similar_pairs / total_pairs if total_pairs > 0 else 0.0

        return {
            "redundancy_rate": rate,
            "redundancy_threshold": self.threshold,
        }
