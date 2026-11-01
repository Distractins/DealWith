# -*- coding: utf-8 -*-
"""特征多样性指标。

衡量选定贴片在特征空间中的多样性：
- 特征向量之间的平均成对欧几里得距离
- 最小成对距离（最差情况下的多样性）
"""

import logging
from typing import Dict, List

import numpy as np

from common.dataclasses import CandidatePatch
from metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)


class FeatureDiversityMetric(BaseMetric):
    """特征空间多样性指标。

    计算：
    - feature_diversity_mean_dist：平均成对 L2 距离
    - feature_diversity_min_dist：最小成对 L2 距离
    """

    name = "Feature Diversity"

    @staticmethod
    def metric_name() -> str:
        return "feature_diversity"

    def evaluate(
        self,
        selected_patches: List[CandidatePatch],
    ) -> Dict[str, float]:
        """计算特征多样性指标。

        参数：
            selected_patches：选定的 CandidatePatch 对象。

        返回：
            特征多样性指标的字典。
        """
        if len(selected_patches) < 2:
            logger.debug("Feature diversity: need >= 2 patches")
            return {
                "feature_diversity_mean_dist": 0.0,
                "feature_diversity_min_dist": 0.0,
            }

        # 提取特征向量
        feats = np.stack([c.feature.values for c in selected_patches])

        # 成对距离
        n = len(feats)
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(feats[i] - feats[j]))
                dists.append(d)

        mean_dist = float(np.mean(dists)) if dists else 0.0
        min_dist = float(np.min(dists)) if dists else 0.0

        return {
            "feature_diversity": mean_dist,  # 主指标，与 metric_name 对应
            "feature_diversity_mean_dist": mean_dist,
            "feature_diversity_min_dist": min_dist,
        }
