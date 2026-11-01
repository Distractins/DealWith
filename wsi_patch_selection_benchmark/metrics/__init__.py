# -*- coding: utf-8 -*-
"""用于评估贴片选择质量的评价指标。

每个指标独立于选择过程，提供贴片集质量的定量度量：
- 空间覆盖率：贴片对组织区域的覆盖程度
- 特征多样性：特征空间中贴片之间的平均成对距离
- 冗余率：相似贴片对的比例
- 覆盖区域数：包含至少一个贴片的空间分箱数量
"""

from metrics.base_metric import BaseMetric
from metrics.spatial_coverage import SpatialCoverageMetric
from metrics.feature_diversity import FeatureDiversityMetric
from metrics.redundancy_rate import RedundancyRateMetric
from metrics.covered_region_count import CoveredRegionCountMetric

__all__ = [
    "BaseMetric",
    "SpatialCoverageMetric",
    "FeatureDiversityMetric",
    "RedundancyRateMetric",
    "CoveredRegionCountMetric",
]
