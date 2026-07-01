# -*- coding: utf-8 -*-
"""用于批量评估算法结果的抽象基类评估器。"""

from abc import ABC, abstractmethod
from typing import Dict, List

from common.dataclasses import AlgorithmResult
from metrics.base_metric import BaseMetric


class BaseEvaluator(ABC):
    """批量评估器的抽象基类。

    评估器接收聚合后的算法结果，并计算所有案例的汇总统计量。
    """

    @abstractmethod
    def evaluate(
        self,
        algo_results: Dict[str, AlgorithmResult],
        metrics: List[BaseMetric],
    ) -> Dict[str, Dict[str, float]]:
        """使用所有指标评估所有算法结果。

        Args:
            algo_results: 算法名称到 AlgorithmResult 的映射。
            metrics: 要计算的指标列表。

        Returns:
            嵌套字典：algo_name -> metric_name -> 平均值。
        """
        ...
