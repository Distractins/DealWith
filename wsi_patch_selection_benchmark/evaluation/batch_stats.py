# -*- coding: utf-8 -*-
"""批量统计：计算所有病例的聚合指标。

对每个算法，收集每个病例所选出的所有图像块，
计算每个病例的指标，然后跨病例聚合（均值/标准差）。
"""

import logging
from typing import Dict, List

import numpy as np

from common.dataclasses import AlgorithmResult, CandidatePatch
from metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)


class BatchEvaluator:
    """跨所有病例计算聚合统计量。

    用法：
        evaluator = BatchEvaluator()
        summary = evaluator.evaluate(all_results, metrics)
    """

    def evaluate(
        self,
        algo_results: Dict[str, AlgorithmResult],
        metrics: List[BaseMetric],
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """使用所有指标评估所有算法的结果。

        Args:
            algo_results: 算法名称 -> AlgorithmResult 的映射。
            metrics: BaseMetric 实例列表。

        Returns:
            嵌套字典：
            algo_name -> metric_name -> {
                'mean': float, 'std': float, 'min': float, 'max': float,
                'n_cases': int
            }
        """
        summary = {}

        for algo_name, algo_result in algo_results.items():
            algo_summary = self._evaluate_algorithm(algo_result, metrics)
            summary[algo_name] = algo_summary

        return summary

    def _evaluate_algorithm(
        self,
        algo_result: AlgorithmResult,
        metrics: List[BaseMetric],
    ) -> Dict[str, Dict[str, float]]:
        """评估单个算法的结果。

        收集所有病例中选出的所有图像块，计算每个病例的指标，
        并进行聚合。

        Args:
            algo_result: 单个算法的聚合结果。
            metrics: 指标列表。

        Returns:
            指标名称 -> 聚合统计字典。
        """
        algo_summary = {}

        # 收集每个病例的指标值
        metric_values: Dict[str, List[float]] = {
            m.metric_name(): [] for m in metrics
        }

        for case_result in algo_result.case_results:
            # 收集该病例的所有图像块
            all_patches: List[CandidatePatch] = []
            for slide_result in case_result.slide_results:
                all_patches.extend(slide_result.selected_patches)

            if len(all_patches) == 0:
                continue

            # 计算每个指标
            for metric in metrics:
                values = metric.evaluate(all_patches)
                # 使用主要指标值（第一个键）
                primary_key = metric.metric_name()
                if primary_key in values:
                    metric_values[primary_key].append(values[primary_key])

        # 聚合
        for m in metrics:
            key = m.metric_name()
            vals = metric_values.get(key, [])
            if vals:
                algo_summary[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "n_cases": len(vals),
                }
            else:
                algo_summary[key] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "n_cases": 0,
                }

        # 添加基本统计信息
        algo_summary["_stats"] = {
            "total_cases": algo_result.total_cases,
            "total_ok": algo_result.total_ok,
            "total_patches_saved": algo_result.total_patches_saved,
        }

        return algo_summary
