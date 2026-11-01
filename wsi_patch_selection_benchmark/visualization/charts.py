# -*- coding: utf-8 -*-
"""图表可视化：基准测试结果的柱状图、箱线图、雷达图、热力图。"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from common.constants import METHOD_COLORS
from visualization.base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # 禁用圆角以避免旧版 freetype DLL 的 bezier 崩溃 (0xc06d007f)
    plt.rcParams['patch.force_edgecolor'] = True
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class ChartVisualizer(BaseVisualizer):
    """从聚合的基准测试结果生成对比图表。

    图表包括:
    - 柱状图：每种方法的指标对比
    - 雷达图：每种方法的多指标画像
    - 热力图：方法 × 指标矩阵
    """

    def draw_bar(
        self,
        metric_summary: Dict[str, Dict[str, Dict[str, float]]],
        metric_name: str,
        output_path: Path,
    ) -> None:
        """绘制比较各方法在某项指标上的柱状图。

        Args:
            metric_summary: 来自 BatchEvaluator 的数据。
            metric_name: 要绘制的指标名称。
            output_path: 基础输出路径。
        """
        if not _HAS_MPL:
            return

        methods = sorted(metric_summary.keys())
        means = []
        stds = []
        colors = []

        for m in methods:
            stats = metric_summary[m].get(metric_name, {})
            means.append(stats.get("mean", 0.0))
            stds.append(stats.get("std", 0.0))
            colors.append(METHOD_COLORS.get(m, "#333333"))

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, alpha=0.85)

        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title(f"Method Comparison: {metric_name.replace('_', ' ').title()}")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")

        # 数值标签
        for bar, mean_val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{mean_val:.3f}", ha="center", va="bottom", fontsize=8,
            )

        plt.tight_layout()
        self._save_figure(fig, output_path)

    def draw_radar(
        self,
        metric_summary: Dict[str, Dict[str, Dict[str, float]]],
        output_path: Path,
    ) -> None:
        """绘制显示多指标画像的雷达图。

        Args:
            metric_summary: 来自 BatchEvaluator 的数据。
            output_path: 基础输出路径。
        """
        if not _HAS_MPL:
            return

        # 收集指标键名（排除 _stats）
        metric_keys = []
        first_algo = next(iter(metric_summary.values()))
        for key in first_algo:
            if not key.startswith("_"):
                metric_keys.append(key)
        metric_keys = sorted(metric_keys)

        if len(metric_keys) < 3:
            logger.debug("Radar chart needs >= 3 metrics")
            return

        n_metrics = len(metric_keys)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # 闭合圆形

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for algo_name in sorted(metric_summary.keys()):
            values = []
            for mk in metric_keys:
                stats = metric_summary[algo_name].get(mk, {})
                values.append(stats.get("mean", 0.0))
            values += values[:1]  # 闭合

            color = METHOD_COLORS.get(algo_name, "#333333")
            ax.plot(angles, values, "o-", linewidth=2, label=algo_name, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([mk.replace("_", " ").title() for mk in metric_keys],
                           fontsize=8)
        ax.set_title("Method Comparison: Multi-Metric Radar")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=7)

        plt.tight_layout()
        self._save_figure(fig, output_path)

    def draw_heatmap(
        self,
        metric_summary: Dict[str, Dict[str, Dict[str, float]]],
        output_path: Path,
    ) -> None:
        """绘制方法 × 指标热力图。

        Args:
            metric_summary: 来自 BatchEvaluator 的数据。
            output_path: 基础输出路径。
        """
        if not _HAS_MPL:
            return

        # 收集指标键名
        metric_keys = []
        first_algo = next(iter(metric_summary.values()))
        for key in first_algo:
            if not key.startswith("_"):
                metric_keys.append(key)
        metric_keys = sorted(metric_keys)

        if not metric_keys:
            return

        methods = sorted(metric_summary.keys())
        data_matrix = np.zeros((len(methods), len(metric_keys)))

        for i, algo in enumerate(methods):
            for j, mk in enumerate(metric_keys):
                stats = metric_summary[algo].get(mk, {})
                data_matrix[i, j] = stats.get("mean", 0.0)

        fig, ax = plt.subplots(figsize=(max(8, len(metric_keys) * 1.5),
                                        max(5, len(methods) * 0.5)))
        im = ax.imshow(data_matrix, aspect="auto", cmap="YlOrRd")

        ax.set_xticks(range(len(metric_keys)))
        ax.set_xticklabels(
            [mk.replace("_", " ").title() for mk in metric_keys],
            rotation=45, ha="right", fontsize=8,
        )
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=8)

        # 添加文本注释
        for i in range(len(methods)):
            for j in range(len(metric_keys)):
                ax.text(j, i, f"{data_matrix[i, j]:.3f}",
                        ha="center", va="center", fontsize=7)

        ax.set_title("Method × Metric Heatmap")
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        self._save_figure(fig, output_path)
