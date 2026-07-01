# -*- coding: utf-8 -*-
"""论文质量图表的生成。

将可视化原语组合成可用于发表的图表，
具有一致的样式、LaTeX 兼容的标签和高 DPI。
"""

import logging
from pathlib import Path
from typing import Dict

from common.dataclasses import ConfigBundle
from visualization.base_visualizer import BaseVisualizer
from visualization.charts import ChartVisualizer

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    # 使用衬线字体用于发表
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class PaperFigureVisualizer(BaseVisualizer):
    """生成可用于发表的高质量组合图表。

    保存到: outputs/figures/paper_figures/
    """

    def draw_all(
        self,
        metric_summary: Dict[str, Dict[str, Dict[str, float]]],
        figures_dir: Path,
    ) -> None:
        """生成所有论文图表。

        Args:
            metric_summary: 来自 BatchEvaluator 的数据。
            figures_dir: 图表的输出目录。
        """
        chart_viz = ChartVisualizer(self.config)
        paper_dir = figures_dir / "paper_figures"
        paper_dir.mkdir(parents=True, exist_ok=True)

        # 图 1: 空间覆盖度的柱状图
        chart_viz.draw_bar(
            metric_summary, "spatial_coverage",
            paper_dir / "fig1_spatial_coverage",
        )

        # 图 2: 特征多样性的柱状图
        chart_viz.draw_bar(
            metric_summary, "feature_diversity",
            paper_dir / "fig2_feature_diversity",
        )

        # 图 3: 冗余率的柱状图
        chart_viz.draw_bar(
            metric_summary, "redundancy_rate",
            paper_dir / "fig3_redundancy_rate",
        )

        # 图 4: 覆盖区域数量的柱状图
        chart_viz.draw_bar(
            metric_summary, "covered_region_count",
            paper_dir / "fig4_covered_regions",
        )

        # 图 5: 雷达图（多指标概览）
        chart_viz.draw_radar(
            metric_summary,
            paper_dir / "fig5_radar",
        )

        # 图 6: 热力图
        chart_viz.draw_heatmap(
            metric_summary,
            paper_dir / "fig6_heatmap",
        )

        logger.info(f"Paper figures saved to {paper_dir}")
