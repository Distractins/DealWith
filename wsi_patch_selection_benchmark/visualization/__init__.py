# -*- coding: utf-8 -*-
"""WSI 补丁选择基准测试的可视化层。

自动生成:
- 带有组织掩膜叠加的 WSI 缩略图
- 补丁叠加（在 WSI 上显示选中的补丁）
- 方法对比（多方法并排视图）
- 图表（柱状图、箱线图、雷达图、热力图）
- 论文质量的图表
"""

from visualization.base_visualizer import BaseVisualizer
from visualization.thumbnail import ThumbnailVisualizer
from visualization.patch_overlay import PatchOverlayVisualizer
from visualization.method_comparison import MethodComparisonVisualizer
from visualization.charts import ChartVisualizer
from visualization.paper_figures import PaperFigureVisualizer

__all__ = [
    "BaseVisualizer",
    "ThumbnailVisualizer",
    "PatchOverlayVisualizer",
    "MethodComparisonVisualizer",
    "ChartVisualizer",
    "PaperFigureVisualizer",
]
