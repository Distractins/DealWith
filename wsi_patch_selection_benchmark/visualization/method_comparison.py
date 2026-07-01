# -*- coding: utf-8 -*-
"""方法对比可视化：多方法并排视图。

在同一张 WSI 缩略图上显示所有方法的选中补丁，
每种方法使用不同的颜色。
"""

import logging
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from common.dataclasses import CandidatePatch, ConfigBundle
from common.constants import METHOD_COLORS
from visualization.base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class MethodComparisonVisualizer(BaseVisualizer):
    """在同一张 WSI 上进行所有方法的并排对比。

    保存为: {case_id}_comparison.png
    """

    def draw(
        self,
        data: dict,
        output_path: Path,
    ) -> None:
        """绘制方法对比图。

        Args:
            data: 包含以下键的字典:
                - 'thumbnail': PIL 图像
                - 'method_patches': Dict[str, List[CandidatePatch]]
                - 'wsi_width': int
                - 'wsi_height': int
                - 'patch_size': int
                - 'case_id': str
            output_path: 基础输出路径。
        """
        if not _HAS_MPL:
            logger.warning("matplotlib not available, skipping comparison")
            return

        thumbnail = data.get("thumbnail")
        method_patches = data.get("method_patches", {})
        w0 = data.get("wsi_width", 1)
        h0 = data.get("wsi_height", 1)
        patch_size = data.get("patch_size", 1024)
        case_id = data.get("case_id", "")

        methods = sorted(method_patches.keys())
        n_methods = len(methods)
        if n_methods == 0:
            return

        # 网格布局
        n_cols = min(3, n_methods)
        n_rows = math.ceil(n_methods / n_cols)

        scale_x = thumbnail.size[0] / w0 if w0 > 0 else 1
        scale_y = thumbnail.size[1] / h0 if h0 > 0 else 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_methods == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax, method in zip(axes, methods):
            ax.imshow(thumbnail)
            color = METHOD_COLORS.get(method, "#333333")
            patches = method_patches[method]

            for c in patches:
                tx = (c.patch.cx - patch_size // 2) * scale_x
                ty = (c.patch.cy - patch_size // 2) * scale_y
                tw = patch_size * scale_x
                th = patch_size * scale_y

                rect = plt.Rectangle(
                    (tx, ty), tw, th,
                    linewidth=2, edgecolor=color,
                    facecolor="none", alpha=0.9,
                )
                ax.add_patch(rect)

            ax.set_title(f"{method} ({len(patches)} patches)")
            ax.axis("off")

        # 隐藏未使用的坐标轴
        for ax in axes[n_methods:]:
            ax.axis("off")

        fig.suptitle(f"Method Comparison: {case_id}", fontsize=14)
        plt.tight_layout()
        self._save_figure(fig, output_path)
