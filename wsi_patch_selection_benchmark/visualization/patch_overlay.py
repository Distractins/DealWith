# -*- coding: utf-8 -*-
"""补丁叠加可视化：在 WSI 缩略图上显示选中的补丁。"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from common.dataclasses import CandidatePatch, ConfigBundle
from common.constants import METHOD_COLORS
from visualization.base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class PatchOverlayVisualizer(BaseVisualizer):
    """在 WSI 缩略图上绘制选中的补丁叠加。

    每个补丁绘制为一个彩色矩形。
    保存为: {case_id}_{algorithm}_overlay.png
    """

    def draw(
        self,
        data: dict,
        output_path: Path,
    ) -> None:
        """绘制补丁叠加图。

        Args:
            data: 包含以下键的字典:
                - 'thumbnail': PIL 缩略图图像对象
                - 'patches': 补丁列表，每个元素为 (cx, cy, patch_size, label)，
                  其中 cx/cy 为补丁中心坐标、patch_size 为补丁尺寸、label 为标签
                - 'title': str，图表标题
                - 'color': str（可选，十六进制颜色码）
            output_path: 基础输出路径。
        """
        if not _HAS_MPL:
            logger.warning("matplotlib not available, skipping patch overlay")
            return

        thumbnail = data.get("thumbnail")
        patches = data.get("patches", [])
        title = data.get("title", "Patch Overlay")
        color = data.get("color", "#E41A1C")

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(thumbnail)

        # 确定 level-0 与缩略图之间的缩放比例
        w0 = data.get("wsi_width", 1)
        h0 = data.get("wsi_height", 1)
        scale_x = thumbnail.size[0] / w0 if w0 > 0 else 1
        scale_y = thumbnail.size[1] / h0 if h0 > 0 else 1

        for cx, cy, patch_size, label in patches:
            # 转换为缩略图坐标
            tx = (cx - patch_size // 2) * scale_x
            ty = (cy - patch_size // 2) * scale_y
            tw = patch_size * scale_x
            th = patch_size * scale_y

            rect = mpatches.Rectangle(
                (tx, ty), tw, th,
                linewidth=1.5,
                edgecolor=color,
                facecolor="none",
                alpha=0.8,
            )
            ax.add_patch(rect)

        ax.set_title(title)
        ax.axis("off")

        self._save_figure(fig, output_path)
