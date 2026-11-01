# -*- coding: utf-8 -*-
"""WSI 缩略图 + 组织掩膜可视化。"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from common.dataclasses import ConfigBundle
from visualization.base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class ThumbnailVisualizer(BaseVisualizer):
    """绘制带有组织掩膜叠加的 WSI 缩略图。

    保存为: {slide_base}_thumbnail.png
    """

    def draw(
        self,
        data: dict,
        output_path: Path,
    ) -> None:
        """绘制带有组织掩膜的缩略图。

        Args:
            data: 包含以下键的字典:
                - 'thumbnail': PIL 图像（WSI 缩略图）
                - 'mask': np.ndarray（组织掩膜，二值图）
                - 'title': str（切片名称）
            output_path: 基础输出路径。
        """
        if not _HAS_MPL:
            logger.warning("matplotlib not available, skipping thumbnail")
            return

        thumbnail = data.get("thumbnail")
        mask = data.get("mask")
        title = data.get("title", "WSI")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 原始缩略图
        axes[0].imshow(thumbnail)
        axes[0].set_title(f"{title} (Thumbnail)")
        axes[0].axis("off")

        # 缩略图 + 掩膜叠加
        axes[1].imshow(thumbnail)
        if mask is not None:
            # 调整掩膜尺寸以匹配缩略图
            mask_resized = np.array(Image.fromarray(
                (mask * 255).astype(np.uint8)
            ).resize(thumbnail.size, Image.NEAREST))
            mask_overlay = np.zeros((*mask_resized.shape, 4), dtype=np.float32)
            mask_overlay[mask_resized > 0] = [0, 1, 0, 0.3]  # 绿色，半透明
            axes[1].imshow(mask_overlay)
        axes[1].set_title(f"{title} (Tissue Mask)")
        axes[1].axis("off")

        plt.tight_layout()
        self._save_figure(fig, output_path)
