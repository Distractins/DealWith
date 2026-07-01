# -*- coding: utf-8 -*-
"""围绕 openslide.OpenSlide 的轻量封装，用于 WSI 读取。

提供上下文管理器支持和错误处理。
此处不应包含任何算法或特征提取逻辑。
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import openslide

logger = logging.getLogger(__name__)


class WSIReader:
    """通过 OpenSlide 读取 WSI 切片的上下文管理器。

    用法:
        with WSIReader("path/to/slide.svs") as slide:
            img = slide.read_region((0, 0), 0, (1024, 1024))

        # 或使用类方法:
        slide = WSIReader.open("path/to/slide.svs")
        ...
        WSIReader.close(slide)
    """

    def __init__(self, slide_path: str):
        """初始化读取器。

        Args:
            slide_path: WSI 文件的路径（.svs, .tiff 等）。
        """
        self.slide_path = Path(slide_path)
        self._slide: Optional[openslide.OpenSlide] = None

    # ------------------------------------------------------------------
    # 上下文管理器
    # ------------------------------------------------------------------
    def __enter__(self) -> openslide.OpenSlide:
        self._slide = self._open_internal()
        return self._slide

    def __exit__(self, *args):
        self._close_internal()

    # ------------------------------------------------------------------
    # 类方法（用于显式打开/关闭）
    # ------------------------------------------------------------------
    @staticmethod
    def open(slide_path: str) -> openslide.OpenSlide:
        """打开一张 WSI 切片。

        Args:
            slide_path: 切片文件路径。

        Returns:
            一个 openslide.OpenSlide 对象。

        Raises:
            FileNotFoundError: 如果切片文件不存在。
            openslide.OpenSlideError: 如果切片无法打开。
        """
        path = Path(slide_path)
        if not path.exists():
            raise FileNotFoundError(f"Slide file not found: {slide_path}")

        try:
            slide = openslide.OpenSlide(str(path))
            logger.debug(f"Opened slide: {path.name} "
                         f"({slide.dimensions[0]}x{slide.dimensions[1]})")
            return slide
        except Exception as e:
            logger.error(f"Failed to open slide {slide_path}: {e}")
            raise

    @staticmethod
    def close(slide: openslide.OpenSlide) -> None:
        """关闭一个 OpenSlide 对象。

        Args:
            slide: 要关闭的 OpenSlide 对象。
        """
        try:
            slide.close()
        except Exception as e:
            logger.warning(f"Error closing slide: {e}")

    @staticmethod
    def get_dimensions(slide: openslide.OpenSlide) -> Tuple[int, int]:
        """获取切片在 level 0 下的尺寸。

        Args:
            slide: 一个 OpenSlide 对象。

        Returns:
            (宽度, 高度) 的元组，以像素为单位。
        """
        return slide.dimensions

    @staticmethod
    def get_thumbnail(
        slide: openslide.OpenSlide,
        size: Tuple[int, int],
    ) -> "PIL.Image.Image":
        """获取切片的下采样缩略图。

        Args:
            slide: 一个 OpenSlide 对象。
            size: 缩略图的目标 (宽度, 高度)。

        Returns:
            RGB 模式的 PIL 图像。
        """
        return slide.get_thumbnail(size).convert("RGB")

    @staticmethod
    def read_region(
        slide: openslide.OpenSlide,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
    ) -> "PIL.Image.Image":
        """从切片中读取一个区域。

        Args:
            slide: 一个 OpenSlide 对象。
            location: (x, y) 左上角坐标，使用 level-0 坐标。
            level: 金字塔层级。
            size: 区域的 (宽度, 高度)。

        Returns:
            RGB 模式的 PIL 图像。
        """
        return slide.read_region(location, level, size).convert("RGB")

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------
    def _open_internal(self) -> openslide.OpenSlide:
        return self.open(str(self.slide_path))

    def _close_internal(self) -> None:
        if self._slide is not None:
            self.close(self._slide)
            self._slide = None
