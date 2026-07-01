# -*- coding: utf-8 -*-
"""抽象基类可视化器。"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from common.dataclasses import ConfigBundle


class BaseVisualizer(ABC):
    """所有可视化器的抽象基类。

    子类通过实现 draw() 来生成特定类型的图表。
    """

    def __init__(self, config: ConfigBundle):
        """使用配置进行初始化。

        Args:
            config: 配置包。
        """
        self.config = config
        self.dpi = config.dpi
        self.formats = config.figure_formats

    def _save_figure(self, fig, output_path: Path) -> None:
        """以所有配置的格式保存图表。

        Args:
            fig: matplotlib 的 Figure 对象。
            output_path: 基础路径（不含扩展名）。
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for fmt in self.formats:
            path = output_path.with_suffix(f".{fmt}")
            fig.savefig(str(path), dpi=self.dpi, bbox_inches="tight")
        from matplotlib import pyplot as plt
        plt.close(fig)
