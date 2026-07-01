# -*- coding: utf-8 -*-
"""用于评估贴片选择质量的抽象基类指标。"""

from abc import ABC, abstractmethod
from typing import Dict, List

from common.dataclasses import CandidatePatch


class BaseMetric(ABC):
    """所有评价指标的抽象基类。

    子类从一组选定的贴片中计算特定的质量度量。
    指标独立于选择算法。
    """

    name: str = "BaseMetric"

    @abstractmethod
    def evaluate(
        self,
        selected_patches: List[CandidatePatch],
    ) -> Dict[str, float]:
        """对一组选定的贴片进行指标评估。

        Args:
            selected_patches: 选定的 CandidatePatch 对象列表。

        Returns:
            将指标名称映射到计算值的字典。
        """
        ...

    @staticmethod
    @abstractmethod
    def metric_name() -> str:
        """返回唯一的指标标识符字符串。"""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
