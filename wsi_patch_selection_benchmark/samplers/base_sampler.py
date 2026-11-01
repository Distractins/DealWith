# -*- coding: utf-8 -*-
"""所有 patch 选择算法的抽象基类。

所有采样器均继承自 BaseSampler 并实现 select_patches()。
它们接收一个预先构建好的 CandidatePool —— 采样器不会重新扫描 WSI。
"""

from abc import ABC, abstractmethod
from typing import List

from common.dataclasses import CandidatePatch, ConfigBundle
from core.candidate_pool import CandidatePool


class BaseSampler(ABC):
    """所有 patch 选择算法的抽象基类。

    子类必须实现：
    - select_patches()：核心选择逻辑
    - algorithm_name()：唯一标识符字符串

    类级别属性：
    - name：人类可读的显示名称
    """

    name: str = "BaseSampler"

    def __init__(self, config: ConfigBundle):
        """使用配置初始化采样器。

        参数:
            config: 包含所有参数的配置包。
        """
        self.config = config

    @abstractmethod
    def select_patches(
        self,
        candidate_pool: CandidatePool,
        num_patches: int,
    ) -> List[CandidatePatch]:
        """从候选池中选择 patch。

        参数:
            candidate_pool: 预先构建的候选 patch 池，包含
                质量指标、评分和特征向量。
            num_patches: 需要选择的目标 patch 数量（K）。

        返回:
            选中的 CandidatePatch 对象列表。如果候选池不足，
            列表长度可能小于 num_patches。
        """
        ...

    @staticmethod
    @abstractmethod
    def algorithm_name() -> str:
        """返回唯一的算法标识符字符串。

        该值必须与 common.enums.AlgorithmName 中的某个值匹配。
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
