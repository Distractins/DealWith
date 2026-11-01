# -*- coding: utf-8 -*-
"""采样器注册表，通过装饰器实现自动注册。

所有采样器均通过 @register_sampler 装饰器自动发现。
调用 get_sampler(name, config) 即可实例化任意采样器。
"""

import logging
from typing import Dict, Type, Optional

from common.dataclasses import ConfigBundle
from samplers.base_sampler import BaseSampler

logger = logging.getLogger(__name__)

#: 全局采样器注册表: algorithm_name -> SamplerClass
SAMPLER_REGISTRY: Dict[str, Type[BaseSampler]] = {}


def register_sampler(cls: Type[BaseSampler]) -> Type[BaseSampler]:
    """装饰器：将采样器类注册到全局注册表中。

    用法：
        @register_sampler
        class GridSampler(BaseSampler):
            ...

    返回：
        原封不动返回该类。
    """
    name = cls.algorithm_name()
    SAMPLER_REGISTRY[name] = cls
    logger.debug(f"Registered sampler: {name} -> {cls.__name__}")
    return cls


def get_sampler(name: str, config: ConfigBundle) -> BaseSampler:
    """按名称实例化采样器。

    参数：
        name: 算法名称字符串（例如 'grid'、'sentinel'）。
        config: 配置包。

    返回：
        BaseSampler 实例。

    异常：
        ValueError: 如果采样器名称未注册。
    """
    # 确保所有采样器均已导入（触发注册）
    _ensure_samplers_imported()

    if name not in SAMPLER_REGISTRY:
        available = list(SAMPLER_REGISTRY.keys())
        raise ValueError(
            f"Unknown sampler: '{name}'. "
            f"Available: {available}"
        )

    return SAMPLER_REGISTRY[name](config)


def list_samplers() -> Dict[str, str]:
    """列出所有已注册的采样器。

    返回：
        字典，映射 algorithm_name -> 类显示名称。
    """
    _ensure_samplers_imported()
    return {name: cls.name for name, cls in SAMPLER_REGISTRY.items()}


def _ensure_samplers_imported() -> None:
    """惰性导入所有采样器模块以触发注册。"""
    if SAMPLER_REGISTRY:
        return  # 已导入

    # 导入所有采样器模块以触发 @register_sampler
    try:
        from samplers.random_sampler import RandomSampler  # noqa: F401
        from samplers.grid_sampler import GridSampler  # noqa: F401
        from samplers.largest_tissue_sampler import LargestTissueSampler  # noqa: F401
        from samplers.stratified_sampler import StratifiedSampler  # noqa: F401
        from samplers.kmeans_sampler import KMeansSampler  # noqa: F401
        from samplers.yottixel_sampler import YottixelSampler  # noqa: F401
        from samplers.splice_sampler import SpliceSampler  # noqa: F401
        from samplers.sdm_sampler import SDMSampler  # noqa: F401
        from samplers.sentinel_sampler import SentinelSampler  # noqa: F401
    except ImportError as e:
        logger.warning(f"Some samplers could not be imported: {e}")
