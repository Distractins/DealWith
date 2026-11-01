# -*- coding: utf-8 -*-
"""用于基准测试流水线性能分析的时间测量工具。"""

import logging
import time
from contextlib import contextmanager
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


class Timer:
    """一个简单的挂钟计时器。

    用法:
        timer = Timer("tissue_mask")
        with timer:
            mask = build_hybrid_tissue_mask(slide)

        # 或手动使用:
        timer.start()
        ...
        elapsed = timer.stop()
    """

    def __init__(self, name: str = ""):
        """初始化计时器。

        Args:
            name: 该计时器的描述名称（用于日志消息）。
        """
        self.name = name
        self._start: float = 0.0
        self._elapsed: float = 0.0
        self._running: bool = False

    def start(self) -> None:
        """启动（或重启）计时器。"""
        self._start = time.perf_counter()
        self._running = True

    def stop(self) -> float:
        """停止计时器并返回经过的秒数。

        Returns:
            经过的时间（以秒为单位）。
        """
        if not self._running:
            logger.warning(f"Timer '{self.name}': stop() called but not running")
            return self._elapsed
        self._elapsed = time.perf_counter() - self._start
        self._running = False
        return self._elapsed

    @property
    def elapsed(self) -> float:
        """获取经过的时间（以秒为单位）。"""
        if self._running:
            return time.perf_counter() - self._start
        return self._elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        elapsed = self.stop()
        if self.name:
            logger.debug(f"[{self.name}] completed in {elapsed:.3f}s")


@contextmanager
def timed(name: str = ""):
    """用于对代码块计时的上下文管理器。

    用法:
        with timed("feature_extraction"):
            metrics = patch_quality_metrics(patch)

    Args:
        name: 被计时代码块的描述名称。
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        if name:
            logger.debug(f"[{name}] completed in {elapsed:.3f}s")
