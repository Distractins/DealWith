# -*- coding: utf-8 -*-
"""WSI 补丁选择基准测试的统一日志设置。

提供同时支持控制台和文件输出的结构化日志。
流水线中的所有异常都通过此模块记录。
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_LOG_INITIALIZED = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """使用控制台和可选的日志文件处理器初始化根日志记录器。

    该方法应在流水线启动时调用一次。
    后续调用均为空操作。

    Args:
        level: 日志级别（DEBUG, INFO, WARNING, ERROR）。
        log_file: 日志文件路径。如果为 None，则不输出文件日志。
        log_format: 自定义日志格式字符串。如果为 None，则使用默认值。

    Returns:
        根日志记录器实例。
    """
    global _LOG_INITIALIZED
    if _LOG_INITIALIZED:
        return logging.getLogger()

    if log_format is None:
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 移除所有已有的处理器
    root_logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # 文件始终记录 DEBUG 级别
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    _LOG_INITIALIZED = True
    root_logger.info(f"Logging initialized at level {level}")
    return root_logger


def get_logger(name: str = "") -> logging.Logger:
    """获取一个命名的日志记录器。

    Args:
        name: 日志记录器名称（通常使用调用模块的 __name__）。

    Returns:
        已配置的日志记录器实例。
    """
    return logging.getLogger(name)
