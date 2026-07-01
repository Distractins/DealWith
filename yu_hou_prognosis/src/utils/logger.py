# -*- coding: utf-8 -*-
"""
logger.py
============================================================================
中文日志配置模块。

功能:
    1. 支持中文/英文双语日志输出
    2. 同时输出到文件和控制台
    3. 自动创建日志目录
    4. 格式化日志消息，包含时间戳和级别
    5. 支持不同级别的日志着色（控制台输出）

使用示例:
    from src.utils.logger import setup_logger
    logger = setup_logger("logs/train.log", level="INFO", language="zh_CN")
    logger.info("训练开始，设备: %s", device)
============================================================================
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional


# ============================================================
# 日志级别中英文映射
# ============================================================
_LEVEL_ZH = {
    "DEBUG": "调试",
    "INFO": "信息",
    "WARNING": "警告",
    "ERROR": "错误",
    "CRITICAL": "严重",
}


# ============================================================
# 控制台日志着色（Windows ANSI支持）
# ============================================================
class _ColoredFormatter(logging.Formatter):
    """
    带颜色编码的控制台日志格式化器。

    在Windows终端中自动启用ANSI颜色支持。
    """
    # ANSI颜色码
    COLORS = {
        "DEBUG": "\033[36m",      # 青色
        "INFO": "\033[32m",       # 绿色
        "WARNING": "\033[33m",    # 黄色
        "ERROR": "\033[31m",      # 红色
        "CRITICAL": "\033[35m",   # 紫色
        "RESET": "\033[0m",       # 重置
    }

    def __init__(self, fmt: str, language: str = "zh_CN"):
        super().__init__(fmt)
        self.language = language
        # 在Windows上启用ANSI
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except Exception:
                pass

    def format(self, record: logging.LogRecord) -> str:
        # 替换级别名为中文
        if self.language == "zh_CN":
            zh_level = _LEVEL_ZH.get(record.levelname, record.levelname)
            record.levelname_zh = zh_level
        else:
            record.levelname_zh = record.levelname

        # 添加颜色
        color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS.get("RESET", "")

        formatted = super().format(record)

        if color and sys.stdout.isatty():
            formatted = f"{color}{formatted}{reset}"

        return formatted


# ============================================================
# 日志器创建
# ============================================================

def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    language: str = "zh_CN",
    log_format: Optional[str] = None,
    console: bool = True,
    name: str = "YuHou",
) -> logging.Logger:
    """
    创建并配置日志器。

    参数:
        log_file: 日志文件路径（相对于项目根目录或绝对路径）。
                  如果为None，仅输出到控制台。
        level: 日志级别 ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        language: 日志语言 ("zh_CN" 或 "en_US")
        log_format: 自定义日志格式。如果为None，使用默认格式。
        console: 是否同时输出到控制台
        name: 日志器名称

    返回:
        logging.Logger: 配置好的日志器实例

    使用示例:
        logger = setup_logger("experiments/default/logs/train.log")
        logger.info("模型初始化完成，总参数量: %d", total_params)
        logger.warning("检测到低质量图像比例: %.1f%%", blur_ratio * 100)
        logger.error("训练中断: CUDA内存不足")
    """
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    # 默认日志格式
    if log_format is None:
        if language == "zh_CN":
            log_format = "%(asctime)s [%(levelname_zh)s] %(message)s"
        else:
            log_format = "%(asctime)s [%(levelname)s] %(message)s"

    date_format = "%Y-%m-%d %H:%M:%S"

    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logger.level)
        console_formatter = _ColoredFormatter(log_format, language=language)
        console_formatter.default_time_format = date_format
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if log_file is not None:
        # 确保日志目录存在
        log_path = Path(log_file)
        if not log_path.is_absolute():
            # 相对于项目根目录
            from pathlib import Path as _Path
            project_root = _Path(__file__).resolve().parents[2]
            log_path = project_root / log_path

        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f"日志文件: {log_path}")

    return logger


def get_logger(name: str = "YuHou") -> logging.Logger:
    """
    获取已存在的日志器实例。

    如果不存在则创建一个默认的控制台日志器。

    参数:
        name: 日志器名称

    返回:
        logging.Logger: 日志器实例
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name=name)
    return logger


# ============================================================
# 便捷函数
# ============================================================

def log_section(logger: logging.Logger, title: str, width: int = 60) -> None:
    """
    在日志中输出分隔标题。

    参数:
        logger: 日志器实例
        title: 标题文本
        width: 分隔线宽度
    """
    logger.info("=" * width)
    logger.info(title)
    logger.info("=" * width)


def log_dict(logger: logging.Logger, data: dict, indent: int = 2, prefix: str = "") -> None:
    """
    以易读格式输出字典内容到日志。

    参数:
        logger: 日志器实例
        data: 要输出的字典
        indent: 缩进空格数
        prefix: 行前缀
    """
    for key, value in data.items():
        if isinstance(value, dict):
            logger.info(f"{prefix}{' ' * indent}{key}:")
            log_dict(logger, value, indent + 2, prefix)
        elif isinstance(value, list):
            logger.info(f"{prefix}{' ' * indent}{key}: {value}")
        else:
            logger.info(f"{prefix}{' ' * indent}{key}: {value}")


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    logger = setup_logger(level="DEBUG")

    logger.debug("这是一条调试信息")
    logger.info("这是一条普通信息")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    logger.critical("这是一条严重错误信息")

    log_section(logger, "配置参数")
    log_dict(logger, {
        "模型名称": "YuHou",
        "融合策略": "CAUGF",
        "批次大小": 4,
        "学习率": 0.0001,
    })

    print("\n日志器测试完成!")
