# -*- coding: utf-8 -*-
"""logger.py - 中英文日志配置 (控制台 + 文件)。"""

import logging
import sys
from pathlib import Path

_LEVEL_ZH = {"DEBUG": "调试", "INFO": "信息", "WARNING": "警告", "ERROR": "错误", "CRITICAL": "严重"}


class _ZhFormatter(logging.Formatter):
    def __init__(self, fmt, language="zh_CN"):
        super().__init__(fmt)
        self.language = language

    def format(self, record):
        if self.language == "zh_CN":
            record.levelname_zh = _LEVEL_ZH.get(record.levelname, record.levelname)
        else:
            record.levelname_zh = record.levelname
        return super().format(record)


def setup_logger(log_file=None, level="INFO", language="zh_CN", name="MucusMIL") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = "%(asctime)s [%(levelname_zh)s] %(message)s" if language == "zh_CN" else \
          "%(asctime)s [%(levelname)s] %(message)s"
    formatter = _ZhFormatter(fmt, language)
    formatter.default_time_format = "%Y-%m-%d %H:%M:%S"

    try:
        if hasattr(sys.stdout, "reconfigure") and not sys.stdout.isatty():
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(p), encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info(f"日志文件: {p}")

    return logger
