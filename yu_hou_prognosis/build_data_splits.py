# -*- coding: utf-8 -*-
"""
build_data_splits.py
============================================================================
构建完整数据划分文件。

此脚本现已委托给 cv_runner._auto_build_full_splits()，与 main.py 首次运行时的
自动构建逻辑完全一致，避免两套逻辑不一致导致的标签/特征错误：
  - 基因组特征做方差过滤 (突变全保留 + RNA-seq/其他按方差取top)
  - 按任务类型构造标签 (surv 用占位 / ncls、n_binary 用N分期)
  - 基于训练集拟合 StandardScaler 并应用到 train/test

使用流程:
    1. python make_split.py          # 生成病人ID级划分 (按任务分层)
    2. python build_data_splits.py   # 构建完整数据划分 (本脚本，可选)
    3. python main.py                # 开始训练

注意: 第2步可省略，main.py 在加载到“仅含病人ID”的pkl时会自动构建完整划分。
"""

import sys
import logging
from pathlib import Path
import pickle

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import load_config
from src.training.cv_runner import _auto_build_full_splits


def build_full_data_splits(config):
    """从 make_split.py 生成的病人ID划分构建完整数据划分。

    参数:
        config: ConfigBundle配置对象

    返回:
        output_folds: 完整数据划分字典 {"fold_1": {"train": {...}, "test": {...}}, ...}
    """
    id_split_path = config.resolve_path(config.data.split_file)
    if not id_split_path.exists():
        raise FileNotFoundError(
            f"病人ID划分文件不存在: {id_split_path}\n请先运行: python make_split.py"
        )

    with open(id_split_path, "rb") as f:
        raw = pickle.load(f)

    # 解析病人ID级划分 (支持 cv_splits 包裹 与 直接 fold 字典两种格式)
    if isinstance(raw, dict) and "cv_splits" in raw:
        id_splits = raw["cv_splits"]
    elif isinstance(raw, dict) and all(isinstance(k, int) for k in raw):
        id_splits = raw
    else:
        raise ValueError(f"无法识别的划分文件格式: {type(raw)}")

    logger = logging.getLogger("YuHou.build_splits")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    return _auto_build_full_splits(id_splits, config, logger)


if __name__ == "__main__":
    config = load_config()
    build_full_data_splits(config)
