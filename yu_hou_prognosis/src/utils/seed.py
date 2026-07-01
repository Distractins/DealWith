# -*- coding: utf-8 -*-
"""
seed.py
============================================================================
随机种子设置模块，确保实验可复现。

功能:
    1. 统一设置Python、NumPy、PyTorch的随机种子
    2. 可选确定性模式（牺牲性能换可复现性）
    3. 记录种子设置日志

使用示例:
    from src.utils.seed import set_seed
    set_seed(2026)
============================================================================
"""

import random
import logging
import numpy as np
import torch


def set_seed(seed: int = 2026, deterministic: bool = False,
             logger: logging.Logger = None) -> None:
    """
    设置所有随机数生成器的种子。

    参数:
        seed: 随机种子值
        deterministic: 是否启用确定性模式。
                       True时禁用cudnn.benchmark并启用确定性算法，
                       牺牲约10-20%的训练速度换取完全可复现结果。
        logger: 可选的日志器实例

    注意:
        确定性模式会显著降低训练速度，仅在需要完全复现结果时启用。
        日常训练建议使用默认的非确定性模式（deterministic=False）。
    """
    # Python随机
    random.seed(seed)

    # NumPy随机
    np.random.seed(seed)

    # PyTorch随机
    torch.manual_seed(seed)

    # CUDA随机
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 确定性配置
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        # 非确定性模式：允许cudnn自动寻找最优卷积算法
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    if logger:
        logger.info(f"随机种子已设置: {seed}")
        if deterministic:
            logger.info("已启用确定性模式（训练速度可能降低10-20%）")
        else:
            logger.info("已启用cudnn.benchmark（自动寻找最优卷积算法）")


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker的随机种子初始化函数。

    用于DataLoader的worker_init_fn参数，确保每个worker使用不同的种子。

    使用示例:
        dataloader = DataLoader(
            dataset,
            worker_init_fn=seed_worker,
        )
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int = 2026, device: str = "cpu") -> torch.Generator:
    """
    创建一个指定种子的PyTorch随机数生成器。

    参数:
        seed: 随机种子
        device: 设备类型

    返回:
        torch.Generator: 随机数生成器
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g
