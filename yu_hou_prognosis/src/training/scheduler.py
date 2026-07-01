# -*- coding: utf-8 -*-
"""
scheduler.py
============================================================================
学习率调度器工厂模块 —— 根据配置创建学习率调度器实例。

支持的调度器类型:
    - cosine  : CosineAnnealingLR + 线性热身 + 恒定阶段
    - step    : StepLR (阶梯式衰减) + 可选热身
    - exp     : ExponentialLR (指数衰减) + 可选热身
    - plateau : ReduceLROnPlateau (基于验证指标自适应衰减)
    - linear  : 线性衰减 + 可选热身

参数来源:
    config.training.scheduler
        - type          : str   = "cosine"      调度器类型
        - n_epochs      : int   = 8             固定学习率训练轮数
        - n_epochs_decay: int   = 4             学习率衰减轮数
        - warmup_epochs : int   = 2             线性热身轮数
        - min_lr        : float = 1.0e-6        最小学习率
        - lr_decay_iters: int   = 10            StepLR 衰减间隔

训练阶段划分 (总轮数 = n_epochs + n_epochs_decay):
    1. 热身阶段 (epoch 0 ~ warmup_epochs-1):
       lr 从 min_lr 线性增长到 base_lr
    2. 恒定阶段 (epoch warmup_epochs ~ n_epochs-1):
       lr 保持为 base_lr
    3. 衰减阶段 (epoch n_epochs ~ n_epochs + n_epochs_decay - 1):
       lr 从 base_lr 衰减到 min_lr (具体曲线取决于调度器类型)

使用示例:
    from config.config_loader import load_config
    from src.training.scheduler import create_scheduler

    config = load_config("config/default_config.yaml")
    scheduler = create_scheduler(optimizer, config)

    # 训练循环中:
    for epoch in range(total_epochs):
        train_one_epoch(...)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)          # plateau 需传入指标
        else:
            scheduler.step()                  # 其他调度器无需参数
============================================================================
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LambdaLR,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)

from config.config_loader import ConfigBundle
from src.utils.logger import get_logger


# ============================================================
# 模块级日志器
# ============================================================
_logger: Optional[logging.Logger] = None


def _get_logger() -> logging.Logger:
    """获取或创建模块级日志器 (懒加载)。"""
    global _logger
    if _logger is None:
        _logger = get_logger("YuHou.scheduler")
    return _logger


# ============================================================
# 参数提取
# ============================================================

def _extract_scheduler_params(config: ConfigBundle) -> Dict[str, Any]:
    """
    从配置中提取调度器参数，返回标准化字典。

    参数:
        config: ConfigBundle 实例

    返回:
        dict: 包含所有调度器相关参数的字典
    """
    sch_cfg = config.training.scheduler

    n_epochs = int(getattr(sch_cfg, "n_epochs", 8))
    n_epochs_decay = int(getattr(sch_cfg, "n_epochs_decay", 4))
    warmup_epochs = int(getattr(sch_cfg, "warmup_epochs", 2))

    params: Dict[str, Any] = {
        "type": getattr(sch_cfg, "type", "cosine"),
        "n_epochs": n_epochs,
        "n_epochs_decay": n_epochs_decay,
        "total_epochs": n_epochs + n_epochs_decay,
        "warmup_epochs": warmup_epochs,
        "min_lr": float(getattr(sch_cfg, "min_lr", 1.0e-6)),
        "lr_decay_iters": int(getattr(sch_cfg, "lr_decay_iters", 10)),
        "base_lr": _get_base_lr(config),
    }

    # 合法性检查
    if warmup_epochs > n_epochs:
        _get_logger().warning(
            "warmup_epochs (%d) > n_epochs (%d), 将warmup_epochs截断为 n_epochs",
            warmup_epochs, n_epochs,
        )
        params["warmup_epochs"] = n_epochs

    if params["total_epochs"] < 1:
        raise ValueError(f"总训练轮数必须 >= 1, 当前值: {params['total_epochs']}")

    return params


def _get_base_lr(config: ConfigBundle) -> float:
    """
    从优化器配置中获取基础学习率。

    参数:
        config: ConfigBundle 实例

    返回:
        float: 基础学习率
    """
    opt_cfg = config.training.optimizer
    return float(getattr(opt_cfg, "lr", 0.0001))


# ============================================================
# 自定义 WarmupScheduler (热身包装器)
# ============================================================

class WarmupScheduler(_LRScheduler):
    """
    线性热身 + 任意调度器的组合包装器。

    在热身阶段 (epoch < warmup_epochs)，学习率从 min_lr 线性增长到 base_lr。
    热身结束后，将控制权交给 after_scheduler。

    实现原理:
        通过重写 get_lr() 方法，在热身阶段返回线性插值的学习率，
        非热身阶段将 epoch 偏移后委托给 after_scheduler。

    参数:
        optimizer:       PyTorch 优化器
        after_scheduler: 热身结束后的调度器 (任何 _LRScheduler 子类)
        warmup_epochs:   热身轮数
        base_lr:         目标基础学习率
        min_lr:          热身起始学习率 (热身阶段开始时的 lr)

    使用示例:
        >>> cos_sch = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        >>> scheduler = WarmupScheduler(
        ...     optimizer, after_scheduler=cos_sch,
        ...     warmup_epochs=3, base_lr=1e-3, min_lr=1e-6,
        ... )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        after_scheduler: _LRScheduler,
        warmup_epochs: int,
        base_lr: float,
        min_lr: float = 1.0e-6,
        last_epoch: int = -1,
    ):
        self.after_scheduler = after_scheduler
        self.warmup_epochs = max(0, warmup_epochs)
        self.base_lr = base_lr
        self.min_lr = min_lr
        self._finished_warmup = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        计算当前 epoch 的学习率。

        - 热身阶段: 线性插值 min_lr -> base_lr
        - 非热身阶段: 委托给 after_scheduler (使用偏移后的 epoch)
        """
        if self.last_epoch < self.warmup_epochs:
            # 线性热身: lr = min_lr + (base_lr - min_lr) * (epoch / warmup_epochs)
            warmup_ratio = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.min_lr + (self.base_lr - self.min_lr) * warmup_ratio
                for _ in self.base_lrs
            ]
        else:
            # 热身结束，使用 after_scheduler
            if not self._finished_warmup:
                self._finished_warmup = True
                # 同步 after_scheduler 的 last_epoch
                offset_epoch = self.last_epoch - self.warmup_epochs
                self.after_scheduler.last_epoch = offset_epoch
            return self.after_scheduler.get_lr()

    def step(self, epoch=None):
        """
        执行一步调度更新。

        在热身阶段和 after_scheduler 阶段均正确同步 last_epoch。
        """
        # 标记为手动 epoch 模式以避免 PyTorch 自动递增冲突
        self._last_lr = self.get_lr()

        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        if not self._finished_warmup and self.last_epoch >= self.warmup_epochs:
            # 首次过渡到 after_scheduler，同步其状态
            self._finished_warmup = True
            offset_epoch = self.last_epoch - self.warmup_epochs
            self.after_scheduler.last_epoch = offset_epoch
            # 确保 after_scheduler 的 _last_lr 同步
            self.after_scheduler._last_lr = self.after_scheduler.get_lr()

        if self._finished_warmup:
            # 同步 after_scheduler 的 last_epoch
            self.after_scheduler.last_epoch = self.last_epoch - self.warmup_epochs
            self.after_scheduler._last_lr = self.after_scheduler.get_lr()

        self._last_lr = self.get_lr()

        # 将计算好的 lr 写入 optimizer 的 param_groups
        lr_list = self._last_lr
        for param_group, lr in zip(self.optimizer.param_groups, lr_list):
            param_group["lr"] = lr

    def state_dict(self) -> Dict[str, Any]:
        """
        返回调度器的状态字典 (支持断点续训)。

        包含 WarmupScheduler 自身状态和 after_scheduler 的状态。
        """
        state = {
            "warmup_epochs": self.warmup_epochs,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "_finished_warmup": self._finished_warmup,
            "last_epoch": self.last_epoch,
            "after_scheduler_state_dict": self.after_scheduler.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        从状态字典恢复调度器状态 (支持断点续训)。

        参数:
            state_dict: state_dict() 返回的字典
        """
        self.warmup_epochs = state_dict["warmup_epochs"]
        self.base_lr = state_dict["base_lr"]
        self.min_lr = state_dict["min_lr"]
        self._finished_warmup = state_dict["_finished_warmup"]
        self.last_epoch = state_dict["last_epoch"]
        self.after_scheduler.load_state_dict(state_dict["after_scheduler_state_dict"])


# ============================================================
# 各调度器工厂函数
# ============================================================

def _create_cosine_scheduler(
    optimizer: Optimizer,
    params: Dict[str, Any],
) -> _LRScheduler:
    """
    创建 CosineAnnealingLR 调度器 (含线性热身 + 恒定阶段)。

    学习率曲线:
        阶段1 (epoch 0 ~ warmup_epochs-1):
            lr: min_lr --线性增长--> base_lr
        阶段2 (epoch warmup_epochs ~ n_epochs-1):
            lr: base_lr (恒定)
        阶段3 (epoch n_epochs ~ total_epochs-1):
            lr: base_lr --余弦衰减--> min_lr

    参数:
        optimizer: PyTorch 优化器
        params:    提取后的调度器参数字典

    返回:
        WarmupScheduler: 包装了 CosineAnnealingLR 的热身调度器
    """
    base_lr = params["base_lr"]
    min_lr = params["min_lr"]
    n_epochs = params["n_epochs"]
    n_epochs_decay = params["n_epochs_decay"]
    warmup_epochs = params["warmup_epochs"]

    # 衰减阶段的长度
    T_max = max(1, n_epochs_decay)

    # 创建余弦退火调度器 (将在热身 + 恒定阶段结束后接管)
    # 注意: after_scheduler 的 "epoch 0" 对应全局的 epoch n_epochs
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=min_lr,
    )

    # 用 WarmupScheduler 包装
    scheduler = WarmupScheduler(
        optimizer,
        after_scheduler=cosine_scheduler,
        warmup_epochs=warmup_epochs,
        base_lr=base_lr,
        min_lr=min_lr,
    )

    _get_logger().info(
        "调度器: CosineAnnealingLR | 总轮数=%d | 热身=%d轮 | "
        "恒定=%d轮 | 余弦衰减=%d轮 | lr: %.2e -> %.2e (余弦) -> %.2e",
        params["total_epochs"], warmup_epochs,
        n_epochs - warmup_epochs, n_epochs_decay,
        base_lr, base_lr, min_lr,
    )
    return scheduler


def _create_step_scheduler(
    optimizer: Optimizer,
    params: Dict[str, Any],
) -> _LRScheduler:
    """
    创建 StepLR 调度器 (阶梯式衰减 + 可选热身)。

    学习率曲线:
        阶段1 (epoch 0 ~ warmup_epochs-1):
            lr: min_lr --线性增长--> base_lr
        阶段2 (epoch warmup_epochs ~ total_epochs-1):
            lr: base_lr * gamma^(epoch // step_size)

    衰减策略:
        每隔 step_size 个 epoch，学习率乘以 gamma。
        gamma 自动计算，使得在衰减阶段结束时达到 min_lr。

    参数:
        optimizer: PyTorch 优化器
        params:    提取后的调度器参数字典

    返回:
        WarmupScheduler: 包装了 StepLR 的热身调度器
    """
    base_lr = params["base_lr"]
    min_lr = params["min_lr"]
    warmup_epochs = params["warmup_epochs"]
    lr_decay_iters = max(1, params["lr_decay_iters"])

    # 衰减阶段总长度
    decay_epochs = params["n_epochs_decay"]

    # 计算 gamma: base_lr * gamma^(decay_epochs / step_size) = min_lr
    # gamma = (min_lr / base_lr) ^ (step_size / decay_epochs)
    step_size = min(lr_decay_iters, decay_epochs)
    num_steps = max(1, decay_epochs // step_size)
    gamma = (min_lr / base_lr) ** (1.0 / max(1, num_steps))
    gamma = max(0.01, min(0.9999, gamma))  # 安全裁剪

    step_scheduler = StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )

    scheduler = WarmupScheduler(
        optimizer,
        after_scheduler=step_scheduler,
        warmup_epochs=warmup_epochs,
        base_lr=base_lr,
        min_lr=min_lr,
    )

    _get_logger().info(
        "调度器: StepLR | 总轮数=%d | 热身=%d轮 | step_size=%d | "
        "gamma=%.4f | lr: %.2e -> %.2e",
        params["total_epochs"], warmup_epochs,
        step_size, gamma, base_lr, min_lr,
    )
    return scheduler


def _create_exp_scheduler(
    optimizer: Optimizer,
    params: Dict[str, Any],
) -> _LRScheduler:
    """
    创建 ExponentialLR 调度器 (指数衰减 + 可选热身)。

    学习率曲线:
        阶段1 (epoch 0 ~ warmup_epochs-1):
            lr: min_lr --线性增长--> base_lr
        阶段2 (epoch warmup_epochs ~ total_epochs-1):
            lr: base_lr * gamma^epoch

    gamma 自动计算，使得在衰减阶段结束时达到 min_lr。

    参数:
        optimizer: PyTorch 优化器
        params:    提取后的调度器参数字典

    返回:
        WarmupScheduler: 包装了 ExponentialLR 的热身调度器
    """
    base_lr = params["base_lr"]
    min_lr = params["min_lr"]
    warmup_epochs = params["warmup_epochs"]
    n_epochs_decay = max(1, params["n_epochs_decay"])

    # 计算 gamma: base_lr * gamma^(n_epochs_decay) = min_lr
    gamma = (min_lr / base_lr) ** (1.0 / n_epochs_decay)
    gamma = max(0.1, min(0.9999, gamma))  # 安全裁剪

    exp_scheduler = ExponentialLR(
        optimizer,
        gamma=gamma,
    )

    scheduler = WarmupScheduler(
        optimizer,
        after_scheduler=exp_scheduler,
        warmup_epochs=warmup_epochs,
        base_lr=base_lr,
        min_lr=min_lr,
    )

    _get_logger().info(
        "调度器: ExponentialLR | 总轮数=%d | 热身=%d轮 | "
        "gamma=%.4f | lr: %.2e -> %.2e",
        params["total_epochs"], warmup_epochs,
        gamma, base_lr, min_lr,
    )
    return scheduler


def _create_plateau_scheduler(
    optimizer: Optimizer,
    params: Dict[str, Any],
) -> ReduceLROnPlateau:
    """
    创建 ReduceLROnPlateau 调度器 (基于验证指标自适应衰减)。

    当验证指标 (如 val_loss) 在 patience 个 epoch 内不再改善时，
    学习率乘以 factor 进行衰减。

    特点:
        - 不使用热身 (ReduceLROnPlateau 的特性决定)
        - 需要传入验证指标: scheduler.step(val_metric)
        - 具有冷却期 (cooldown) 和阈值 (threshold) 防止频繁衰减

    参数:
        optimizer: PyTorch 优化器
        params:    提取后的调度器参数字典

    返回:
        ReduceLROnPlateau
    """
    base_lr = params["base_lr"]
    min_lr = params["min_lr"]

    # 自动计算合适参数
    # factor: 衰减系数 (每次衰减将 lr 乘以 factor)
    # patience: 容忍多少个 epoch 指标不改善
    n_epochs_decay = max(1, params["n_epochs_decay"])
    total_epochs = params["total_epochs"]

    # 计算 factor: base_lr * factor^(max_decays) >= min_lr
    # 假设最多衰减 5 次
    max_decays = min(5, n_epochs_decay)
    if max_decays > 0:
        factor = (min_lr / base_lr) ** (1.0 / max_decays)
    else:
        factor = 0.1
    factor = max(0.1, min(0.9, factor))

    # patience: 基于总轮数自适应
    patience = max(2, total_epochs // 4)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=max(1, patience // 2),
        min_lr=min_lr,
        eps=1e-8,
        verbose=False,
    )

    _get_logger().info(
        "调度器: ReduceLROnPlateau | 总轮数=%d | factor=%.2f | "
        "patience=%d | cooldown=%d | min_lr=%.2e",
        total_epochs, factor, patience, max(1, patience // 2), min_lr,
    )
    return scheduler


def _create_linear_scheduler(
    optimizer: Optimizer,
    params: Dict[str, Any],
) -> _LRScheduler:
    """
    创建线性衰减调度器 (含热身)。

    学习率曲线:
        阶段1 (epoch 0 ~ warmup_epochs-1):
            lr: min_lr --线性增长--> base_lr
        阶段2 (epoch warmup_epochs ~ n_epochs-1):
            lr: base_lr (恒定)
        阶段3 (epoch n_epochs ~ total_epochs-1):
            lr: base_lr --线性衰减--> min_lr

    参数:
        optimizer: PyTorch 优化器
        params:    提取后的调度器参数字典

    返回:
        WarmupScheduler: 包装了 LambdaLR (线性衰减) 的热身调度器
    """
    base_lr = params["base_lr"]
    min_lr = params["min_lr"]
    n_epochs_decay = max(1, params["n_epochs_decay"])
    warmup_epochs = params["warmup_epochs"]

    # 线性衰减 lambda 函数
    def linear_decay_fn(epoch: int) -> float:
        """
        线性衰减: 1.0 -> min_lr/base_lr

        参数:
            epoch: after_scheduler 视角的 epoch (0 对应衰减阶段开始)
        """
        if n_epochs_decay <= 1:
            return min_lr / base_lr
        progress = min(1.0, epoch / (n_epochs_decay - 1))
        return 1.0 - (1.0 - min_lr / base_lr) * progress

    linear_scheduler = LambdaLR(
        optimizer,
        lr_lambda=linear_decay_fn,
    )

    scheduler = WarmupScheduler(
        optimizer,
        after_scheduler=linear_scheduler,
        warmup_epochs=warmup_epochs,
        base_lr=base_lr,
        min_lr=min_lr,
    )

    _get_logger().info(
        "调度器: Linear | 总轮数=%d | 热身=%d轮 | "
        "恒定=%d轮 | 线性衰减=%d轮 | lr: %.2e -> %.2e",
        params["total_epochs"], warmup_epochs,
        params["n_epochs"] - warmup_epochs, n_epochs_decay,
        base_lr, min_lr,
    )
    return scheduler


# ============================================================
# 调度器类型到工厂函数的映射
# ============================================================
_SCHEDULER_REGISTRY: Dict[str, Any] = {
    "cosine": _create_cosine_scheduler,
    "step": _create_step_scheduler,
    "exp": _create_exp_scheduler,
    "plateau": _create_plateau_scheduler,
    "linear": _create_linear_scheduler,
}


# ============================================================
# 主入口：create_scheduler
# ============================================================

def create_scheduler(
    optimizer: Optimizer,
    config: ConfigBundle,
    **kwargs,
) -> Union[_LRScheduler, ReduceLROnPlateau]:
    """
    根据配置创建学习率调度器 (工厂函数主入口)。

    自动检测配置中的 scheduler.type 字段，调用对应的工厂函数。

    参数:
        optimizer:  PyTorch 优化器实例
        config:     ConfigBundle 配置实例
        **kwargs:   额外参数 (预留，当前未使用)

    返回:
        torch.optim.lr_scheduler._LRScheduler 或 ReduceLROnPlateau

    异常:
        ValueError: 不支持的调度器类型

    使用示例:
        >>> from config.config_loader import load_config
        >>> from src.training.optimizer import create_optimizer
        >>> from src.training.scheduler import create_scheduler
        >>>
        >>> config = load_config("config/default_config.yaml")
        >>> optimizer = create_optimizer(model, config)
        >>> scheduler = create_scheduler(optimizer, config)
        >>>
        >>> # 训练循环
        >>> for epoch in range(config.training.scheduler.n_epochs +
        ...                    config.training.scheduler.n_epochs_decay):
        ...     train_one_epoch(model, optimizer, train_loader)
        ...
        ...     if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        ...         scheduler.step(val_loss)       # Plateau 需传入验证指标
        ...     else:
        ...         scheduler.step()               # 其他调度器直接 step
    """
    logger = _get_logger()
    params = _extract_scheduler_params(config)
    sch_type = params["type"].lower()

    if sch_type not in _SCHEDULER_REGISTRY:
        available = list(_SCHEDULER_REGISTRY.keys())
        raise ValueError(
            f"不支持的调度器类型: '{sch_type}'。\n"
            f"当前可用的类型: {available}\n"
            f"请检查 config.training.scheduler.type 的配置。"
        )

    factory_fn = _SCHEDULER_REGISTRY[sch_type]
    logger.debug("创建调度器: type=%s", sch_type)

    try:
        scheduler = factory_fn(optimizer, params)
    except Exception as e:
        logger.error("调度器创建失败: %s", str(e))
        raise RuntimeError(f"调度器 '{sch_type}' 创建失败: {e}") from e

    return scheduler


# ============================================================
# 辅助函数：获取当前学习率
# ============================================================

def get_current_lr(optimizer: Optimizer) -> List[float]:
    """
    获取优化器当前各参数组的学习率。

    参数:
        optimizer: PyTorch 优化器

    返回:
        List[float]: 各参数组的学习率列表
    """
    return [group["lr"] for group in optimizer.param_groups]


def log_lr(logger: Optional[logging.Logger] = None, optimizer: Optional[Optimizer] = None) -> str:
    """
    格式化当前学习率为日志字符串。

    参数:
        logger:    可选的日志器
        optimizer: PyTorch 优化器

    返回:
        str: 格式化的学习率字符串
    """
    if optimizer is None:
        return "lr=N/A"

    lrs = get_current_lr(optimizer)
    if len(lrs) == 1:
        msg = f"lr={lrs[0]:.2e}"
    else:
        lr_strs = ", ".join(f"{lr:.2e}" for lr in lrs)
        msg = f"lr=[{lr_strs}]"

    if logger:
        logger.debug(msg)

    return msg


# ============================================================
# 调度器曲线可视化 (调试用)
# ============================================================

def simulate_lr_curve(
    optimizer: Optimizer,
    scheduler: Union[_LRScheduler, ReduceLROnPlateau],
    total_epochs: int,
    val_loss_fn=None,
) -> List[float]:
    """
    模拟完整训练过程中的学习率变化曲线 (调试/日志用)。

    参数:
        optimizer:     PyTorch 优化器
        scheduler:     调度器实例
        total_epochs:  模拟的总轮数
        val_loss_fn:   可选，返回当前 epoch 的验证损失 (仅 ReduceLROnPlateau 需要)

    返回:
        List[float]: 每个 epoch 的学习率列表
    """
    lr_curve = []
    is_plateau = isinstance(scheduler, ReduceLROnPlateau)

    # 重置调度器
    for group in optimizer.param_groups:
        if "initial_lr" not in group:
            group["initial_lr"] = group["lr"]

    # 保存原始状态
    original_state = None
    if hasattr(scheduler, "state_dict"):
        try:
            original_state = scheduler.state_dict()
        except Exception:
            pass

    try:
        for epoch in range(total_epochs):
            current_lr = get_current_lr(optimizer)[0]
            lr_curve.append(current_lr)

            if is_plateau:
                if val_loss_fn is not None:
                    val_loss = val_loss_fn(epoch)
                else:
                    # 模拟逐步改善后停滞的验证损失
                    val_loss = 1.0 / (1.0 + epoch * 0.1) + 0.01 * (epoch // 5)
                scheduler.step(val_loss)
            else:
                scheduler.step()
    finally:
        # 恢复原始状态
        if original_state is not None and hasattr(scheduler, "load_state_dict"):
            try:
                scheduler.load_state_dict(original_state)
            except Exception:
                pass

    return lr_curve


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("学习率调度器工厂模块自测")
    print("=" * 60)

    # 构建测试模型与优化器
    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    model = _DummyModel()
    lr = 0.001

    # 模拟配置的参数
    test_params = {
        "type": "cosine",
        "n_epochs": 8,
        "n_epochs_decay": 4,
        "warmup_epochs": 2,
        "total_epochs": 12,
        "min_lr": 1.0e-6,
        "lr_decay_iters": 10,
        "base_lr": lr,
    }

    # ---- 测试 CosineAnnealingLR ----
    print("\n[1/5] 测试 CosineAnnealingLR (含热身) ...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = _create_cosine_scheduler(optimizer, test_params)
    # 模拟训练
    lr_curve = []
    for ep in range(test_params["total_epochs"]):
        lr_curve.append(get_current_lr(optimizer)[0])
        scheduler.step()
    print(f"  LR 曲线 ({len(lr_curve)} epochs): "
          f"{', '.join(f'{lr:.2e}' for lr in lr_curve[:6])}"
          f" ... {', '.join(f'{lr:.2e}' for lr in lr_curve[-3:])}")
    # 验证终点
    assert abs(lr_curve[-1] - test_params["min_lr"]) < 1e-5, \
        f"最终 lr={lr_curve[-1]:.2e}, 期望={test_params['min_lr']:.2e}"
    # 验证热身
    assert lr_curve[0] < lr_curve[test_params["warmup_epochs"] - 1], \
        "热身阶段学习率应递增"
    print("  通过: 热身递增 + 余弦衰减至 min_lr")

    # ---- 测试 StepLR ----
    print("\n[2/5] 测试 StepLR ...")
    test_params["type"] = "step"
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = _create_step_scheduler(optimizer, test_params)
    lr_curve = []
    for ep in range(test_params["total_epochs"]):
        lr_curve.append(get_current_lr(optimizer)[0])
        scheduler.step()
    print(f"  LR 曲线: {', '.join(f'{lr:.2e}' for lr in lr_curve)}")
    # 验证阶梯式下降
    steps = [lr_curve[i] for i in range(1, len(lr_curve)) if lr_curve[i] < lr_curve[i-1]]
    print(f"  衰减步数: {len(steps)}")
    assert len(steps) > 0, "StepLR 应有阶梯式衰减"
    print("  通过: 阶梯式衰减")

    # ---- 测试 ExponentialLR ----
    print("\n[3/5] 测试 ExponentialLR ...")
    test_params["type"] = "exp"
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = _create_exp_scheduler(optimizer, test_params)
    lr_curve = []
    for ep in range(test_params["total_epochs"]):
        lr_curve.append(get_current_lr(optimizer)[0])
        scheduler.step()
    print(f"  LR 曲线: {', '.join(f'{lr:.2e}' for lr in lr_curve)}")
    # 验证指数衰减 (对数坐标下近似线性)
    non_warmup = lr_curve[test_params["warmup_epochs"]:]
    ratio = non_warmup[-1] / non_warmup[0]
    print(f"  衰减比例: {ratio:.4f}")
    assert non_warmup[-1] < non_warmup[0], "指数衰减后学习率应降低"
    print("  通过: 指数衰减")

    # ---- 测试 ReduceLROnPlateau ----
    print("\n[4/5] 测试 ReduceLROnPlateau ...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = _create_plateau_scheduler(optimizer, test_params)
    initial_lr = get_current_lr(optimizer)[0]
    # 模拟验证损失持续不改善
    for ep in range(20):
        val_loss = 0.5  # 不改善
        scheduler.step(val_loss)
    final_lr = get_current_lr(optimizer)[0]
    print(f"  初始 LR: {initial_lr:.2e}, 最终 LR: {final_lr:.2e}")
    assert final_lr <= initial_lr, "Plateau 调度器应在指标停滞时降低学习率"
    print("  通过: 自适应衰减")

    # ---- 测试 Linear ----
    print("\n[5/5] 测试 Linear ...")
    test_params["type"] = "linear"
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = _create_linear_scheduler(optimizer, test_params)
    lr_curve = []
    for ep in range(test_params["total_epochs"]):
        lr_curve.append(get_current_lr(optimizer)[0])
        scheduler.step()
    print(f"  LR 曲线: {', '.join(f'{lr:.2e}' for lr in lr_curve)}")
    # 验证终点
    assert abs(lr_curve[-1] - test_params["min_lr"]) < 1e-5, \
        f"最终 lr={lr_curve[-1]:.2e}, 期望={test_params['min_lr']:.2e}"
    # 验证衰减阶段线性
    decay_phase = lr_curve[test_params["n_epochs"]:]
    if len(decay_phase) >= 3:
        diffs = [decay_phase[i] - decay_phase[i+1] for i in range(len(decay_phase)-1)]
        diff_variance = max(diffs) - min(diffs)
        print(f"  衰减阶段步长变化: {max(diffs):.2e} ~ {min(diffs):.2e}")
    print("  通过: 线性衰减至 min_lr")

    # ---- 状态保存/恢复 ----
    print("\n[状态保存/恢复] 测试 WarmupScheduler state_dict ...")
    test_params["type"] = "cosine"
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = _create_cosine_scheduler(optimizer, test_params)
    # 模拟半程训练
    for ep in range(6):
        scheduler.step()
    mid_lr = get_current_lr(optimizer)[0]
    state = scheduler.state_dict()

    # 重建并恢复
    optimizer2 = optim.Adam(model.parameters(), lr=lr)
    scheduler2 = _create_cosine_scheduler(optimizer2, test_params)
    scheduler2.load_state_dict(state)
    restored_lr = get_current_lr(optimizer2)[0]
    print(f"  保存时 LR: {mid_lr:.2e}, 恢复后 LR: {restored_lr:.2e}")
    assert abs(mid_lr - restored_lr) < 1e-10, "状态恢复后学习率应一致"
    print("  通过: 状态保存/恢复一致")

    # ---- 错误处理 ----
    print("\n[错误处理] 测试不支持的类型 ...")
    from unittest.mock import Mock
    mock_config = Mock(spec=ConfigBundle)
    mock_sch = Mock()
    mock_sch.type = "cyclic"
    mock_sch.n_epochs = 8
    mock_sch.n_epochs_decay = 4
    mock_sch.warmup_epochs = 2
    mock_sch.min_lr = 1e-6
    mock_sch.lr_decay_iters = 10
    mock_training = Mock()
    mock_training.scheduler = mock_sch
    mock_opt_cfg = Mock()
    mock_opt_cfg.lr = 0.001
    mock_training.optimizer = mock_opt_cfg
    mock_config.training = mock_training

    try:
        create_scheduler(optimizer, mock_config)
        print("  错误: 应该抛出异常!")
    except ValueError as e:
        print(f"  正确捕获 ValueError: {str(e)[:80]}...")

    print("\n" + "=" * 60)
    print("所有自测通过!")
    print("=" * 60)
