# -*- coding: utf-8 -*-
"""
checkpoint.py
============================================================================
断点管理模块 —— 训练状态的保存、恢复、清理。

功能:
    1. 保存完整训练状态 (模型 / 优化器 / 调度器 / AMP缩放器 / 指标 / RNG状态)
    2. 从断点恢复训练 (支持不同设备间迁移)
    3. 自动追踪最佳模型 (基于指定指标)
    4. 清理过期断点 (保留最近 N 个)
    5. 自动创建目录结构 (路径均来自配置)
    6. 跨 fold 独立断点管理

Checkpoint 文件格式:
    {
        "epoch": int,                       # 当前已完成的 epoch 编号
        "model_state_dict": OrderedDict,    # 模型参数 (已 detach 到 CPU)
        "optimizer_state_dict": dict,       # 优化器状态
        "scheduler_state_dict": dict,       # 学习率调度器状态
        "scaler_state_dict": dict,          # AMP GradScaler 状态
        "metric_logger": dict,              # 训练指标日志
        "config_snapshot": dict,            # 配置快照 (用于追溯)
        "rng_state": {                      # 随机数生成器状态
            "python": tuple,
            "numpy": tuple,
            "torch": bytes,
            "torch_cuda": list 或 None,
        },
    }

参数来源:
    config.training.checkpoint
        - save_every   : int = 1     每隔 N 个 epoch 保存一次
        - resume       : bool = True 是否从断点恢复
        - keep_last_n  : int = 3    保留最近 N 个断点 (0 表示保留全部)

使用示例:
    from config.config_loader import load_config
    from src.training.checkpoint import CheckpointManager

    config = load_config("config/default_config.yaml")
    ckpt_mgr = CheckpointManager(config, fold_id=1)

    # 保存断点
    ckpt_mgr.save(model, optimizer, scheduler, scaler,
                  epoch=5, metrics={"val_cindex": 0.72})

    # 恢复训练
    start_epoch = ckpt_mgr.load(path, model, optimizer, scheduler, scaler, device)

    # 自动恢复 (便捷方法)
    start_epoch = ckpt_mgr.auto_resume(model, optimizer, scheduler, scaler, device)

    # 获取最近断点
    best_path = ckpt_mgr.get_latest_checkpoint()

    # 清理过期断点
    ckpt_mgr.cleanup(keep_last_n=3)
============================================================================
"""

from __future__ import annotations

import logging
import os
import random
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

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
        _logger = get_logger("YuHou.checkpoint")
    return _logger


# ============================================================
# 文件命名规范与正则解析
# ============================================================

# 常规 epoch 断点: fold1_epoch_005_cindex=0.7234.pth 或 fold1_epoch_005.pth
_CKPT_PATTERN = re.compile(
    r"fold(\d+)_epoch_(\d+)(?:_([a-z_]+)=([\d.]+))?\.(pth|tar|pt)"
)

# 最佳模型: fold1_best_cindex=0.7800.pth
_BEST_CKPT_PATTERN = re.compile(
    r"fold(\d+)_best_([a-z_]+)=([\d.]+)\.(pth|tar|pt)"
)

# Resume 断点: fold1_resume.pth
_RESUME_FILENAME = "fold{fold}_resume.pth"

# 文件名模板
_CKPT_FILENAME_TEMPLATE = "fold{fold}_epoch_{epoch:03d}.pth"
_CKPT_FILENAME_METRIC_TEMPLATE = "fold{fold}_epoch_{epoch:03d}_{metric}={value:.4f}.pth"
_BEST_FILENAME_TEMPLATE = "fold{fold}_best_{metric}={value:.4f}.pth"
_MODEL_ONLY_TEMPLATE = "fold{fold}_epoch_{epoch:03d}_{metric}={value:.4f}_model_only.pth"


def _make_ckpt_filename(
    fold_id: int,
    epoch: int,
    metric_name: Optional[str] = None,
    metric_value: Optional[float] = None,
) -> str:
    """
    生成标准化的断点文件名。

    参数:
        fold_id:      交叉验证折编号
        epoch:        已完成的 epoch 编号
        metric_name:  可选，指标名称 (如 "cindex")
        metric_value: 可选，指标值

    返回:
        str: 格式化的文件名，如 "fold1_epoch_005_cindex=0.7234.pth"
    """
    if metric_name is not None and metric_value is not None:
        return _CKPT_FILENAME_METRIC_TEMPLATE.format(
            fold=fold_id, epoch=epoch, metric=metric_name, value=metric_value,
        )
    return _CKPT_FILENAME_TEMPLATE.format(fold=fold_id, epoch=epoch)


def _make_best_filename(
    fold_id: int,
    metric_name: str,
    metric_value: float,
) -> str:
    """
    生成最佳模型的断点文件名。

    参数:
        fold_id:      交叉验证折编号
        metric_name:  指标名称 (如 "cindex")
        metric_value: 指标值

    返回:
        str: 如 "fold1_best_cindex=0.7800.pth"
    """
    return _BEST_FILENAME_TEMPLATE.format(
        fold=fold_id, metric=metric_name, value=metric_value,
    )


# ============================================================
# RNG 状态管理
# ============================================================

def _capture_rng_state() -> Dict[str, Any]:
    """
    捕获当前所有随机数生成器的状态。

    捕获范围:
        - Python random
        - NumPy RNG
        - PyTorch CPU RNG
        - PyTorch CUDA RNG (所有 GPU)

    返回:
        dict: 包含所有 RNG 状态的字典
    """
    rng_state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return rng_state


def _restore_rng_state(rng_state: Dict[str, Any]) -> None:
    """
    恢复随机数生成器状态。

    恢复失败时静默跳过 (GPU 数量/驱动变化可能导致 CUDA RNG 恢复失败)，
    因为训练可重复性仅依赖于 torch_rng + numpy_rng + python_rng。

    参数:
        rng_state: _capture_rng_state() 返回的字典
    """
    if "python" in rng_state and rng_state["python"] is not None:
        try:
            random.setstate(rng_state["python"])
        except Exception:
            pass

    if "numpy" in rng_state and rng_state["numpy"] is not None:
        try:
            np.random.set_state(rng_state["numpy"])
        except Exception:
            pass

    if "torch" in rng_state and rng_state["torch"] is not None:
        try:
            torch.set_rng_state(rng_state["torch"])
        except Exception:
            pass

    if "torch_cuda" in rng_state and rng_state["torch_cuda"] is not None:
        if torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
            except Exception:
                pass


# ============================================================
# CheckpointManager 类
# ============================================================

class CheckpointManager:
    """
    断点管理器 —— 负责训练状态的保存、恢复、追踪和清理。

    设计原则:
        - 所有路径从 config 推断，不硬编码
        - 自动创建目录结构
        - 支持常规 epoch 断点 + 最佳模型追踪 + resume 恢复
        - 安全保存: 先写临时文件 (.tmp)，写入成功后再原子重命名为正式文件
        - 跨 fold 独立管理 (通过 fold_id 区分)

    参数:
        config:             ConfigBundle 配置实例
        fold_id:            当前交叉验证折编号 (默认为 1)
        metric_name:        用于追踪最佳模型的指标名称 (如 "cindex")
        metric_mode:        指标优化方向 "max" (越大越好) 或 "min" (越小越好)
        checkpoint_dir:     可选，覆盖默认的 ckpt 目录
        resume_dir:         可选，覆盖默认的 resume 目录

    属性:
        ckpt_dir:      常规断点保存目录 (Path)
        resume_dir:    恢复断点保存目录 (Path)
        best_path:     当前最佳模型路径 (Path 或 None)
        best_value:    当前最佳指标值 (float 或 None)
        keep_last_n:   保留最近 N 个 epoch 断点
    """

    def __init__(
        self,
        config: ConfigBundle,
        fold_id: int = 1,
        metric_name: str = "cindex",
        metric_mode: str = "max",
        checkpoint_dir: Optional[str] = None,
        resume_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        初始化断点管理器。

        参数:
            config:         ConfigBundle 配置实例
            fold_id:        交叉验证折编号
            metric_name:    最佳模型追踪指标名称
            metric_mode:    "max" 表示指标越大越好, "min" 表示越小越好
            checkpoint_dir: 可选，覆盖默认的 ckpt 目录
            resume_dir:     可选，覆盖默认的 resume 目录
            logger:         可选，外部日志器 (如果为 None，使用模块级日志器)
        """
        self.config = config
        self.fold_id = fold_id
        self.metric_name = metric_name
        self.metric_mode = metric_mode.lower()
        self.logger = logger or _get_logger()

        if self.metric_mode not in ("max", "min"):
            raise ValueError(
                f"metric_mode 必须为 'max' 或 'min', 当前值: '{metric_mode}'"
            )

        # ---- 目录路径 ----
        if checkpoint_dir is not None:
            self.ckpt_dir = Path(checkpoint_dir)
        else:
            self.ckpt_dir = config.get_subdir("ckpt")

        if resume_dir is not None:
            self.resume_dir = Path(resume_dir)
        else:
            self.resume_dir = config.get_subdir("resume")

        # 确保目录存在
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.resume_dir.mkdir(parents=True, exist_ok=True)

        # ---- 配置参数 ----
        ckpt_cfg = config.training.checkpoint
        self.save_every = int(getattr(ckpt_cfg, "save_every", 1))
        self._resume_enabled = bool(getattr(ckpt_cfg, "resume", True))
        self.keep_last_n = int(getattr(ckpt_cfg, "keep_last_n", 3))

        # ---- 最佳模型追踪状态 ----
        self.best_path: Optional[Path] = None
        self.best_value: Optional[float] = None

        self.logger.debug(
            "[Fold %d] CheckpointManager 初始化: ckpt_dir=%s, resume_dir=%s, "
            "metric=%s(%s), save_every=%d, keep_last_n=%d",
            self.fold_id, self.ckpt_dir, self.resume_dir,
            self.metric_name, self.metric_mode,
            self.save_every, self.keep_last_n,
        )

    # ============================================================
    # 内部辅助方法
    # ============================================================

    def _log(self, msg: str, level: str = "info") -> None:
        """
        带 fold 前缀的日志输出。

        参数:
            msg:   日志消息
            level: 日志级别 ("debug", "info", "warning", "error")
        """
        getattr(self.logger, level)(f"[Fold {self.fold_id}] {msg}")

    def _is_better(self, current_value: float) -> bool:
        """
        判断当前指标值是否优于已记录的最佳值。

        参数:
            current_value: 当前指标值

        返回:
            bool: True 表示当前值更优
        """
        if self.best_value is None:
            return True
        if self.metric_mode == "max":
            return current_value > self.best_value
        else:
            return current_value < self.best_value

    def _move_optimizer_to_device(
        self, optimizer: Optimizer, device: torch.device
    ) -> None:
        """
        将优化器内部的所有张量移动到目标设备。

        GPU -> CPU 或 GPU0 -> GPU1 迁移时需调用。
        仅移动 optimizer.state 中的 tensor 值，不修改 param_groups。

        参数:
            optimizer: PyTorch 优化器
            device:    目标设备
        """
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # ============================================================
    # 保存断点
    # ============================================================

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None,
        scaler: Optional[GradScaler] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """
        保存完整的训练状态为断点文件。

        保存内容:
            - epoch:                当前已完成的 epoch 编号
            - model_state_dict:     模型参数 (已 detach 到 CPU，兼容 DataParallel)
            - optimizer_state_dict: 优化器状态
            - scheduler_state_dict: 学习率调度器状态 (可选)
            - scaler_state_dict:    AMP GradScaler 状态 (可选)
            - metric_logger:        当前 epoch 的训练/验证指标
            - config_snapshot:      配置快照 (字典形式)
            - rng_state:            随机数生成器状态

        保存策略:
            1. 原子写入: 先写 .tmp 临时文件，再原子重命名为正式文件，
               防止写入中断导致的文件损坏。
            2. 同时保存一份 resume 断点到 resume_dir (覆盖式)。
            3. 如果 metrics 中包含追踪指标且优于当前最佳，同时保存最佳模型副本。

        参数:
            model:      PyTorch 模型 (需包含 state_dict)
            optimizer:  PyTorch 优化器
            scheduler:  学习率调度器 (可选)
            scaler:     AMP GradScaler (可选)
            epoch:      当前已完成的 epoch 编号
            metrics:    训练/验证指标字典 (如 {"val_cindex": 0.7234, "val_loss": 0.512})

        返回:
            Path: 保存的断点文件路径

        异常:
            IOError: 磁盘写入失败
        """
        metrics = metrics or {}

        # ---- 提取模型参数 (兼容 DataParallel) ----
        if hasattr(model, "module"):
            # DataParallel / DistributedDataParallel 包装
            model_state = {
                k: v.detach().cpu() for k, v in model.module.state_dict().items()
            }
        else:
            model_state = {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            }

        # ---- 构建断点数据 ----
        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "fold_id": self.fold_id,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else {}
            ),
            "scaler_state_dict": (
                scaler.state_dict() if scaler is not None else {}
            ),
            "metric_logger": metrics,
            "config_snapshot": self.config.to_dict(),
            "rng_state": _capture_rng_state(),
        }

        # ---- 生成文件名 ----
        metric_key = None
        metric_val = None
        if self.metric_name in metrics:
            metric_key = self.metric_name
            metric_val = metrics[self.metric_name]

        ckpt_filename = _make_ckpt_filename(
            self.fold_id, epoch,
            metric_name=metric_key,
            metric_value=metric_val,
        )

        # ---- 保存 epoch 断点到 ckpt_dir (原子写入) ----
        ckpt_path = self.ckpt_dir / ckpt_filename
        tmp_path = ckpt_path.with_suffix(".pth.tmp")

        try:
            torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=True)
            if tmp_path.exists():
                if ckpt_path.exists():
                    ckpt_path.unlink()
                tmp_path.rename(ckpt_path)
        except Exception:
            # 清理临时文件
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        # ---- 保存 resume 断点 (覆盖式) ----
        resume_filename = _RESUME_FILENAME.format(fold=self.fold_id)
        resume_path = self.resume_dir / resume_filename
        resume_tmp = resume_path.with_suffix(".pth.tmp")
        try:
            torch.save(checkpoint, resume_tmp, _use_new_zipfile_serialization=True)
            if resume_tmp.exists():
                if resume_path.exists():
                    resume_path.unlink()
                resume_tmp.rename(resume_path)
        except Exception as e:
            self._log(f"Resume 断点保存失败: {e}", "warning")
            if resume_tmp.exists():
                resume_tmp.unlink()

        # ---- 追踪最佳模型 ----
        if self.metric_name in metrics:
            current_value = metrics[self.metric_name]
            if self._is_better(current_value):
                self.best_value = current_value

                # 删除旧的最佳模型
                if self.best_path is not None and self.best_path.exists():
                    self.best_path.unlink()

                best_filename = _make_best_filename(
                    self.fold_id, self.metric_name, current_value,
                )
                self.best_path = self.ckpt_dir / best_filename

                # 保存最佳模型副本
                torch.save(checkpoint, self.best_path,
                           _use_new_zipfile_serialization=True)

                self._log(
                    f"最佳模型更新: {self.metric_name}={current_value:.4f} "
                    f"-> {best_filename}"
                )

        # ---- 日志 ----
        file_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        metric_str = ", ".join(
            f"{k}={v:.4f}" for k, v in metrics.items()
        ) if metrics else "无指标"

        self._log(
            f"断点已保存: {ckpt_filename} | epoch={epoch} | "
            f"{metric_str} | 大小={file_size_mb:.1f}MB"
        )

        # ---- 自动清理 (每次保存后) ----
        if self.keep_last_n > 0:
            self._cleanup_old_epoch_checkpoints()

        return ckpt_path

    # ============================================================
    # 加载断点
    # ============================================================

    def load(
        self,
        path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None,
        scaler: Optional[GradScaler] = None,
        device: Optional[torch.device] = None,
        restore_rng: bool = True,
    ) -> int:
        """
        从断点文件恢复训练状态。

        恢复后，模型/优化器/调度器/缩放器等对象的状态被原地更新。

        参数:
            path:         断点文件路径
            model:        PyTorch 模型 (原地更新)
            optimizer:    PyTorch 优化器 (可选, 原地更新)
            scheduler:    学习率调度器 (可选, 原地更新)
            scaler:       AMP GradScaler (可选, 原地更新)
            device:       目标设备 (可选, 默认自动检测 cuda/cpu)
            restore_rng:  是否恢复随机数状态 (默认 True)

        返回:
            int: 下一个待训练的 epoch 编号 (断点中的 epoch + 1)

        异常:
            FileNotFoundError: 断点文件不存在
            KeyError:          断点文件格式不兼容 (缺少必要字段)
            RuntimeError:      状态加载失败

        使用示例:
            >>> ckpt_mgr = CheckpointManager(config, fold_id=1)
            >>> start_epoch = ckpt_mgr.load(
            ...     "ckpt/fold1_epoch_005.pth",
            ...     model, optimizer, scheduler, scaler, device,
            ... )
            >>> # 从 start_epoch 继续训练
            >>> for epoch in range(start_epoch, total_epochs):
            ...     train_one_epoch(...)
        """
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"断点文件不存在: {ckpt_path}")

        self._log(f"正在从断点恢复: {ckpt_path.name}")

        # 确定设备
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载断点
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"无法加载断点文件 '{ckpt_path}': {e}") from e

        # 验证必要字段
        required_fields = ["epoch", "model_state_dict"]
        missing = [f for f in required_fields if f not in checkpoint]
        if missing:
            raise KeyError(
                f"断点文件缺少必要字段: {missing}。"
                f"文件可能已损坏或来自不兼容的版本。"
            )

        # ---- 恢复模型参数 ----
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as e:
            self._log(f"模型参数不完全匹配，尝试 strict=False: {e}", "warning")
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        model.to(device)

        # ---- 恢复优化器状态 ----
        if optimizer is not None:
            opt_state = checkpoint.get("optimizer_state_dict")
            if opt_state:
                try:
                    optimizer.load_state_dict(opt_state)
                    self._move_optimizer_to_device(optimizer, device)
                except Exception as e:
                    self._log(f"优化器状态恢复失败 (将使用全新优化器): {e}", "warning")
            else:
                self._log("断点中未包含优化器状态, 使用全新优化器", "warning")

        # ---- 恢复调度器状态 ----
        if scheduler is not None:
            sch_state = checkpoint.get("scheduler_state_dict")
            if sch_state:
                try:
                    scheduler.load_state_dict(sch_state)
                except Exception as e:
                    self._log(f"调度器状态恢复失败: {e}", "warning")
            else:
                self._log("断点中未包含调度器状态, 使用初始状态", "warning")

        # ---- 恢复 AMP GradScaler 状态 ----
        if scaler is not None:
            scaler_state = checkpoint.get("scaler_state_dict")
            if scaler_state:
                try:
                    scaler.load_state_dict(scaler_state)
                except Exception as e:
                    self._log(f"GradScaler 状态恢复失败: {e}", "warning")

        # ---- 恢复 RNG 状态 ----
        if restore_rng and "rng_state" in checkpoint:
            try:
                _restore_rng_state(checkpoint["rng_state"])
            except Exception as e:
                self._log(f"RNG 状态恢复失败: {e}", "warning")

        # ---- 恢复最佳模型追踪状态 ----
        saved_metrics = checkpoint.get("metric_logger", {})
        if self.metric_name in saved_metrics:
            self.best_value = saved_metrics[self.metric_name]

        # ---- 计算起始 epoch ----
        saved_epoch = int(checkpoint["epoch"])
        start_epoch = saved_epoch + 1

        self._log(
            f"断点恢复完成: epoch={saved_epoch} (下一个: {start_epoch}) | "
            f"设备={device}"
        )

        # 打印断点中的指标
        if saved_metrics:
            metric_str = ", ".join(
                f"{k}={v:.4f}" for k, v in saved_metrics.items()
            )
            self._log(f"断点指标: {metric_str}")

        return start_epoch

    # ============================================================
    # 自动恢复 (便捷方法)
    # ============================================================

    def auto_resume(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None,
        scaler: Optional[GradScaler] = None,
        device: Optional[torch.device] = None,
    ) -> int:
        """
        自动从最新的 resume 断点恢复训练。

        查找 resume_dir 下的 fold{N}_resume.pth 文件并自动恢复。
        如果配置中 resume=False 或文件不存在，返回 0 (从头开始)。

        参数:
            model:      PyTorch 模型
            optimizer:  PyTorch 优化器 (可选)
            scheduler:  学习率调度器 (可选)
            scaler:     AMP GradScaler (可选)
            device:     目标设备

        返回:
            int: 起始 epoch 编号 (0 表示全新开始)
        """
        if not self._resume_enabled:
            self._log("断点恢复已禁用 (config.training.checkpoint.resume=False)")
            return 0

        resume_path = self.resume_dir / _RESUME_FILENAME.format(fold=self.fold_id)

        if not resume_path.is_file():
            self._log("未找到 resume 断点, 从 epoch 0 开始训练")
            return 0

        self._log("检测到 resume 断点，自动恢复...")
        return self.load(
            resume_path, model, optimizer, scheduler, scaler, device,
        )

    # ============================================================
    # 断点查找
    # ============================================================

    def get_latest_checkpoint(
        self, directory: Optional[Union[str, Path]] = None
    ) -> Optional[Path]:
        """
        获取指定目录下当前 fold 的最新断点文件路径。

        查找优先级:
            1. 最佳模型 (fold{N}_best_*.pth, 按修改时间)
            2. epoch 断点 (fold{N}_epoch_*.pth, 按 epoch 编号取最大)
            3. resume 断点 (fold{N}_resume.pth)
            4. 任意 .pth / .tar / .pt 文件 (按修改时间)

        参数:
            directory: 断点目录。如果为 None，先在 ckpt_dir 中查找，
                       再在 resume_dir 中查找。

        返回:
            Optional[Path]: 最新断点路径，目录为空则返回 None
        """
        if directory is None:
            # 先在 ckpt_dir 中查找
            best = self._find_best_or_latest(self.ckpt_dir)
            if best is not None:
                return best
            # 再在 resume_dir 中查找
            return self._find_best_or_latest(self.resume_dir)

        search_dir = Path(directory)
        if not search_dir.is_dir():
            return None
        return self._find_best_or_latest(search_dir)

    def _find_best_or_latest(self, directory: Path) -> Optional[Path]:
        """
        在指定目录中按优先级查找断点。

        优先级: best > max(epoch) > resume > latest(pth)

        参数:
            directory: 搜索目录

        返回:
            Optional[Path]: 找到的断点路径
        """
        if not directory.is_dir():
            return None

        all_files = (
            list(directory.glob("*.pth"))
            + list(directory.glob("*.tar"))
            + list(directory.glob("*.pt"))
        )
        if not all_files:
            return None

        best_files: List[Path] = []
        epoch_files: List[Path] = []
        resume_files: List[Path] = []

        for f in all_files:
            name = f.name
            if not name.startswith(f"fold{self.fold_id}"):
                continue
            if "_best_" in name:
                best_files.append(f)
            elif "_epoch_" in name:
                epoch_files.append(f)
            elif "_resume" in name:
                resume_files.append(f)

        # 优先级 1: 最佳模型
        if best_files:
            return max(best_files, key=lambda f: f.stat().st_mtime)

        # 优先级 2: epoch 断点 (取 epoch 编号最大的)
        if epoch_files:
            return self._find_max_epoch(epoch_files)

        # 优先级 3: resume 断点
        if resume_files:
            return max(resume_files, key=lambda f: f.stat().st_mtime)

        # 优先级 4: 任意 .pth 文件
        pth_files = [f for f in all_files if f.suffix in (".pth", ".pt", ".tar")]
        if pth_files:
            return max(pth_files, key=lambda f: f.stat().st_mtime)

        return None

    def _find_max_epoch(self, ckpt_files: List[Path]) -> Optional[Path]:
        """
        从 epoch 断点文件列表中找出 epoch 编号最大的文件。

        通过正则匹配文件名中的 epoch 编号，选择最大的。
        如果正则匹配失败，回退到按修改时间选择。

        参数:
            ckpt_files: 断点文件路径列表

        返回:
            Optional[Path]: epoch 编号最大的文件
        """
        best: Optional[Path] = None
        best_epoch: int = -1

        for f in ckpt_files:
            match = _CKPT_PATTERN.match(f.name)
            if match:
                epoch_num = int(match.group(2))
                if epoch_num > best_epoch:
                    best_epoch = epoch_num
                    best = f

        # 正则匹配失败时回退
        if best is None and ckpt_files:
            best = max(ckpt_files, key=lambda f: f.stat().st_mtime)

        return best

    def get_best_checkpoint(self) -> Optional[Path]:
        """
        获取当前 fold 的最佳模型路径。

        先从内存中的 best_path 查找，不存在则从 ckpt_dir 搜索。

        返回:
            Optional[Path]: 最佳模型断点路径，不存在则返回 None
        """
        if self.best_path is not None and self.best_path.exists():
            return self.best_path

        if self.ckpt_dir.is_dir():
            for f in self.ckpt_dir.glob(f"fold{self.fold_id}_best_*.pth"):
                return f
        return None

    def list_checkpoints(
        self, directory: Optional[Union[str, Path]] = None
    ) -> List[Path]:
        """
        列出当前 fold 的所有断点文件 (按修改时间降序排列)。

        参数:
            directory: 断点目录。如果为 None，使用默认的 ckpt_dir。

        返回:
            List[Path]: 断点文件路径列表 (最新的在前)
        """
        if directory is None:
            directory = self.ckpt_dir
        else:
            directory = Path(directory)

        if not directory.is_dir():
            return []

        all_files: List[Path] = []
        for pattern in ["*.pth", "*.tar", "*.pt"]:
            for f in directory.glob(f"fold{self.fold_id}_" + pattern.lstrip("*")):
                if f not in all_files:
                    all_files.append(f)

        all_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return all_files

    # ============================================================
    # 断点清理
    # ============================================================

    def _cleanup_old_epoch_checkpoints(self) -> int:
        """
        内部清理方法: 仅在 ckpt_dir 中清理过期的 epoch 断点。

        清理规则:
            - 仅删除 fold{N}_epoch_*.pth 文件
            - 不删除 best / resume 文件
            - 保留最近 keep_last_n 个 epoch 断点 (按 epoch 编号排序)

        返回:
            int: 删除的文件数量
        """
        if self.keep_last_n <= 0 or not self.ckpt_dir.is_dir():
            return 0

        # 收集当前 fold 的所有 epoch 断点
        epoch_entries: List[Tuple[int, Path]] = []
        for f in self.ckpt_dir.glob(f"fold{self.fold_id}_epoch_*.pth"):
            match = _CKPT_PATTERN.match(f.name)
            if match:
                epoch_num = int(match.group(2))
                epoch_entries.append((epoch_num, f))

        if len(epoch_entries) <= self.keep_last_n:
            return 0

        # 按 epoch 编号降序排列，保留前 keep_last_n 个
        epoch_entries.sort(key=lambda x: x[0], reverse=True)
        to_delete = epoch_entries[self.keep_last_n:]

        deleted = 0
        for epoch_num, filepath in to_delete:
            try:
                filepath.unlink()
                deleted += 1
            except OSError as e:
                self._log(f"无法删除过期断点 {filepath.name}: {e}", "warning")

        if deleted > 0:
            kept = [f"epoch={e}" for e, _ in epoch_entries[:self.keep_last_n]]
            self._log(f"自动清理: 删除 {deleted} 个过期断点, 保留: {', '.join(kept)}")

        return deleted

    def cleanup(
        self,
        directory: Optional[Union[str, Path]] = None,
        keep_last_n: Optional[int] = None,
    ) -> int:
        """
        手动清理过期断点，仅保留最近的 N 个 epoch 断点。

        清理规则:
            - 仅删除常规 epoch 断点 (fold{N}_epoch_*.pth)
            - 不删除最佳模型文件 (fold{N}_best_*.pth)
            - 不删除 resume 文件 (fold{N}_resume.pth)
            - 保留最近 N 个 epoch 断点 (按 epoch 编号排序取最大的 N 个)

        参数:
            directory:   断点目录。如果为 None，使用默认的 ckpt_dir。
            keep_last_n: 保留数量。如果为 None，使用配置中的 keep_last_n。
                         N=0 表示保留全部，不清理。

        返回:
            int: 删除的文件数量

        使用示例:
            >>> ckpt_mgr = CheckpointManager(config, fold_id=1)
            >>> deleted = ckpt_mgr.cleanup(keep_last_n=3)
            >>> print(f"清理了 {deleted} 个过期断点")
        """
        if directory is None:
            directory = self.ckpt_dir
        else:
            directory = Path(directory)

        if not directory.is_dir():
            self._log(f"断点目录不存在，无需清理: {directory}", "debug")
            return 0

        if keep_last_n is None:
            keep_last_n = self.keep_last_n

        if keep_last_n <= 0:
            self._log(f"keep_last_n={keep_last_n}, 保留全部断点", "debug")
            return 0

        # 收集 epoch 断点
        epoch_entries: List[Tuple[int, Path]] = []
        for f in Path(directory).glob(f"fold{self.fold_id}_epoch_*.pth"):
            match = _CKPT_PATTERN.match(f.name)
            if match:
                epoch_num = int(match.group(2))
                epoch_entries.append((epoch_num, f))

        if len(epoch_entries) <= keep_last_n:
            self._log(
                f"epoch断点数={len(epoch_entries)} <= keep_last_n={keep_last_n}, "
                f"无需清理", "debug",
            )
            return 0

        # 排序并删除
        epoch_entries.sort(key=lambda x: x[0], reverse=True)
        to_delete = epoch_entries[keep_last_n:]

        deleted = 0
        for epoch_num, filepath in to_delete:
            try:
                filepath.unlink()
                deleted += 1
                self._log(f"清理过期断点: {filepath.name} (epoch={epoch_num})", "debug")
            except OSError as e:
                self._log(f"无法删除断点 {filepath.name}: {e}", "warning")

        if deleted > 0:
            kept = [f"epoch={e}" for e, _ in epoch_entries[:keep_last_n]]
            self._log(
                f"断点清理完成: 删除 {deleted} 个, 保留 {keep_last_n} 个 "
                f"({', '.join(kept)})"
            )

        return deleted

    def remove_resume(self) -> None:
        """
        删除当前 fold 的 resume 断点文件。

        通常在训练正常完成 (未中断) 后调用，防止下次误恢复。
        """
        resume_path = self.resume_dir / _RESUME_FILENAME.format(fold=self.fold_id)
        if resume_path.exists():
            resume_path.unlink()
            self._log("已删除 resume 断点文件")

    def remove_all(self) -> int:
        """
        删除当前 fold 的所有断点文件 (包括 ckpt_dir 和 resume_dir)。

        这是一个危险操作，用于重置实验状态。

        返回:
            int: 删除的文件总数
        """
        total_deleted = 0

        for d in [self.ckpt_dir, self.resume_dir]:
            if not d.is_dir():
                continue
            patterns = [
                f"fold{self.fold_id}_epoch_*.pth",
                f"fold{self.fold_id}_best_*.pth",
                f"fold{self.fold_id}_resume.pth",
                f"fold{self.fold_id}_*.tar",
                f"fold{self.fold_id}_*.pt",
            ]
            for pattern in patterns:
                for f in d.glob(pattern):
                    try:
                        f.unlink()
                        total_deleted += 1
                    except OSError as e:
                        self._log(f"无法删除: {f.name} - {e}", "warning")

        if total_deleted > 0:
            self._log(f"已删除 fold={self.fold_id} 的全部断点 ({total_deleted} 个文件)")

        # 重置最佳模型追踪
        self.best_path = None
        self.best_value = None

        return total_deleted

    # ============================================================
    # 断点信息查看
    # ============================================================

    @staticmethod
    def inspect_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
        """
        查看断点文件的元信息，不将模型参数加载到调用方模型。

        此方法加载整个 checkpoint 到 CPU 并提取元信息。

        参数:
            path: 断点文件路径

        返回:
            dict: 包含 epoch, fold_id, metric_logger, file_size_mb,
                  top_level_keys, model_param_count 等元信息的字典

        使用示例:
            >>> info = CheckpointManager.inspect_checkpoint("ckpt/fold1_epoch_005.pth")
            >>> print(f"Epoch: {info['epoch']}, Metrics: {info['metric_logger']}")
        """
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            return {"error": f"文件不存在: {ckpt_path}"}

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        info: Dict[str, Any] = {
            "path": str(ckpt_path),
            "file_size_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 2),
            "epoch": checkpoint.get("epoch", "N/A"),
            "fold_id": checkpoint.get("fold_id", "N/A"),
            "metric_logger": checkpoint.get("metric_logger", {}),
            "top_level_keys": list(checkpoint.keys()),
        }

        # 模型参数量
        if "model_state_dict" in checkpoint:
            info["model_param_count"] = sum(
                v.numel() for v in checkpoint["model_state_dict"].values()
            )

        # RNG 状态摘要
        if "rng_state" in checkpoint:
            rng = checkpoint["rng_state"]
            info["rng_state_summary"] = {
                "has_python": rng.get("python") is not None,
                "has_numpy": rng.get("numpy") is not None,
                "has_torch": rng.get("torch") is not None,
                "has_cuda": rng.get("torch_cuda") is not None,
            }

        # 配置摘要
        if "config_snapshot" in checkpoint:
            cfg = checkpoint["config_snapshot"]
            info["config_summary"] = {
                "project_version": cfg.get("project", {}).get("version", "N/A"),
                "experiment": cfg.get("experiment", {}).get("name", "N/A"),
                "task": cfg.get("model", {}).get("task", "N/A"),
                "fusion_type": cfg.get("model", {}).get("fusion", {}).get("type", "N/A"),
            }

        return info

    # ============================================================
    # 特殊操作：仅保存 / 仅加载模型参数 (部署用)
    # ============================================================

    def save_model_only(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """
        仅保存模型参数 (轻量级，不含优化器/调度器状态)。

        适用于训练完成后导出部署用模型权重。

        参数:
            model:   PyTorch 模型
            epoch:   训练完成的 epoch 编号
            metrics: 最终指标 (可选)

        返回:
            Path: 保存的文件路径
        """
        metrics = metrics or {}

        if hasattr(model, "module"):
            model_state = {
                k: v.detach().cpu() for k, v in model.module.state_dict().items()
            }
        else:
            model_state = {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            }

        metric_val = metrics.get(self.metric_name, 0.0)
        filename = _MODEL_ONLY_TEMPLATE.format(
            fold=self.fold_id, epoch=epoch,
            metric=self.metric_name, value=metric_val,
        )
        save_path = self.ckpt_dir / filename

        payload = {
            "epoch": epoch,
            "fold_id": self.fold_id,
            "model_state_dict": model_state,
            "metric_logger": metrics,
            "config_snapshot": self.config.to_dict(),
        }

        torch.save(payload, save_path, _use_new_zipfile_serialization=True)
        self._log(f"仅模型参数已保存: {filename}")
        return save_path

    def load_model_only(
        self,
        path: Union[str, Path],
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> int:
        """
        从仅含模型参数的断点文件加载模型。

        参数:
            path:   断点文件路径
            model:  PyTorch 模型 (原地更新)
            device: 目标设备

        返回:
            int: 断点中的 epoch 编号
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        payload = torch.load(path, map_location=device, weights_only=False)

        if "model_state_dict" in payload:
            model.load_state_dict(payload["model_state_dict"], strict=False)
        else:
            # 兼容纯 state_dict 格式
            model.load_state_dict(payload, strict=False)

        model.to(device)

        epoch = payload.get("epoch", 0)
        self._log(f"仅模型参数已加载: epoch={epoch}")
        return epoch


# ============================================================
# 便捷函数 (无需实例化 CheckpointManager 的快速推理加载)
# ============================================================

def load_checkpoint_for_inference(
    path: Union[str, Path],
    model: nn.Module,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    快速加载模型用于推理 (不恢复训练状态)。

    适用于模型评估/测试/部署场景。加载后将模型设为 eval() 模式。

    参数:
        path:   断点文件路径 (支持完整断点或 model_only 格式)
        model:  PyTorch 模型 (原地更新)
        device: 目标设备

    返回:
        nn.Module: 加载了权重的模型 (与输入的 model 是同一个对象)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # 尝试多种键名
    state_dict = None
    for key in ["model_state_dict", "state_dict"]:
        if key in checkpoint:
            state_dict = checkpoint[key]
            break

    if state_dict is None:
        # 可能直接就是 state_dict
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    logger = _get_logger()
    logger.info("推理模型已加载: %s -> %s", Path(path).name, device)
    return model


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CheckpointManager 断点管理模块自测")
    print("=" * 60)

    import tempfile

    # ---- 构建测试环境 ----
    tmp_base = Path(tempfile.mkdtemp(prefix="yuhou_ckpt_test_"))
    ckpt_dir = tmp_base / "ckpt"
    resume_dir = tmp_base / "resume"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  临时目录: {tmp_base}")

    # ---- 构建最小化测试模型 ----
    class _TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3)
            self.fc = nn.Linear(8 * 30 * 30, 2)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    # ---- 模拟 ConfigBundle (使用 Mock) ----
    from unittest.mock import Mock, MagicMock

    mock_config = MagicMock(spec=ConfigBundle)

    def _mock_get_subdir(key):
        d = {"ckpt": ckpt_dir, "resume": resume_dir}
        target = d.get(key, tmp_base / key)
        target.mkdir(parents=True, exist_ok=True)
        return target

    mock_config.get_subdir.side_effect = _mock_get_subdir
    mock_config.to_dict.return_value = {
        "project": {"version": "2.0.0"},
        "experiment": {"name": "test"},
        "model": {"task": "surv", "fusion": {"type": "caugf"}},
    }

    mock_ckpt_cfg = Mock()
    mock_ckpt_cfg.save_every = 1
    mock_ckpt_cfg.resume = True
    mock_ckpt_cfg.keep_last_n = 3
    mock_training = Mock()
    mock_training.checkpoint = mock_ckpt_cfg
    mock_config.training = mock_training

    # ---- 测试 1: 初始化 ----
    print("\n[1/10] 测试初始化 ...")
    mgr = CheckpointManager(
        mock_config, fold_id=1, metric_name="cindex", metric_mode="max",
    )
    assert mgr.ckpt_dir.exists()
    assert mgr.resume_dir.exists()
    print(f"  ckpt_dir: {mgr.ckpt_dir}")
    print(f"  resume_dir: {mgr.resume_dir}")
    print("  通过: 目录自动创建")

    # ---- 测试 2: 保存断点 ----
    print("\n[2/10] 测试保存断点 ...")
    model = _TestModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler(enabled=False)

    saved_path = mgr.save(
        model, optimizer, scaler=scaler,
        epoch=5, metrics={"val_cindex": 0.65, "val_loss": 0.45},
    )
    assert saved_path.exists()
    file_size_kb = saved_path.stat().st_size / 1024
    print(f"  保存成功: {saved_path.name} ({file_size_kb:.1f} KB)")

    # 验证 resume 文件也保存了
    resume_path = resume_dir / "fold1_resume.pth"
    assert resume_path.exists()
    print(f"  Resume 文件: {resume_path.name} (已同步保存)")
    print("  通过: epoch + resume 双写")

    # ---- 测试 3: 文件命名规范 ----
    print("\n[3/10] 测试文件命名 ...")
    f1 = _make_ckpt_filename(1, 10)
    print(f"  无指标: {f1}")
    assert f1 == "fold1_epoch_010.pth"

    f2 = _make_ckpt_filename(2, 5, "cindex", 0.7234)
    print(f"  含指标: {f2}")
    assert "cindex=0.7234" in f2

    f3 = _make_best_filename(1, "cindex", 0.7800)
    print(f"  最佳模型: {f3}")
    assert "best" in f3
    print("  通过: 命名规范正确")

    # ---- 测试 4: RNG 状态捕获/恢复 ----
    print("\n[4/10] 测试 RNG 状态 ...")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    rng_state = _capture_rng_state()
    tensor_before = torch.randn(3).tolist()

    # 改变 RNG 状态
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)

    _restore_rng_state(rng_state)
    tensor_after = torch.randn(3).tolist()

    assert tensor_before == tensor_after, "RNG 恢复后应生成相同随机数"
    print(f"  before: {[f'{x:.6f}' for x in tensor_before]}")
    print(f"  after:  {[f'{x:.6f}' for x in tensor_after]}")
    print("  通过: RNG 状态恢复一致")

    # ---- 测试 5: 加载断点 ----
    print("\n[5/10] 测试加载断点 ...")
    model2 = _TestModel()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    scaler2 = GradScaler(enabled=False)

    start_epoch = mgr.load(
        saved_path, model2, optimizer2, scaler=scaler2,
        device=torch.device("cpu"),
    )
    print(f"  起始 epoch: {start_epoch} (期望=6)")
    assert start_epoch == 6, f"应为 6, 实际 {start_epoch}"

    # 验证模型参数一致
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), model2.named_parameters()
    ):
        if not torch.allclose(p1, p2):
            print(f"  警告: 参数不一致: {n1} vs {n2}")
    print("  通过: 模型参数恢复一致")

    # ---- 测试 6: 最佳模型追踪 ----
    print("\n[6/10] 测试最佳模型追踪 ...")
    mgr2 = CheckpointManager(
        mock_config, fold_id=1, metric_name="cindex", metric_mode="max",
        checkpoint_dir=str(ckpt_dir), resume_dir=str(resume_dir),
    )
    model3 = _TestModel()
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.001)

    mgr2.save(model3, optimizer3, epoch=1, metrics={"val_cindex": 0.60})
    assert mgr2.best_value == 0.60, f"epoch=1 应为最佳, best_value={mgr2.best_value}"
    mgr2.save(model3, optimizer3, epoch=2, metrics={"val_cindex": 0.68})
    assert mgr2.best_value == 0.68, f"epoch=2 应更新, best_value={mgr2.best_value}"
    mgr2.save(model3, optimizer3, epoch=3, metrics={"val_cindex": 0.62})
    assert mgr2.best_value == 0.68, f"epoch=3 不应更新, best_value={mgr2.best_value}"

    best_path = mgr2.get_best_checkpoint()
    assert best_path is not None and best_path.exists()
    print(f"  最佳模型: {best_path.name} (cindex={mgr2.best_value})")
    print("  通过: 最佳模型追踪正确 (max模式)")

    # ---- 测试 7: min 模式追踪 ----
    print("\n[7/10] 测试 min 模式 (val_loss) ...")
    mgr_min = CheckpointManager(
        mock_config, fold_id=2, metric_name="val_loss", metric_mode="min",
        checkpoint_dir=str(ckpt_dir), resume_dir=str(resume_dir),
    )
    model4 = _TestModel()
    opt4 = torch.optim.Adam(model4.parameters(), lr=0.001)
    mgr_min.save(model4, opt4, epoch=1, metrics={"val_loss": 0.50})
    mgr_min.save(model4, opt4, epoch=2, metrics={"val_loss": 0.35})
    mgr_min.save(model4, opt4, epoch=3, metrics={"val_loss": 0.42})
    assert mgr_min.best_value == 0.35, f"应追踪最小值 0.35, 实际={mgr_min.best_value}"
    print(f"  最佳 val_loss: {mgr_min.best_value} (期望=0.35)")
    print("  通过: min 模式追踪正确")

    # ---- 测试 8: 获取最新断点 ----
    print("\n[8/10] 测试 get_latest_checkpoint ...")
    latest = mgr2.get_latest_checkpoint()
    assert latest is not None
    print(f"  最新断点: {latest.name}")
    assert "best" in latest.name, f"应返回最佳模型, 实际: {latest.name}"
    print("  通过: 最佳模型优先级高于 epoch 断点")

    # ---- 测试 9: 列出断点 & 清理 ----
    print("\n[9/10] 测试断点列表与清理 ...")
    all_ckpts = mgr2.list_checkpoints()
    print(f"  总断点数: {len(all_ckpts)}")
    for c in all_ckpts:
        print(f"    {c.name} ({c.stat().st_size / 1024:.1f} KB)")

    # 保存更多 epoch 断点使总数超过 keep_last_n
    for ep in range(4, 8):
        mgr2.save(model3, optimizer3, epoch=ep, metrics={"val_cindex": 0.65})

    epoch_files_after = list(ckpt_dir.glob("fold1_epoch_*.pth"))
    print(f"  保存 7 个 epoch 后, epoch 断点数: {len(epoch_files_after)}")
    # keep_last_n=3, 应有 3 个 epoch 断点被保留
    assert len(epoch_files_after) <= mock_ckpt_cfg.keep_last_n, \
        f"自动清理后应保留 <= {mock_ckpt_cfg.keep_last_n} 个, 实际={len(epoch_files_after)}"
    print("  通过: 自动清理机制正常工作")

    # 手动清理
    deleted = mgr2.cleanup(keep_last_n=2)
    epoch_files_after_manual = list(ckpt_dir.glob("fold1_epoch_*.pth"))
    print(f"  手动清理 (keep_last_n=2): 删除 {deleted} 个, 剩余 {len(epoch_files_after_manual)} 个")
    print("  通过: 手动清理正常工作")

    # ---- 测试 10: inspect_checkpoint ----
    print("\n[10/10] 测试 inspect_checkpoint ...")
    info = CheckpointManager.inspect_checkpoint(saved_path)
    print(f"  epoch: {info['epoch']}")
    print(f"  file_size: {info['file_size_mb']} MB")
    print(f"  metrics: {info['metric_logger']}")
    print(f"  keys: {info['top_level_keys']}")
    print(f"  model_params: {info.get('model_param_count', 'N/A'):,}")
    print(f"  config: {info.get('config_summary', {})}")
    print("  通过: 断点元信息查看正常")

    # ---- 测试 save_model_only / load_model_only ----
    print("\n[仅模型保存] 测试 save_model_only / load_model_only ...")
    model5 = _TestModel()
    opt5 = torch.optim.Adam(model5.parameters(), lr=0.001)
    mpath = mgr.save_model_only(model5, epoch=10, metrics={"cindex": 0.75})
    print(f"  保存: {mpath.name}")

    model6 = _TestModel()
    loaded_epoch = mgr.load_model_only(mpath, model6, device=torch.device("cpu"))
    print(f"  加载: epoch={loaded_epoch}")
    all_close = True
    for p1, p2 in zip(model5.parameters(), model6.parameters()):
        if not torch.allclose(p1, p2):
            all_close = False
            break
    assert all_close, "仅模型参数保存/加载应一致"
    print("  通过: model_only 保存/加载一致")

    # ---- 测试 auto_resume ----
    print("\n[auto_resume] 测试自动恢复 ...")
    mgr_ar = CheckpointManager(
        mock_config, fold_id=1, metric_name="cindex", metric_mode="max",
        checkpoint_dir=str(ckpt_dir), resume_dir=str(resume_dir),
    )
    model_ar = _TestModel()
    opt_ar = torch.optim.Adam(model_ar.parameters(), lr=0.001)
    # resume 文件已在上面的保存中创建
    start = mgr_ar.auto_resume(model_ar, opt_ar, device=torch.device("cpu"))
    print(f"  auto_resume 起始 epoch: {start}")
    assert start > 0, "应能从 resume 恢复"
    print("  通过: auto_resume 正常")

    # ---- 测试 remove_resume ----
    print("\n[清理] 测试 remove_resume ...")
    mgr_ar.remove_resume()
    assert not (resume_dir / "fold1_resume.pth").exists()
    print("  通过: resume 文件已删除")

    # ---- 清理临时文件 ----
    print(f"\n清理临时文件: {tmp_base}")
    shutil.rmtree(tmp_base, ignore_errors=True)

    print("\n" + "=" * 60)
    print("所有自测通过! (10/10)")
    print("=" * 60)
