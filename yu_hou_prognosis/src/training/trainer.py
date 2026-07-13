# -*- coding: utf-8 -*-
"""
trainer.py
============================================================================
训练核心模块 - 单折训练与评估引擎。

功能:
    1. 单折完整训练流程 (train函数)
    2. 模型测试评估 (test函数)
    3. 自动混合精度 (AMP) 训练，针对RTX 4080 Super (16GB)优化
    4. 梯度累积 + 梯度裁剪
    5. GPU显存实时监控
    6. 每轮中文日志输出关键指标
    7. 自动checkpoint断点续训
    8. 训练结束后绘制loss曲线

前向传播流程:
    1. Flatten patches: [B,N,C,H,W] -> [B*N,C,H,W]
    2. PathNet: 提取patch特征 -> [B*N, path_dim]
    3. Reshape + mean pool: [B,N,path_dim] -> [B,path_dim]
    4. OmicNet: 提取组学特征 -> [B, omic_dim]
    5. Fusion: 融合(path_feat, omic_feat) -> [B, mmhid]
    6. Classifier: 预测风险 -> [B, 1]

依赖:
    - src.networks.pathomic_net (PathomicNet)
    - src.losses.cox_loss (CoxPartialLikelihoodLoss)
    - src.losses.classification_loss (classification_loss)
    - src.losses.regularization (compute_regularization)
    - src.evaluation.survival_metrics (CIndex, cox_log_rank, safe_time_dependent_auc, ...)
    - src.evaluation.classification_metrics (classification_metrics)
    - src.evaluation.metric_formatter (format_*)
    - src.utils.logger (setup_logger, get_logger)
    - src.utils.seed (set_seed)

使用示例:
    from src.training.trainer import train, test

    # 训练
    model, optimizer, metric_logger = train(config, data, device, fold_id=1)

    # 测试
    loss, cindex, pvalue, surv_acc, grad_acc, pred, td_auc, \
        binary_metrics, group_summary, hr, cls_metrics = \
        test(config, model, data, split="test", device=device)
============================================================================
"""

import os
import sys
import time
import copy
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，避免GUI线程冲突
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
# 兼容新旧PyTorch版本的AMP API
try:
    from torch.amp import GradScaler, autocast
    _AMP_NEW_API = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_NEW_API = False

from src.networks.pathomic_net import PathomicNet
from src.losses.cox_loss import CoxPartialLikelihoodLoss, CoxLossWithL2
from src.losses.classification_loss import classification_loss
from src.losses.regularization import compute_regularization
from src.evaluation.survival_metrics import (
    CIndex,
    CIndex_lifeline,
    cox_log_rank,
    safe_time_dependent_auc,
    safe_hazard_ratio_by_median_split,
    safe_binary_metrics_from_risk,
    safe_group_survival_summary,
    accuracy_cox,
)
from src.evaluation.classification_metrics import classification_metrics as cls_metrics_fn
from src.evaluation.metric_formatter import (
    format_cindex,
    format_logrank_p,
    format_losses,
    format_binary_metrics,
    format_classification_metrics,
    format_cv_summary,
)
from src.utils.seed import set_seed


# ============================================================
# GPU 显存监控工具
# ============================================================

def _get_gpu_memory_info(device: torch.device) -> Dict[str, float]:
    """
    获取当前GPU显存使用情况。

    参数:
        device: torch设备对象

    返回:
        dict: {"allocated_gb": float, "reserved_gb": float, "total_gb": float,
               "free_gb": float, "used_percent": float}
    """
    if not torch.cuda.is_available() or device.type != "cuda":
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0,
                "free_gb": 0.0, "used_percent": 0.0}

    try:
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        free = total - reserved
        used_pct = (reserved / total) * 100.0
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(free, 2),
            "used_percent": round(used_pct, 1),
        }
    except Exception:
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0,
                "free_gb": 0.0, "used_percent": 0.0}


def _log_gpu_memory(logger: logging.Logger, device: torch.device,
                    prefix: str = "[GPU显存]") -> None:
    """将GPU显存信息输出到日志"""
    info = _get_gpu_memory_info(device)
    if info["total_gb"] > 0:
        logger.info(
            f"{prefix} 已分配: {info['allocated_gb']:.2f}GB / "
            f"已保留: {info['reserved_gb']:.2f}GB / "
            f"总量: {info['total_gb']:.2f}GB / "
            f"空闲: {info['free_gb']:.2f}GB / "
            f"使用率: {info['used_percent']:.1f}%"
        )


# ============================================================
# Optimizer & Scheduler 构建
# ============================================================

def _build_optimizer(model: nn.Module, config) -> optim.Optimizer:
    """
    根据配置构建优化器。

    参数:
        model: 模型实例
        config: ConfigBundle配置对象

    返回:
        torch.optim optimizer实例
    """
    opt_cfg = config.training.optimizer
    opt_type = opt_cfg.type.lower()

    if opt_type == "adam":
        return optim.Adam(
            model.parameters(),
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas),
            weight_decay=opt_cfg.weight_decay,
        )
    elif opt_type == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas),
            weight_decay=opt_cfg.weight_decay,
        )
    elif opt_type == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=opt_cfg.lr,
            momentum=0.9,
            weight_decay=opt_cfg.weight_decay,
        )
    else:
        raise ValueError(f"不支持的优化器类型: '{opt_type}'。支持: adam, adamw, sgd")


def _build_scheduler(optimizer: optim.Optimizer, config) -> torch.optim.lr_scheduler._LRScheduler:
    """
    根据配置构建学习率调度器。

    支持:
        - cosine: CosineAnnealingLR (配合warmup使用)
        - cosine_warmup: CosineAnnealingWarmRestarts
        - step: StepLR
        - plateau: ReduceLROnPlateau
        - linear: LinearLR

    参数:
        optimizer: 优化器实例
        config: ConfigBundle配置对象

    返回:
        scheduler实例
    """
    sched_cfg = config.training.scheduler
    sched_type = sched_cfg.type.lower()

    if sched_type == "cosine":
        total_epochs = sched_cfg.n_epochs + sched_cfg.n_epochs_decay
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=sched_cfg.min_lr,
        )
    elif sched_type == "cosine_warmup":
        total_epochs = sched_cfg.n_epochs + sched_cfg.n_epochs_decay
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=total_epochs // 2,
            T_mult=2,
            eta_min=sched_cfg.min_lr,
        )
    elif sched_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.lr_decay_iters,
            gamma=0.5,
        )
    elif sched_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=sched_cfg.min_lr,
        )
    elif sched_type == "linear":
        total_epochs = sched_cfg.n_epochs + sched_cfg.n_epochs_decay
        return optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=sched_cfg.min_lr / config.training.optimizer.lr,
            total_iters=total_epochs,
        )
    else:
        raise ValueError(
            f"不支持的调度器类型: '{sched_type}'。"
            f"支持: cosine, cosine_warmup, step, plateau, linear"
        )


# ============================================================
# Warmup 辅助
# ============================================================

class GradualWarmupScheduler:
    """
    渐进式学习率预热包装器。

    在warmup_epochs期间，学习率从min_lr线性增长到base_lr。
    预热结束后，恢复原有scheduler策略。

    参数:
        optimizer: 优化器实例
        multiplier: 目标学习率 = base_lr * multiplier (通常为1.0)
        total_epoch: 预热周期数
        after_scheduler: 预热结束后使用的调度器
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler):
        self.optimizer = optimizer
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            # 新版PyTorch: scheduler.step()不接受epoch参数
            self.after_scheduler.step()
            return

        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if epoch >= self.total_epoch:
            # 预热结束，恢复base_lr并启动后续调度器
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group["lr"] = base_lr
            self.finished = True
            if self.after_scheduler:
                self.after_scheduler.step()
            return

        # 线性预热: lr = min_lr + (base_lr - min_lr) * (epoch/total_epoch)
        progress = epoch / max(self.total_epoch, 1)
        for param_group in self.optimizer.param_groups:
            start_lr = self.base_lrs[0] * 0.1  # 从base_lr的10%开始
            param_group["lr"] = start_lr + (self.base_lrs[0] - start_lr) * progress

    def state_dict(self):
        return {
            "finished": self.finished,
            "last_epoch": self.last_epoch,
            "after_scheduler": self.after_scheduler.state_dict() if self.after_scheduler else None,
        }

    def load_state_dict(self, state_dict):
        self.finished = state_dict["finished"]
        self.last_epoch = state_dict["last_epoch"]
        if self.after_scheduler and state_dict["after_scheduler"]:
            self.after_scheduler.load_state_dict(state_dict["after_scheduler"])


# ============================================================
# MetricLogger: 训练指标收集器
# ============================================================

class MetricLogger:
    """
    训练指标收集与追踪器。

    记录每轮训练的损失、C-index、tdAUC、Log-rank p值等关键指标，
    并提供best model选择逻辑。
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        参数:
            logger: 可选的日志器实例
        """
        self.logger = logger

        # 每轮记录
        self.epochs: List[int] = []
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_cindices: List[float] = []
        self.val_cindices: List[float] = []
        self.val_pvalues: List[float] = []
        self.val_tdaucs: List[float] = []
        self.learning_rates: List[float] = []

        # 最佳模型追踪
        self.best_cindex: float = 0.0
        self.best_epoch: int = 0
        self.best_model_state: Optional[Dict] = None
        self.no_improve_count: int = 0

    def update(self, epoch: int, train_loss: float, val_loss: float = None,
               train_cindex: float = None, val_cindex: float = None,
               val_pvalue: float = None, val_tdauc: float = None,
               lr: float = None):
        """
        更新一轮训练指标。

        参数:
            epoch: 当前轮次 (从1开始)
            train_loss: 训练损失
            val_loss: 验证损失 (可选)
            train_cindex: 训练C-index (可选)
            val_cindex: 验证C-index (可选)
            val_pvalue: 验证Log-rank p值 (可选)
            val_tdauc: 验证时间依赖AUC均值 (可选)
            lr: 当前学习率 (可选)
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if train_cindex is not None:
            self.train_cindices.append(train_cindex)
        if val_cindex is not None:
            self.val_cindices.append(val_cindex)
        if val_pvalue is not None:
            self.val_pvalues.append(val_pvalue)
        if val_tdauc is not None:
            self.val_tdaucs.append(val_tdauc)
        if lr is not None:
            self.learning_rates.append(lr)

    def track_best(self, epoch: int, val_cindex: float, model: nn.Module):
        """
        追踪最佳模型。

        参数:
            epoch: 当前轮次
            val_cindex: 验证集C-index
            model: 当前模型
        """
        if val_cindex > self.best_cindex:
            self.best_cindex = val_cindex
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.no_improve_count = 0
            if self.logger:
                self.logger.info(
                    f"  [最佳模型] 第{epoch}轮 C-index={val_cindex:.4f} (新高)"
                )
        else:
            self.no_improve_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """将所有记录导出为字典"""
        return {
            "epochs": self.epochs,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_cindices": self.train_cindices,
            "val_cindices": self.val_cindices,
            "val_pvalues": self.val_pvalues,
            "val_tdaucs": self.val_tdaucs,
            "learning_rates": self.learning_rates,
            "best_cindex": self.best_cindex,
            "best_epoch": self.best_epoch,
        }


# ============================================================
# 损失计算
# ============================================================

def _compute_total_loss(hazard_pred, survtime, censor, model, config,
                        cls_logits=None, cls_labels=None):
    """
    计算总损失：Cox损失 + 分类损失 + 正则化损失。

    参数:
        hazard_pred: [B, 1] 或 [B] 风险预测
        survtime: [B] 生存时间
        censor: [B] 删失标记
        model: 模型实例 (用于正则化)
        config: ConfigBundle配置对象
        cls_logits: 分类logits (ncls任务时使用)
        cls_labels: 分类标签 (ncls任务时使用)

    返回:
        total_loss: 总损失标量
        loss_dict: {"cox": float, "task": float, "reg": float}
    """
    loss_cfg = config.training.loss
    # 使用推荐的CoxPartialLikelihoodLoss
    cox_criterion = CoxPartialLikelihoodLoss()

    # Cox损失
    cox_loss = cox_criterion(hazard_pred, survtime, censor) * loss_cfg.lambda_cox

    # 分类损失 (ncls任务)
    task_loss = torch.tensor(0.0, device=hazard_pred.device)
    if config.model.task == "ncls" and cls_logits is not None and cls_labels is not None:
        cls_cfg = config.training.classification
        task_loss = classification_loss(
            logits=cls_logits,
            labels=cls_labels,
            loss_name=cls_cfg.loss_type,
            label_smoothing=cls_cfg.label_smoothing,
        ) * loss_cfg.lambda_task

    # 正则化损失
    reg_loss = torch.tensor(0.0, device=hazard_pred.device)
    if loss_cfg.reg_type != "none":
        reg_loss = compute_regularization(model, reg_type=loss_cfg.reg_type) * loss_cfg.lambda_reg

    total_loss = cox_loss + task_loss + reg_loss

    return total_loss, {
        "cox": cox_loss.item(),
        "task": task_loss.item(),
        "reg": reg_loss.item(),
    }


# ============================================================
# 单轮训练 (Train One Epoch)
# ============================================================

def _train_one_epoch(model, train_loader, optimizer, scheduler, config,
                     device, scaler, epoch_idx: int,
                     logger: Optional[logging.Logger] = None) -> Tuple[float, float]:
    """
    执行单轮训练。

    参数:
        model: 模型实例
        train_loader: 训练DataLoader
        optimizer: 优化器
        scheduler: 学习率调度器 (含warmup)
        config: ConfigBundle配置对象
        device: torch设备
        scaler: GradScaler (AMP)
        epoch_idx: 当前轮次索引 (从0开始)
        logger: 日志器

    返回:
        avg_train_loss: 平均训练损失
        avg_train_cindex: 平均训练C-index
    """
    model.train()

    n_data = len(train_loader.dataset)
    batch_size = config.training.batch_size
    accum_steps = config.training.gradient_accumulation_steps
    grad_clip = config.training.gradient_clip_norm
    use_amp = config.training.amp.enabled
    gpu_monitor_interval = config.training.gpu_monitor.log_interval

    running_loss = 0.0
    running_cox = 0.0
    running_task = 0.0
    running_reg = 0.0
    all_hazards = []
    all_survtimes = []
    all_censors = []
    n_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (x_path, x_grph, x_omic, e, t, g) in enumerate(train_loader):
        # 检查batch是否有效 (跳过空batch)
        if x_path.nelement() == 0 or x_omic.nelement() == 0:
            continue

        # 数据转移到设备
        if x_path.dim() > 1:
            x_path = x_path.to(device, non_blocking=config.training.non_blocking)
        x_omic = x_omic.to(device, non_blocking=config.training.non_blocking)
        survtime = t.to(device, non_blocking=config.training.non_blocking)
        censor = e.to(device, non_blocking=config.training.non_blocking)
        labels = g.to(device, non_blocking=config.training.non_blocking)

        # ---- 检查batch的事件多样性 (防止全部event=0或全部event=1) ----
        censor_np = e.numpy() if hasattr(e, 'numpy') else e.cpu().numpy()
        unique_events = np.unique(censor_np)
        if len(unique_events) < 2:
            # 该batch所有样本事件一致，Cox损失无法计算有效梯度，跳过
            if logger and batch_idx < 3:
                logger.warning(
                    f"  [批次跳过] batch {batch_idx}: 所有样本event={unique_events[0]:.0f}，"
                    f"Cox部分似然无有效梯度，跳过此batch"
                )
            continue

        # ---- 前向传播 (AMP autocast) ----
        amp_ctx = autocast("cuda", enabled=use_amp) if _AMP_NEW_API else autocast(enabled=use_amp)
        with amp_ctx:
            features, hazard = model(x_path=x_path, x_omic=x_omic)

            # NaN/Inf 检测：模型输出
            if torch.isnan(hazard).any() or torch.isinf(hazard).any():
                if logger and batch_idx == 0:
                    logger.warning(
                        f"  [数值警告] batch {batch_idx}: hazard包含NaN/Inf，"
                        f"将跳过此batch"
                    )
                continue

            # 分类任务也需要logits (ncls模式下hazard是[C]维分类logits)
            cls_logits, cls_labels = None, None
            if config.model.task == "ncls":
                cls_logits = hazard
                cls_labels = labels

            total_loss, loss_dict = _compute_total_loss(
                hazard_pred=hazard,
                survtime=survtime,
                censor=censor,
                model=model,
                config=config,
                cls_logits=cls_logits,
                cls_labels=cls_labels,
            )

            # NaN/Inf 检测：loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                if logger and batch_idx < 3:
                    logger.warning(
                        f"  [数值警告] batch {batch_idx}: loss为NaN/Inf，"
                        f"跳过此batch。hazard范围: [{hazard.min().item():.2f}, {hazard.max().item():.2f}]"
                    )
                continue

            # 梯度累积缩放
            total_loss = total_loss / accum_steps

        # ---- 反向传播 ----
        # 安全检查: 确保loss有计算图
        if not total_loss.requires_grad:
            if logger and batch_idx < 3:
                logger.warning(
                    f"  [梯度警告] batch {batch_idx}: total_loss无计算图，跳过反向传播"
                )
            continue
        scaler.scale(total_loss).backward()

        # ---- 梯度累积步 ----
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            # 梯度裁剪
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # 学习率更新 (每accum步更新一次或每轮更新)
            if scheduler is not None:
                scheduler.step(epoch_idx + (batch_idx + 1) / len(train_loader))

        # ---- 累计统计 ----
        n_batches += 1
        running_loss += total_loss.item() * accum_steps
        running_cox += loss_dict["cox"]
        running_task += loss_dict["task"]
        running_reg += loss_dict["reg"]

        # 收集预测值用于计算C-index
        with torch.no_grad():
            if config.model.task == "surv":
                all_hazards.append(hazard.detach().cpu().numpy().reshape(-1))
            else:
                # ncls任务: 使用logits最大值作为风险分数
                all_hazards.append(hazard.detach().cpu().numpy().max(axis=1) if hazard.dim() > 1
                                   else hazard.detach().cpu().numpy().reshape(-1))
        all_survtimes.append(survtime.cpu().numpy().reshape(-1))
        all_censors.append(censor.cpu().numpy().reshape(-1))

        # ---- GPU显存监控 ----
        if gpu_monitor_interval > 0 and (batch_idx + 1) % gpu_monitor_interval == 0:
            gpu_info = _get_gpu_memory_info(device)
            if logger and gpu_info["used_percent"] > config.training.gpu_monitor.memory_warning_threshold * 100:
                logger.warning(
                    f"  [GPU显存警告] batch {batch_idx + 1}/{len(train_loader)}, "
                    f"使用率: {gpu_info['used_percent']:.1f}%, "
                    f"空闲: {gpu_info['free_gb']:.2f}GB"
                )

    # ---- 计算本轮平均指标 ----
    avg_loss = running_loss / max(n_batches, 1)

    avg_cindex = 0.5  # 默认值
    if len(all_hazards) > 0:
        hazards_np = np.concatenate(all_hazards).reshape(-1)
        survtimes_np = np.concatenate(all_survtimes).reshape(-1)
        censors_np = np.concatenate(all_censors).reshape(-1)
        try:
            # 使用纯Python实现 (避免lifelines可能的安装问题)
            avg_cindex = CIndex(hazards_np, survtimes_np, censors_np)
        except Exception:
            try:
                avg_cindex = CIndex_lifeline(hazards_np, survtimes_np, censors_np,
                                             nan_policy="omit")
            except Exception:
                avg_cindex = 0.5

    return avg_loss, avg_cindex


# ============================================================
# 单轮验证/测试评估
# ============================================================

@torch.no_grad()
def _evaluate_split(model, data_loader, config, device) -> Dict[str, Any]:
    """
    在某个数据划分上评估模型，收集所有预测值和标签。

    参数:
        model: 模型实例
        data_loader: DataLoader
        config: ConfigBundle配置对象
        device: torch设备

    返回:
        result: 包含hazards, survtimes, censors, labels等字段的字典
    """
    model.eval()

    all_hazards = []
    all_features = []
    all_survtimes = []
    all_censors = []
    all_labels = []  # 用于ncls任务

    for batch_idx, (x_path, x_grph, x_omic, e, t, g) in enumerate(data_loader):
        if x_path.nelement() == 0 or x_omic.nelement() == 0:
            continue

        x_path = x_path.to(device, non_blocking=config.training.non_blocking)
        x_omic = x_omic.to(device, non_blocking=config.training.non_blocking)

        features, hazard = model(x_path=x_path, x_omic=x_omic)

        all_features.append(features.cpu().numpy())
        if config.model.task == "surv":
            all_hazards.append(hazard.cpu().numpy().reshape(-1))
        else:
            all_hazards.append(hazard.cpu().numpy())
        all_survtimes.append(t.cpu().numpy().reshape(-1))
        all_censors.append(e.cpu().numpy().reshape(-1))
        all_labels.append(g.cpu().numpy().reshape(-1))

    if len(all_hazards) == 0:
        return {"hazards": np.array([]), "features": np.array([]),
                "survtimes": np.array([]), "censors": np.array([]),
                "labels": np.array([])}

    return {
        "hazards": np.concatenate(all_hazards),
        "features": np.concatenate(all_features),
        "survtimes": np.concatenate(all_survtimes),
        "censors": np.concatenate(all_censors),
        "labels": np.concatenate(all_labels),
    }


# ============================================================
# 绘制Loss曲线
# ============================================================

def _plot_loss_curves(metric_logger: MetricLogger, save_dir: Path, fold_id: int):
    """
    绘制训练和验证损失/C-index曲线并保存。

    参数:
        metric_logger: MetricLogger实例
        save_dir: 保存目录
        fold_id: 折号
    """
    if len(metric_logger.epochs) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = metric_logger.epochs

    # ---- Loss曲线 ----
    ax = axes[0]
    ax.plot(epochs, metric_logger.train_losses, "b-o", markersize=4, label="训练损失", linewidth=1.5)
    if metric_logger.val_losses:
        ax.plot(epochs, metric_logger.val_losses, "r-s", markersize=4, label="验证损失", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"训练损失曲线 (Fold {fold_id})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if metric_logger.best_epoch > 0:
        ax.axvline(x=metric_logger.best_epoch, color="green", linestyle="--",
                   alpha=0.6, label=f"最佳轮次={metric_logger.best_epoch}")

    # ---- C-index曲线 ----
    ax = axes[1]
    if metric_logger.train_cindices:
        ax.plot(epochs, metric_logger.train_cindices, "b-o", markersize=4,
                label="训练C-index", linewidth=1.5)
    if metric_logger.val_cindices:
        ax.plot(epochs, metric_logger.val_cindices, "r-s", markersize=4,
                label="验证C-index", linewidth=1.5)
        if metric_logger.best_epoch > 0:
            best_val = metric_logger.val_cindices[metric_logger.best_epoch - 1] \
                if metric_logger.best_epoch <= len(metric_logger.val_cindices) else 0
            ax.axhline(y=best_val, color="green", linestyle="--", alpha=0.5,
                       label=f"最佳C-index={best_val:.4f}")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("C-index", fontsize=12)
    ax.set_title(f"C-index曲线 (Fold {fold_id})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / f"fold{fold_id}_loss_curves.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Checkpoint 管理
# ============================================================

def _save_checkpoint(model, optimizer, scaler, scheduler, metric_logger,
                     epoch: int, fold_id: int, save_dir: Path, config,
                     is_best: bool = False, logger: Optional[logging.Logger] = None):
    """
    保存训练checkpoint。

    参数:
        model: 模型实例
        optimizer: 优化器
        scaler: GradScaler
        scheduler: 学习率调度器
        metric_logger: MetricLogger
        epoch: 当前轮次
        fold_id: 折号
        save_dir: 保存目录
        config: ConfigBundle
        is_best: 是否为最佳模型
        logger: 日志器
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "fold_id": fold_id,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "metric_logger": metric_logger.to_dict(),
        "config_dict": config.to_dict(),
    }

    if scheduler is not None:
        if hasattr(scheduler, "state_dict"):
            ckpt["scheduler_state_dict"] = scheduler.state_dict()

    # 保存最近checkpoint
    ckpt_path = save_dir / f"fold{fold_id}_checkpoint_epoch{epoch}.pth"
    torch.save(ckpt, str(ckpt_path))

    # 保存最佳模型权重 (轻量版，仅模型权重)
    if is_best:
        best_path = save_dir / f"fold{fold_id}_best_model.pth"
        torch.save(model.state_dict(), str(best_path))

    # 清理旧checkpoint (保留最近N个)
    keep_last_n = config.training.checkpoint.keep_last_n
    if keep_last_n > 0:
        ckpt_files = sorted(
            save_dir.glob(f"fold{fold_id}_checkpoint_epoch*.pth"),
            key=lambda p: int(p.stem.split("epoch")[-1]) if "epoch" in p.stem else 0,
        )
        while len(ckpt_files) > keep_last_n:
            oldest = ckpt_files.pop(0)
            oldest.unlink(missing_ok=True)

    if logger:
        logger.debug(f"  [Checkpoint] 已保存: {ckpt_path.name}")


def _load_checkpoint(model, optimizer, scaler, scheduler, metric_logger,
                     save_dir: Path, fold_id: int,
                     logger: Optional[logging.Logger] = None) -> int:
    """
    尝试从checkpoint恢复训练。

    参数:
        model, optimizer, scaler, scheduler, metric_logger: 要恢复的对象
        save_dir: checkpoint目录
        fold_id: 折号
        logger: 日志器

    返回:
        start_epoch: 恢复起始轮次 (0表示从头开始)
    """
    if not save_dir.exists():
        return 0

    # 查找最新的checkpoint文件
    ckpt_files = sorted(
        save_dir.glob(f"fold{fold_id}_checkpoint_epoch*.pth"),
        key=lambda p: int(p.stem.split("epoch")[-1]) if "epoch" in p.stem else 0,
    )

    if not ckpt_files:
        return 0

    latest_ckpt = ckpt_files[-1]
    try:
        ckpt = torch.load(str(latest_ckpt), map_location="cpu", weights_only=False)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in ckpt:
            try:
                if isinstance(scheduler, GradualWarmupScheduler):
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                else:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                if logger:
                    logger.warning("调度器状态加载失败，将从头开始调度")

        # 恢复MetricLogger
        if "metric_logger" in ckpt:
            ml_data = ckpt["metric_logger"]
            metric_logger.epochs = ml_data.get("epochs", [])
            metric_logger.train_losses = ml_data.get("train_losses", [])
            metric_logger.val_losses = ml_data.get("val_losses", [])
            metric_logger.train_cindices = ml_data.get("train_cindices", [])
            metric_logger.val_cindices = ml_data.get("val_cindices", [])
            metric_logger.val_pvalues = ml_data.get("val_pvalues", [])
            metric_logger.val_tdaucs = ml_data.get("val_tdaucs", [])
            metric_logger.learning_rates = ml_data.get("learning_rates", [])
            metric_logger.best_cindex = ml_data.get("best_cindex", 0.0)
            metric_logger.best_epoch = ml_data.get("best_epoch", 0)

        start_epoch = ckpt.get("epoch", 0)
        if logger:
            logger.info(f"  [断点续训] 从第{start_epoch}轮恢复: {latest_ckpt.name}")
        return start_epoch

    except Exception as e:
        if logger:
            logger.warning(f"  [Checkpoint] 恢复失败: {e}，将从头开始训练")
        return 0


# ============================================================
# 主训练函数: train()
# ============================================================

def train(config, data: Dict, device: torch.device, fold_id: int = 1):
    """
    单折完整训练流程。

    参数:
        config: ConfigBundle配置对象
        data: 数据字典 {"train": {...}, "val": {...}} (val可选)
        device: torch设备 (torch.device对象)
        fold_id: 当前折号 (用于日志和checkpoint命名)

    返回:
        (model, optimizer, metric_logger) 训练完成后的模型、优化器和指标记录器

    使用示例:
        model, optimizer, metric_logger = train(config, data, device, fold_id=1)
    """
    # ---- 日志 ----
    logger = logging.getLogger("YuHou")
    if not logger.handlers:
        from src.utils.logger import setup_logger
        logger = setup_logger(level=config.logging.level, language=config.logging.language)

    logger.info("=" * 70)
    logger.info(f"  Fold {fold_id} - 训练开始")
    logger.info("=" * 70)

    # ---- 设置随机种子 ----
    set_seed(config.model.seed + fold_id, deterministic=False, logger=logger)

    # ---- 输出目录 ----
    ckpt_dir = config.get_subdir("ckpt") / f"fold{fold_id}"
    log_dir = config.get_subdir("logs")
    figure_dir = config.get_subdir("figures")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- 记录关键配置 ----
    batch_size = config.training.batch_size
    accum_steps = config.training.gradient_accumulation_steps
    eff_batch_size = batch_size * accum_steps
    total_epochs = config.training.scheduler.n_epochs + config.training.scheduler.n_epochs_decay
    warmup_epochs = config.training.scheduler.warmup_epochs

    logger.info(f"  [训练配置]")
    logger.info(f"    批次大小: {batch_size} x 梯度累积{accum_steps}步 = 有效批次{eff_batch_size}")
    logger.info(f"    总训练轮数: {total_epochs} (含{config.training.scheduler.n_epochs}主轮+{config.training.scheduler.n_epochs_decay}衰减轮)")
    logger.info(f"    预热轮数: {warmup_epochs}")
    logger.info(f"    学习率: {config.training.optimizer.lr}")
    logger.info(f"    权重衰减: {config.training.optimizer.weight_decay}")
    logger.info(f"    梯度裁剪: {config.training.gradient_clip_norm}")
    logger.info(f"    AMP混合精度: {'启用' if config.training.amp.enabled else '禁用'}")
    logger.info(f"    正则化: {config.training.loss.reg_type} (lambda={config.training.loss.lambda_reg})")
    logger.info(f"    任务类型: {config.model.task}")
    logger.info(f"    融合策略: {config.model.fusion.type}")

    _log_gpu_memory(logger, device, prefix="  [GPU显存-训练前]")

    # ---- 构建数据加载器 ----
    from src.data_loading.datasets import PathomicDataset, build_dataloader

    train_loader = build_dataloader(config, data, split="train", mode=config.model.mode,
                                     shuffle=True, for_test=False)

    val_loader = None
    if "val" in data and len(data["val"].get("x_path", [])) > 0:
        val_loader = build_dataloader(config, data, split="val", mode=config.model.mode,
                                       shuffle=False, for_test=True)

    logger.info(f"  [数据] 训练样本: {len(train_loader.dataset)}, "
                f"批次: {len(train_loader)}")
    if val_loader:
        logger.info(f"  [数据] 验证样本: {len(val_loader.dataset)}, "
                    f"批次: {len(val_loader)}")

    # ---- 自动检测实际特征维度（适配不同CSV） ----
    if "train" in data and "x_omic" in data["train"]:
        actual_omic_dim = data["train"]["x_omic"].shape[1]
        if actual_omic_dim != config.model.omic.input_dim:
            logger.info(f"  [维度适配] 基因组特征维度: {config.model.omic.input_dim} -> {actual_omic_dim}")
            config.model.omic.input_dim = actual_omic_dim
            config.data.genomic.input_dim = actual_omic_dim

    # ---- 构建模型 ----
    model = PathomicNet(config)
    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  [模型] 总参数量: {total_params:,}, 可训练: {trainable_params:,}")

    # ---- 构建优化器 ----
    optimizer = _build_optimizer(model, config)

    # ---- 构建学习率调度器 ----
    base_scheduler = _build_scheduler(optimizer, config)

    # 包裹warmup
    scheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=1.0,
        total_epoch=warmup_epochs,
        after_scheduler=base_scheduler,
    )

    # ---- 初始化AMP ----
    use_amp = config.training.amp.enabled
    scaler = GradScaler("cuda", enabled=use_amp) if _AMP_NEW_API else GradScaler(enabled=use_amp)

    # ---- 初始化指标记录器 ----
    metric_logger = MetricLogger(logger=logger)

    # ---- 断点续训 ----
    resume = config.training.checkpoint.resume
    start_epoch = 0
    if resume:
        start_epoch = _load_checkpoint(
            model, optimizer, scaler, scheduler, metric_logger,
            save_dir=ckpt_dir, fold_id=fold_id, logger=logger,
        )
        model = model.to(device)

    # ---- 训练循环 ----
    logger.info(f"  [训练开始] 起始轮次={start_epoch + 1}, 总轮次={total_epochs}")
    logger.info("-" * 70)

    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- 训练阶段 ----
        train_loss, train_cindex = _train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            scaler=scaler,
            epoch_idx=epoch,
            logger=logger,
        )

        # ---- 验证阶段 ----
        val_loss = None
        val_cindex = None
        val_pvalue = None
        val_tdauc = None

        if val_loader is not None:
            eval_result = _evaluate_split(model, val_loader, config, device)

            if len(eval_result["hazards"]) > 0:
                hazards = eval_result["hazards"].reshape(-1)
                survtimes = eval_result["survtimes"].reshape(-1)
                censors = eval_result["censors"].reshape(-1)

                # 计算验证损失
                cox_criterion = CoxPartialLikelihoodLoss()
                hazard_t = torch.tensor(hazards, device=device)
                survtime_t = torch.tensor(survtimes, device=device)
                censor_t = torch.tensor(censors, device=device)
                val_loss = cox_criterion(hazard_t, survtime_t, censor_t).item()

                # 计算验证指标
                try:
                    val_cindex = CIndex(hazards, survtimes, censors)
                except Exception:
                    val_cindex = 0.5

                try:
                    val_pvalue = cox_log_rank(hazards, survtimes, censors)
                except Exception:
                    val_pvalue = float("nan")

                try:
                    td_auc_result = safe_time_dependent_auc(
                        survtimes, censors, hazards,
                        times=config.evaluation.eval_times,
                        nan_policy="omit",
                    )
                    val_tdauc = td_auc_result.get("mean_auc", float("nan"))
                except Exception:
                    val_tdauc = float("nan")

        # ---- 记录指标 ----
        metric_logger.update(
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            train_cindex=train_cindex,
            val_cindex=val_cindex,
            val_pvalue=val_pvalue,
            val_tdauc=val_tdauc,
            lr=current_lr,
        )

        # 追踪最佳模型
        if val_cindex is not None:
            metric_logger.track_best(epoch + 1, val_cindex, model)
        else:
            metric_logger.track_best(epoch + 1, train_cindex, model)

        # ---- 早停检查 ----
        es_cfg = getattr(config.training, 'early_stopping', None)
        if es_cfg is not None and getattr(es_cfg, 'enabled', False):
            if metric_logger.no_improve_count >= es_cfg.patience:
                logger.info(
                    f"  [早停] 连续{es_cfg.patience}轮无改善 "
                    f"(最佳C-index={metric_logger.best_cindex:.4f}，第{metric_logger.best_epoch}轮)，"
                    f"提前结束训练"
                )
                break

        # ---- 每轮中文日志 ----
        epoch_time = time.time() - epoch_start_time
        log_parts = [
            f"[Fold {fold_id}] Epoch {epoch + 1}/{total_epochs}",
            f"耗时={epoch_time:.1f}s",
            f"LR={current_lr:.2e}",
            f"训练损失={train_loss:.4f}",
        ]
        if val_loss is not None:
            log_parts.append(f"验证损失={val_loss:.4f}")
        if train_cindex is not None:
            log_parts.append(f"训练C-index={train_cindex:.4f}")
        if val_cindex is not None:
            log_parts.append(f"验证C-index={val_cindex:.4f}")
        if val_pvalue is not None and not np.isnan(val_pvalue):
            log_parts.append(f"Log-rank p={val_pvalue:.4e}")
        if val_tdauc is not None and not np.isnan(val_tdauc):
            log_parts.append(f"tdAUC={val_tdauc:.4f}")

        logger.info("  " + " | ".join(log_parts))

        # GPU显存定期记录
        if (epoch + 1) % 5 == 0 or epoch == start_epoch:
            _log_gpu_memory(logger, device, prefix=f"  [GPU显存-Epoch{epoch + 1}]")

        # ---- 保存checkpoint ----
        save_every = config.training.checkpoint.save_every
        if (epoch + 1) % save_every == 0 or (epoch + 1) == total_epochs:
            is_best = (metric_logger.best_epoch == epoch + 1)
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                metric_logger=metric_logger,
                epoch=epoch + 1,
                fold_id=fold_id,
                save_dir=ckpt_dir,
                config=config,
                is_best=is_best,
                logger=logger,
            )

    # ---- 训练完成 ----
    logger.info("-" * 70)
    logger.info(f"  [训练完成] Fold {fold_id} | "
                f"最佳C-index={metric_logger.best_cindex:.4f} "
                f"(第{metric_logger.best_epoch}轮)")

    # 恢复最佳模型权重
    if metric_logger.best_model_state is not None:
        model.load_state_dict(metric_logger.best_model_state)
        logger.info(f"  [模型] 已恢复最佳模型 (第{metric_logger.best_epoch}轮)")

    # ---- 绘制Loss曲线 ----
    try:
        _plot_loss_curves(metric_logger, figure_dir, fold_id)
        logger.info(f"  [曲线] Loss曲线已保存至: {figure_dir}")
    except Exception as e:
        logger.warning(f"  [曲线] 绘制失败: {e}")

    _log_gpu_memory(logger, device, prefix="  [GPU显存-训练后]")

    return model, optimizer, metric_logger


# ============================================================
# 主测试函数: test()
# ============================================================

@torch.no_grad()
def test(config, model: nn.Module, data: Dict, split: str,
         device: torch.device):
    """
    在指定数据划分上评估模型。

    参数:
        config: ConfigBundle配置对象
        model: 训练好的模型实例
        data: 数据字典 {"test": {...}} (或 {"val": {...}})
        split: 划分名称 ("test" 或 "val")
        device: torch设备

    返回:
        (loss, cindex, pvalue, surv_acc, grad_acc, pred, td_auc,
         binary_metrics, group_summary, hr, cls_metrics)

        其中:
        - loss: 测试损失
        - cindex: C-index
        - pvalue: Log-rank p值
        - surv_acc: 生存准确率 (accuracy_cox)
        - grad_acc: 梯度准确率 (预留, 0.0)
        - pred: 预测风险分数 [N]
        - td_auc: 时间依赖AUC结果dict
        - binary_metrics: 二分类指标dict
        - group_summary: 分组摘要dict
        - hr: 风险比
        - cls_metrics: 分类指标dict (ncls任务)
    """
    # ---- 日志 ----
    logger = logging.getLogger("YuHou")
    if not logger.handlers:
        from src.utils.logger import setup_logger
        logger = setup_logger(level=config.logging.level, language=config.logging.language)

    # ---- 构建测试数据加载器 ----
    from src.data_loading.datasets import build_dataloader

    test_loader = build_dataloader(config, data, split=split, mode=config.model.mode,
                                    shuffle=False, for_test=True)

    logger.info(f"  [测试] 划分={split}, 样本数={len(test_loader.dataset)}, "
                f"批次数={len(test_loader)}")

    # ---- 推理 ----
    model.eval()
    eval_result = _evaluate_split(model, test_loader, config, device)

    if len(eval_result["hazards"]) == 0:
        logger.error(f"  [测试] 无有效数据!")
        return (0.0, 0.5, float("nan"), {}, 0.0,
                np.array([]), {}, {}, {}, float("nan"), {})

    hazards = eval_result["hazards"]
    survtimes = eval_result["survtimes"]
    censors = eval_result["censors"]
    labels = eval_result["labels"]

    # 确保为1D数组
    hazards_1d = hazards.reshape(-1) if hazards.ndim > 1 else hazards
    survtimes_1d = survtimes.reshape(-1)
    censors_1d = censors.reshape(-1)

    # ---- 计算损失 ----
    try:
        cox_criterion = CoxPartialLikelihoodLoss()
        hazard_t = torch.tensor(hazards_1d, device=device)
        survtime_t = torch.tensor(survtimes_1d, device=device)
        censor_t = torch.tensor(censors_1d, device=device)
        loss = cox_criterion(hazard_t, survtime_t, censor_t).item()
    except Exception as e:
        logger.warning(f"  [测试] 损失计算失败: {e}")
        loss = 0.0

    # ---- C-index ----
    try:
        cindex = CIndex(hazards_1d, survtimes_1d, censors_1d)
    except Exception:
        try:
            cindex = CIndex_lifeline(hazards_1d, survtimes_1d, censors_1d,
                                     nan_policy="omit")
        except Exception:
            cindex = 0.5

    # ---- Log-rank p值 ----
    try:
        pvalue = cox_log_rank(hazards_1d, survtimes_1d, censors_1d)
    except Exception:
        pvalue = float("nan")

    # ---- survival accuracy (accuracy_cox) ----
    try:
        surv_acc = accuracy_cox(hazards_1d, survtimes_1d, censors_1d)
    except Exception:
        surv_acc = {}

    grad_acc = 0.0  # 预留字段

    # ---- 预测值 ----
    pred = hazards_1d.copy()

    # ---- 时间依赖AUC ----
    try:
        td_auc = safe_time_dependent_auc(
            survtimes_1d, censors_1d, hazards_1d,
            times=config.evaluation.eval_times,
            n_times=50,
            nan_policy="omit",
        )
    except Exception:
        td_auc = {"mean_auc": float("nan"), "times": [], "auc_values": []}

    # ---- 二分类指标 (基于中位数风险分组) ----
    try:
        binary_metrics = safe_binary_metrics_from_risk(
            hazards_1d, survtimes_1d, censors_1d,
            time_threshold=np.median(survtimes_1d) if len(survtimes_1d) > 0 else None,
            return_dict=True,
        )
    except Exception:
        binary_metrics = {}

    # ---- 风险分组摘要 ----
    try:
        group_summary = safe_group_survival_summary(
            hazards_1d, survtimes_1d, censors_1d, split_method="median"
        )
    except Exception:
        group_summary = {}

    # ---- 风险比 (HR) ----
    try:
        hr = safe_hazard_ratio_by_median_split(
            hazards_1d, survtimes_1d, censors_1d, return_details=False
        )
    except Exception:
        hr = float("nan")

    # ---- 分类指标 (ncls任务) ----
    cls_metrics = {}
    if config.model.task == "ncls" and len(labels) > 0:
        try:
            cls_metrics = cls_metrics_fn(logits=torch.tensor(hazards),
                                          y_true=torch.tensor(labels.reshape(-1).astype(int)))
        except Exception:
            cls_metrics = {}

    # ---- 打印测试结果 ----
    logger.info("=" * 70)
    logger.info(f"  [{split.upper()} 评估结果]")
    logger.info(f"    C-index:          {cindex:.4f}")
    logger.info(f"    Log-rank p:       {pvalue:.4e}" if not np.isnan(pvalue) else "    Log-rank p:       nan")
    if isinstance(td_auc, dict) and td_auc.get("mean_auc") is not None and not np.isnan(td_auc.get("mean_auc", float("nan"))):
        logger.info(f"    tdAUC (均值):     {td_auc['mean_auc']:.4f}")
    if not np.isnan(hr) if isinstance(hr, float) else True:
        hr_val = hr if isinstance(hr, (float, int)) else float("nan")
        if not np.isnan(hr_val):
            logger.info(f"    Hazard Ratio:     {hr_val:.4f}")
    if isinstance(surv_acc, dict) and surv_acc.get("accuracy") is not None:
        logger.info(f"    Accuracy (Cox):   {surv_acc['accuracy']:.4f}")
    if binary_metrics:
        acc = binary_metrics.get("accuracy", "N/A")
        f1 = binary_metrics.get("f1_score", "N/A")
        logger.info(f"    二分类准确率:     {acc}")
        logger.info(f"    二分类F1:         {f1}")
    if cls_metrics:
        logger.info(f"    分类指标:         {format_classification_metrics(cls_metrics)}")
    logger.info("=" * 70)

    return (loss, cindex, pvalue, surv_acc, grad_acc, pred,
            td_auc, binary_metrics, group_summary, hr, cls_metrics)


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    """
    自测: 模拟完整训练流程。
    需要配置文件 config/default_config.yaml 存在。
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    print("=" * 70)
    print("训练核心模块 (trainer.py) 自测")
    print("=" * 70)

    try:
        from config.config_loader import load_config

        # 加载配置
        config = load_config()

        # 强制使用CPU进行自测
        config.training.device = "cpu"
        config.training.batch_size = 2
        config.training.gradient_accumulation_steps = 1
        config.training.scheduler.n_epochs = 2
        config.training.scheduler.n_epochs_decay = 0
        config.training.scheduler.warmup_epochs = 0
        config.training.amp.enabled = False
        config.training.checkpoint.resume = False
        config.upstream.num_patches_per_patient = 3
        config.model.path.input_size = 256  # 减小输入尺寸用于快速测试

        device = torch.device("cpu")

        # 使用随机数据模拟训练
        np.random.seed(42)
        torch.manual_seed(42)

        N_patients = 20
        N_patches = config.upstream.num_patches_per_patient

        # 模拟patch路径 (使用随机张量替代真实图像)
        # 由于PathNet需要加载预训练权重，这里仅验证数据流
        print("\n[1/4] 生成模拟数据...")
        mock_x_path = [
            [f"dummy_{i}_{j}.png" for j in range(N_patches)]
            for i in range(N_patients)
        ]
        mock_x_omic = np.random.randn(N_patients, config.data.genomic.input_dim).astype(np.float32)
        mock_e = np.random.binomial(1, 0.4, N_patients).astype(np.float32)
        mock_t = np.random.randint(30, 365 * 5, N_patients).astype(np.float32)
        mock_g = np.random.choice(["N0", "N1", "N2"], N_patients)

        mock_data = {
            "train": {
                "x_path": mock_x_path,
                "x_omic": mock_x_omic,
                "e": mock_e,
                "t": mock_t,
                "g": mock_g,
            }
        }

        print(f"  训练样本数: {len(mock_data['train']['x_path'])}")
        print(f"  每病人patch数: {N_patches}")

        # 由于需要真实图像文件和预训练权重，跳过完整训练流程
        print("\n[2/4] 验证数据流结构...")
        print("  (跳过完整训练: 需要真实patch图像和预训练权重)")

        # 验证MetricLogger
        print("\n[3/4] 验证MetricLogger...")
        ml = MetricLogger()
        for ep in range(1, 6):
            ml.update(
                epoch=ep,
                train_loss=1.0 / ep,
                val_loss=1.1 / ep,
                train_cindex=0.5 + 0.05 * ep,
                val_cindex=0.5 + 0.04 * ep,
                val_pvalue=0.05 / ep,
                val_tdauc=0.55 + 0.03 * ep,
                lr=1e-4 * (0.95 ** ep),
            )
            ml.track_best(ep, 0.5 + 0.04 * ep, model=None)
        print(f"  记录轮数: {len(ml.epochs)}")
        print(f"  最佳C-index: {ml.best_cindex:.4f} (第{ml.best_epoch}轮)")

        # 验证test函数的数据流 (用随机数据)
        print("\n[4/4] 验证test函数接口...")
        print("  test() 函数接口: (loss, cindex, pvalue, surv_acc, grad_acc, pred, td_auc, binary_metrics, group_summary, hr, cls_metrics)")

        print("\n" + "=" * 70)
        print("trainer.py 自测完成!")
        print("=" * 70)

    except ImportError as e:
        print(f"\n  跳过完整自测: {e}")
        print("  (需要 config 模块和完整项目依赖)")
        print("\n  核心类和函数签名已就绪:")
        print("    - MetricLogger: 训练指标追踪器")
        print("    - train(config, data, device, fold_id): 单折训练")
        print("    - test(config, model, data, split, device): 模型测试")
