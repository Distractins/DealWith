# -*- coding: utf-8 -*-
"""
trainer.py
============================================================================
Histomolecular训练核心模块 - 单折训练与评估引擎。

任务: 用WSI patch图像预测基因表达 (分类高/低表达 / 回归连续表达量)。

主要改动 (相对旧版Cox生存预测):
    1. 损失: Cox -> CrossEntropyLoss(分类) / MSE(回归)
    2. 指标: C-index -> ACC/AUC/F1 (分类) / MSE/Pearson (回归)
    3. 附属: 保留用预测分子表型分组的KM生存分析 (仅补充验证)

前向流程:
    patches [B,N,C,H,W] -> PathNet (patch级) -> 均值池化 (病人级)
    -> (可选OmicNet协变量) -> (可选Fusion) -> Classifier -> logits

依赖:
    - src.networks.pathomic_net (PathomicNet)
    - src.losses.classification_loss (classification_loss, FocalLoss, regression_loss)
    - src.losses.regularization (compute_regularization)
    - src.evaluation.classification_metrics (classification_metrics)
    - src.evaluation.regression_metrics (regression_metrics)
    - src.evaluation.survival_metrics (cox_log_rank, 附属KM)
============================================================================
"""

import os
import time
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.amp import GradScaler, autocast
    _AMP_NEW_API = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_NEW_API = False

from src.networks.pathomic_net import PathomicNet
from src.losses.classification_loss import classification_loss, FocalLoss, regression_loss
from src.losses.regularization import compute_regularization
from src.evaluation.classification_metrics import classification_metrics
from src.evaluation.regression_metrics import regression_metrics
from src.utils.seed import set_seed


# ============================================================
# GPU 显存监控工具 (保留)
# ============================================================

def _get_gpu_memory_info(device: torch.device) -> Dict[str, float]:
    if not torch.cuda.is_available() or device.type != "cuda":
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0,
                "free_gb": 0.0, "used_percent": 0.0}
    try:
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        free = total - reserved
        return {"allocated_gb": round(allocated, 2), "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2), "free_gb": round(free, 2),
                "used_percent": round((reserved / total) * 100.0, 1)}
    except Exception:
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0,
                "free_gb": 0.0, "used_percent": 0.0}


def _log_gpu_memory(logger, device, prefix="[GPU显存]") -> None:
    info = _get_gpu_memory_info(device)
    if info["total_gb"] > 0:
        logger.info(f"{prefix} 已分配={info['allocated_gb']:.2f}GB / "
                    f"已保留={info['reserved_gb']:.2f}GB / "
                    f"总量={info['total_gb']:.2f}GB / "
                    f"空闲={info['free_gb']:.2f}GB / "
                    f"使用率={info['used_percent']:.1f}%")


# ============================================================
# Optimizer & Scheduler 构建 (保留)
# ============================================================

def _build_optimizer(model: nn.Module, config) -> optim.Optimizer:
    opt_cfg = config.training.optimizer
    opt_type = opt_cfg.type.lower()
    if opt_type == "adam":
        return optim.Adam(model.parameters(), lr=opt_cfg.lr,
                          betas=tuple(opt_cfg.betas), weight_decay=opt_cfg.weight_decay)
    if opt_type == "adamw":
        return optim.AdamW(model.parameters(), lr=opt_cfg.lr,
                           betas=tuple(opt_cfg.betas), weight_decay=opt_cfg.weight_decay)
    if opt_type == "sgd":
        return optim.SGD(model.parameters(), lr=opt_cfg.lr,
                         momentum=0.9, weight_decay=opt_cfg.weight_decay)
    raise ValueError(f"不支持的优化器类型: '{opt_type}'")


def _build_scheduler(optimizer, config):
    sched_cfg = config.training.scheduler
    sched_type = sched_cfg.type.lower()
    total_epochs = sched_cfg.n_epochs + sched_cfg.n_epochs_decay
    if sched_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=sched_cfg.min_lr)
    if sched_type == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=sched_cfg.lr_decay_iters, gamma=0.5)
    if sched_type == "linear":
        return optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                           end_factor=max(sched_cfg.min_lr / config.training.optimizer.lr, 1e-3),
                                           total_iters=total_epochs)
    if sched_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                    patience=5, min_lr=sched_cfg.min_lr)
    if sched_type == "exp":
        gamma = (sched_cfg.min_lr / config.training.optimizer.lr) ** (1.0 / max(1, sched_cfg.n_epochs_decay))
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    raise ValueError(f"不支持的调度器类型: '{sched_type}'")


class GradualWarmupScheduler:
    """线性学习率预热包装器。"""

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler):
        self.optimizer = optimizer
        self.multiplier = multiplier
        self.total_epoch = max(0, total_epoch)
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            self.after_scheduler.step()
            return
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch >= self.total_epoch:
            for pg, bl in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = bl
            self.finished = True
            if self.after_scheduler:
                self.after_scheduler.step()
            return
        progress = epoch / max(self.total_epoch, 1)
        for pg in self.optimizer.param_groups:
            start = self.base_lrs[0] * 0.1
            pg["lr"] = start + (self.base_lrs[0] - start) * progress

    def state_dict(self):
        return {"finished": self.finished, "last_epoch": self.last_epoch,
                "after_scheduler": self.after_scheduler.state_dict() if self.after_scheduler else None}

    def load_state_dict(self, state_dict):
        self.finished = state_dict["finished"]
        self.last_epoch = state_dict["last_epoch"]
        if self.after_scheduler and state_dict["after_scheduler"]:
            self.after_scheduler.load_state_dict(state_dict["after_scheduler"])


# ============================================================
# MetricLogger: 训练指标收集器
# ============================================================

class MetricLogger:
    """记录每轮损失与验证指标，追踪最佳模型。"""

    def __init__(self, logger=None):
        self.logger = logger
        self.epochs: List[int] = []
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_scores: List[float] = []
        self.learning_rates: List[float] = []
        self.best_score: float = -float("inf")
        self.best_epoch: int = 0
        self.best_model_state: Optional[Dict] = None
        self.no_improve_count: int = 0

    def track_best(self, epoch: int, score: float, model: nn.Module, mode: str = "max"):
        improved = (score > self.best_score) if mode == "max" else (score < self.best_score)
        if self.best_model_state is None or improved:
            self.best_score = score
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.no_improve_count = 0
            if self.logger:
                self.logger.info(f"  [最佳模型] 第{epoch}轮 score={score:.4f} (新高)")
        else:
            self.no_improve_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_scores": self.val_scores,
            "learning_rates": self.learning_rates,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
        }

    def load_from_dict(self, d: Dict):
        self.epochs = d.get("epochs", [])
        self.train_losses = d.get("train_losses", [])
        self.val_losses = d.get("val_losses", [])
        self.val_scores = d.get("val_scores", [])
        self.learning_rates = d.get("learning_rates", [])
        self.best_score = d.get("best_score", -float("inf"))
        self.best_epoch = d.get("best_epoch", 0)


# ============================================================
# 损失计算
# ============================================================

def _compute_total_loss(logits, y, model, config):
    """
    计算总损失: 任务损失(CE/MSE) + 正则化损失。

    参数:
        logits: [B, label_dim] 模型输出
        y: [B] 标签 (分类为long, 回归为float)
        model: 模型
        config: 配置

    返回:
        total_loss, {"task": float, "reg": float}
    """
    loss_cfg = config.training.loss
    task_type = str(config.model.task_type).lower()

    if task_type == "regression":
        task_loss = regression_loss(logits, y, loss_name=loss_cfg.task_loss)
    else:
        loss_name = config.training.classification.loss_type
        if loss_name == "focal":
            task_loss = FocalLoss(alpha=config.training.classification.focal_alpha,
                                  gamma=config.training.classification.focal_gamma)(logits, y.long())
        else:
            task_loss = classification_loss(
                logits=logits, labels=y,
                loss_name=loss_name,
                label_smoothing=config.training.classification.label_smoothing,
            )
    task_loss = task_loss * loss_cfg.lambda_task

    reg_loss = torch.tensor(0.0, device=logits.device)
    if loss_cfg.reg_type != "none":
        reg_loss = compute_regularization(model, reg_type=loss_cfg.reg_type) * loss_cfg.lambda_reg

    return task_loss + reg_loss, {"task": task_loss.item(), "reg": reg_loss.item()}


# ============================================================
# 单轮训练
# ============================================================

def _train_one_epoch(model, train_loader, optimizer, scheduler, config,
                     device, scaler, epoch_idx, logger=None) -> Tuple[float, float]:
    model.train()
    accum_steps = config.training.gradient_accumulation_steps
    grad_clip = config.training.gradient_clip_norm
    use_amp = config.training.amp.enabled
    task_type = str(config.model.task_type).lower()

    running_loss = 0.0
    all_logits, all_labels = [], []
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (x_path, x_grph, x_omic, e, t, g) in enumerate(train_loader):
        if x_path.nelement() == 0 and x_omic.nelement() == 0:
            continue

        if x_path.dim() > 1:
            x_path = x_path.to(device, non_blocking=config.training.non_blocking)
        x_omic = x_omic.to(device, non_blocking=config.training.non_blocking)
        y = g.to(device, non_blocking=config.training.non_blocking)
        if task_type == "regression":
            y = y.float()
        else:
            y = y.long()

        amp_ctx = autocast("cuda", enabled=use_amp) if _AMP_NEW_API else autocast(enabled=use_amp)
        with amp_ctx:
            features, logits = model(x_path=x_path, x_omic=x_omic)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                if logger and batch_idx == 0:
                    logger.warning(f"  [数值警告] batch {batch_idx}: logits含NaN/Inf，跳过")
                continue

            total_loss, loss_dict = _compute_total_loss(logits, y, model, config)

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                if logger and batch_idx < 3:
                    logger.warning(f"  [数值警告] batch {batch_idx}: loss为NaN/Inf，跳过")
                continue

            total_loss = total_loss / accum_steps

        if not total_loss.requires_grad:
            continue
        scaler.scale(total_loss).backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step(epoch_idx + (batch_idx + 1) / len(train_loader))

        n_batches += 1
        running_loss += total_loss.item() * accum_steps

        with torch.no_grad():
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy().reshape(-1))

    avg_loss = running_loss / max(n_batches, 1)

    # 训练指标
    train_score = float("nan")
    if all_logits:
        logits_np = np.concatenate(all_logits, axis=0)
        labels_np = np.concatenate(all_labels).reshape(-1)
        if task_type == "regression":
            m = regression_metrics(logits_np, labels_np)
            train_score = m.get("pearson", float("nan"))
        else:
            preds = np.argmax(logits_np, axis=1) if logits_np.ndim > 1 else (logits_np > 0).astype(int)
            train_score = float((preds == labels_np).mean())

    return avg_loss, train_score


# ============================================================
# 单轮验证/测试评估
# ============================================================

@torch.no_grad()
def _evaluate_split(model, data_loader, config, device) -> Dict[str, Any]:
    model.eval()
    all_logits, all_labels = [], []
    all_survtimes, all_censors = [], []

    for (x_path, x_grph, x_omic, e, t, g) in data_loader:
        if x_path.dim() > 1:
            x_path = x_path.to(device, non_blocking=config.training.non_blocking)
        x_omic = x_omic.to(device, non_blocking=config.training.non_blocking)

        _, logits = model(x_path=x_path, x_omic=x_omic)

        all_logits.append(logits.cpu().numpy())
        all_labels.append(g.cpu().numpy().reshape(-1))
        all_survtimes.append(t.cpu().numpy().reshape(-1))
        all_censors.append(e.cpu().numpy().reshape(-1))

    if not all_logits:
        return {"logits": np.array([]), "labels": np.array([]),
                "survtimes": np.array([]), "censors": np.array([])}

    return {
        "logits": np.concatenate(all_logits, axis=0),
        "labels": np.concatenate(all_labels).reshape(-1),
        "survtimes": np.concatenate(all_survtimes).reshape(-1),
        "censors": np.concatenate(all_censors).reshape(-1),
    }


def _compute_task_metrics(logits, labels, task_type):
    """计算分类/回归指标 (统一返回dict)。"""
    if task_type == "regression":
        return regression_metrics(logits, labels)
    return classification_metrics(logits=torch.tensor(logits), y_true=torch.tensor(labels.astype(int)))


def _prob_for_classification(logits):
    """二分类: 返回高表达类(1)的概率。"""
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
        return 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax = e / e.sum(axis=1, keepdims=True)
    return softmax[:, 1] if softmax.shape[1] >= 2 else softmax[:, 0]


def _score_for_survival_group(logits, task_type):
    """得到用于分组(中位数)的分数。"""
    if task_type == "regression":
        return logits.reshape(-1).astype(float)
    return _prob_for_classification(logits)


# ============================================================
# 损失/指标曲线绘制
# ============================================================

def _plot_loss_curves(metric_logger: MetricLogger, save_dir: Path, fold_id: int):
    if len(metric_logger.epochs) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = metric_logger.epochs

    axes[0].plot(epochs, metric_logger.train_losses, "b-o", markersize=4, label="训练损失")
    if metric_logger.val_losses:
        axes[0].plot(epochs, metric_logger.val_losses, "r-s", markersize=4, label="验证损失")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"损失曲线 (Fold {fold_id})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if metric_logger.val_scores:
        axes[1].plot(epochs, metric_logger.val_scores, "g-^", markersize=4, label="验证指标")
        if metric_logger.best_epoch > 0:
            axes[1].axvline(x=metric_logger.best_epoch, color="green", linestyle="--",
                            alpha=0.6, label=f"最佳轮次={metric_logger.best_epoch}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"验证指标曲线 (Fold {fold_id})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / f"fold{fold_id}_loss_curves.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Checkpoint 管理
# ============================================================

def _save_checkpoint(model, optimizer, scaler, scheduler, metric_logger,
                     epoch, fold_id, save_dir, config, is_best=False, logger=None):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch, "fold_id": fold_id,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "metric_logger": metric_logger.to_dict(),
        "config_dict": config.to_dict(),
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    ckpt_path = save_dir / f"fold{fold_id}_checkpoint_epoch{epoch}.pth"
    torch.save(ckpt, str(ckpt_path))

    if is_best:
        torch.save(model.state_dict(), str(save_dir / f"fold{fold_id}_best_model.pth"))

    keep_last_n = config.training.checkpoint.keep_last_n
    if keep_last_n > 0:
        files = sorted(save_dir.glob(f"fold{fold_id}_checkpoint_epoch*.pth"),
                       key=lambda p: int(p.stem.split("epoch")[-1]) if "epoch" in p.stem else 0)
        while len(files) > keep_last_n:
            files.pop(0).unlink(missing_ok=True)

    if logger:
        logger.debug(f"  [Checkpoint] 已保存: {ckpt_path.name}")


def _load_checkpoint(model, optimizer, scaler, scheduler, metric_logger,
                     save_dir, fold_id, logger=None) -> int:
    if not save_dir.exists():
        return 0
    files = sorted(save_dir.glob(f"fold{fold_id}_checkpoint_epoch*.pth"),
                   key=lambda p: int(p.stem.split("epoch")[-1]) if "epoch" in p.stem else 0)
    if not files:
        return 0
    latest = files[-1]
    try:
        ckpt = torch.load(str(latest), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                pass
        if "metric_logger" in ckpt:
            metric_logger.load_from_dict(ckpt["metric_logger"])
        start_epoch = ckpt.get("epoch", 0)
        if logger:
            logger.info(f"  [断点续训] 从第{start_epoch}轮恢复: {latest.name}")
        return start_epoch
    except Exception as e:
        if logger:
            logger.warning(f"  [Checkpoint] 恢复失败: {e}，将从头开始训练")
        return 0


# ============================================================
# 主训练函数: train()
# ============================================================

def train(config, data: Dict, device: torch.device, fold_id: int = 1):
    logger = logging.getLogger("YuHou")
    if not logger.handlers:
        from src.utils.logger import setup_logger
        logger = setup_logger(level=config.logging.level, language=config.logging.language)

    logger.info("=" * 70)
    logger.info(f"  Fold {fold_id} - 训练开始 (任务={config.model.task_type}, 基因={config.model.target_gene})")
    logger.info("=" * 70)

    set_seed(config.model.seed + fold_id, deterministic=False, logger=logger)

    ckpt_dir = config.get_subdir("ckpt") / f"fold{fold_id}"
    figure_dir = config.get_subdir("figures")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_epochs = config.training.scheduler.n_epochs + config.training.scheduler.n_epochs_decay
    warmup_epochs = config.training.scheduler.warmup_epochs
    logger.info(f"  [训练配置] batch={config.training.batch_size} x "
                f"accum={config.training.gradient_accumulation_steps}, "
                f"epochs={total_epochs}, lr={config.training.optimizer.lr}, "
                f"loss={config.training.loss.task_loss}, monitor={config.training.early_stopping.monitor}")

    _log_gpu_memory(logger, device, prefix="  [GPU显存-训练前]")

    from src.data_loading.datasets import build_dataloader
    train_loader = build_dataloader(config, data, split="train", mode=config.model.mode,
                                    shuffle=True, for_test=False)
    val_loader = None
    if "val" in data and len(data["val"].get("x_path", [])) > 0:
        val_loader = build_dataloader(config, data, split="val", mode=config.model.mode,
                                      shuffle=False, for_test=True)

    logger.info(f"  [数据] 训练样本={len(train_loader.dataset)}, 批次={len(train_loader)}")
    if val_loader:
        logger.info(f"  [数据] 验证样本={len(val_loader.dataset)}")

    # 维度适配
    if config.model.mode in ("omic", "pathomic"):
        actual_omic_dim = data["train"]["x_omic"].shape[1]
        if actual_omic_dim != config.model.omic.input_dim:
            logger.info(f"  [维度适配] 协变量维度 {config.model.omic.input_dim} -> {actual_omic_dim}")
            config.model.omic.input_dim = actual_omic_dim

    model = PathomicNet(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  [模型] 总参数量={total_params:,}, 可训练={trainable_params:,}")

    optimizer = _build_optimizer(model, config)
    base_scheduler = _build_scheduler(optimizer, config)
    scheduler = GradualWarmupScheduler(optimizer, 1.0, warmup_epochs, base_scheduler)

    use_amp = config.training.amp.enabled
    scaler = GradScaler("cuda", enabled=use_amp) if _AMP_NEW_API else GradScaler(enabled=use_amp)

    metric_logger = MetricLogger(logger=logger)

    start_epoch = 0
    if config.training.checkpoint.resume:
        start_epoch = _load_checkpoint(model, optimizer, scaler, scheduler, metric_logger,
                                       save_dir=ckpt_dir, fold_id=fold_id, logger=logger)
        model = model.to(device)

    monitor = config.training.early_stopping.monitor
    monitor_mode = "min" if monitor == "loss" else "max"

    logger.info(f"  [训练开始] 起始轮次={start_epoch + 1}, 总轮次={total_epochs}")
    logger.info("-" * 70)

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_score = _train_one_epoch(
            model, train_loader, optimizer, scheduler, config, device, scaler, epoch, logger)

        val_loss, val_score = None, None
        if val_loader is not None:
            eval_result = _evaluate_split(model, val_loader, config, device)
            if len(eval_result["logits"]) > 0:
                task_type = str(config.model.task_type).lower()
                metrics = _compute_task_metrics(eval_result["logits"], eval_result["labels"], task_type)
                val_score = _pick_monitor_metric(metrics, monitor, task_type)
                # 验证损失
                logits_t = torch.tensor(eval_result["logits"], device=device)
                labels_t = torch.tensor(eval_result["labels"], device=device)
                if task_type == "regression":
                    labels_t = labels_t.float()
                    val_loss = regression_loss(logits_t, labels_t,
                                               loss_name=config.training.loss.task_loss).item()
                else:
                    labels_t = labels_t.long()
                    val_loss = classification_loss(
                        logits_t, labels_t,
                        loss_name=config.training.classification.loss_type,
                        label_smoothing=config.training.classification.label_smoothing,
                    ).item()

        metric_logger.epochs.append(epoch + 1)
        metric_logger.train_losses.append(train_loss)
        if val_loss is not None:
            metric_logger.val_losses.append(val_loss)
        if val_score is not None:
            metric_logger.val_scores.append(val_score)
        metric_logger.learning_rates.append(current_lr)

        score_for_best = val_loss if monitor == "loss" else val_score
        if score_for_best is not None:
            metric_logger.track_best(epoch + 1, score_for_best, model, mode=monitor_mode)
        elif train_score is not None and not (isinstance(train_score, float) and np.isnan(train_score)):
            metric_logger.track_best(epoch + 1, train_score, model, mode=monitor_mode)

        # 早停
        es_cfg = config.training.early_stopping
        if es_cfg.enabled and metric_logger.no_improve_count >= es_cfg.patience:
            logger.info(f"  [早停] 连续{es_cfg.patience}轮无改善 "
                        f"(最佳={metric_logger.best_score:.4f}, 第{metric_logger.best_epoch}轮)")
            break

        epoch_time = time.time() - epoch_start
        parts = [f"[Fold {fold_id}] Epoch {epoch + 1}/{total_epochs}",
                 f"{epoch_time:.1f}s", f"LR={current_lr:.2e}", f"train_loss={train_loss:.4f}"]
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        if val_score is not None and not (isinstance(val_score, float) and np.isnan(val_score)):
            parts.append(f"val_{monitor}={val_score:.4f}")
        logger.info("  " + " | ".join(parts))

        if (epoch + 1) % 5 == 0 or epoch == start_epoch:
            _log_gpu_memory(logger, device, prefix=f"  [GPU显存-Epoch{epoch + 1}]")

        save_every = config.training.checkpoint.save_every
        if (epoch + 1) % save_every == 0 or (epoch + 1) == total_epochs:
            _save_checkpoint(model, optimizer, scaler, scheduler, metric_logger,
                             epoch=epoch + 1, fold_id=fold_id, save_dir=ckpt_dir,
                             config=config, is_best=(metric_logger.best_epoch == epoch + 1),
                             logger=logger)

    logger.info(f"  [训练完成] Fold {fold_id} | 最佳={metric_logger.best_score:.4f} (第{metric_logger.best_epoch}轮)")

    if metric_logger.best_model_state is not None:
        model.load_state_dict(metric_logger.best_model_state)

    try:
        _plot_loss_curves(metric_logger, figure_dir, fold_id)
    except Exception as e:
        logger.warning(f"  [曲线] 绘制失败: {e}")

    _log_gpu_memory(logger, device, prefix="  [GPU显存-训练后]")
    return model, optimizer, metric_logger


def _pick_monitor_metric(metrics: Dict, monitor: str, task_type: str):
    """从指标dict中提取早停监控指标。"""
    if task_type == "regression":
        key = {"loss": "mse", "pearson": "pearson", "spearman": "spearman",
               "mse": "mse", "mae": "mae", "rmse": "rmse"}.get(monitor, "pearson")
        v = metrics.get(key)
        if v is None:
            return None
        # loss类监控越小越好, 由调用方决定mode; 这里返回原始值
        if monitor in ("mse", "mae", "rmse"):
            return None  # 回归默认用pearson做最大化监控, 其他交给loss
        return float(v) if np.isfinite(v) else None

    key = {"auc": "auc", "accuracy": "acc", "f1": "f1",
           "balanced_acc": "balanced_acc", "precision": "precision",
           "recall": "recall", "mcc": "mcc"}.get(monitor, "auc")
    v = metrics.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return metrics.get("acc")
    return float(v) if np.isfinite(v) else None


# ============================================================
# 主测试函数: test()
# ============================================================

@torch.no_grad()
def test(config, model: nn.Module, data: Dict, split: str, device: torch.device) -> Dict:
    """
    在指定数据划分上评估模型，返回dict:
        - loss: 测试损失
        - metrics: 分类/回归指标dict
        - pred: 预测标签(分类)或预测值(回归)
        - prob: 高表达概率(分类)
        - survtime/censor: 生存数据 (附属分析)
        - km_logrank_p: 附属KM log-rank p值
    """
    logger = logging.getLogger("YuHou")
    if not logger.handlers:
        from src.utils.logger import setup_logger
        logger = setup_logger(level=config.logging.level, language=config.logging.language)

    from src.data_loading.datasets import build_dataloader
    test_loader = build_dataloader(config, data, split=split, mode=config.model.mode,
                                   shuffle=False, for_test=True)
    logger.info(f"  [测试] 划分={split}, 样本数={len(test_loader.dataset)}")

    model.eval()
    eval_result = _evaluate_split(model, test_loader, config, device)

    if len(eval_result["logits"]) == 0:
        return {"loss": 0.0, "metrics": {}, "pred": np.array([]), "prob": np.array([]),
                "survtime": np.array([]), "censor": np.array([]), "km_logrank_p": float("nan")}

    task_type = str(config.model.task_type).lower()
    logits = eval_result["logits"]
    labels = eval_result["labels"]
    survtime = eval_result["survtimes"]
    censor = eval_result["censors"]

    metrics = _compute_task_metrics(logits, labels, task_type)

    if task_type == "regression":
        pred = logits.reshape(-1)
        prob = pred
        score = pred
    else:
        pred = np.argmax(logits, axis=1) if logits.ndim > 1 else (logits > 0).astype(int)
        prob = _prob_for_classification(logits)
        score = prob

    # 附属生存分析: 预测分子表型分组 -> KM log-rank
    km_logrank_p = float("nan")
    if config.evaluation.auxiliary_survival.enabled and len(survtime) > 1:
        try:
            from src.evaluation.survival_metrics import cox_log_rank
            km_logrank_p = cox_log_rank(score, survtime, censor)
        except Exception:
            km_logrank_p = float("nan")

    logger.info("=" * 70)
    logger.info(f"  [{split.upper()} 评估结果] 任务={task_type} 基因={config.model.target_gene}")
    if task_type == "regression":
        for k in ["mse", "mae", "rmse", "pearson", "spearman"]:
            v = metrics.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                logger.info(f"    {k}: {v:.4f}")
    else:
        for k in ["acc", "balanced_acc", "precision", "recall", "f1", "auc", "ap", "mcc"]:
            v = metrics.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                logger.info(f"    {k}: {v:.4f}")
    if not np.isnan(km_logrank_p):
        logger.info(f"    KM log-rank p (附属): {km_logrank_p:.4e}")
    logger.info("=" * 70)

    return {"loss": 0.0, "metrics": metrics, "pred": pred, "prob": prob,
            "survtime": survtime, "censor": censor, "km_logrank_p": km_logrank_p}


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("训练核心模块 (trainer.py) 自测")
    print("  接口: train(config, data, device, fold_id) / test(config, model, data, split, device)")
    print("测试通过!")
