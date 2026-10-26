# -*- coding: utf-8 -*-
"""
trainer.py - 训练/评估引擎 (MIL回归)。

损失: MSE。 早停监控: spearman/pearson/loss。
支持混合精度(AMP)、梯度累积、断点续训。
"""

import copy
import math
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.amp import GradScaler, autocast
    _AMP_NEW = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_NEW = False

from src.networks.clinical_mlp import ClinicalMLP
from src.networks.image_mil_model import ImageMILModel
from src.networks.multimodal_model import MultimodalModel
from src.losses.regression_loss import mse_loss
from src.evaluation.regression_metrics import regression_metrics


# ============================================================
# 模型工厂
# ============================================================

def build_model(config):
    mt = config.model.model_type
    if mt == "clinical_mlp":
        return ClinicalMLP(config.model.clinical_input_dim,
                           config.model.clinical_hidden_dims,
                           config.model.clinical_dropout)
    if mt == "image_mil":
        return ImageMILModel(config)
    if mt == "multimodal":
        return MultimodalModel(config)
    raise ValueError(f"不支持的model_type: {mt}")


# ============================================================
# 调度器 (线性热身 + 余弦)
# ============================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr):
        self.optimizer = optimizer
        self.warmup = max(1, warmup_epochs)
        self.total = max(1, total_epochs)
        self.min_lr = min_lr
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.last_epoch = 0

    def _lr(self, epoch):
        base = self.base_lrs[0]
        if epoch < self.warmup:
            return self.min_lr + (base - self.min_lr) * (epoch + 1) / self.warmup
        progress = (epoch - self.warmup) / max(self.total - self.warmup, 1)
        return self.min_lr + (base - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    def step(self):
        lr = self._lr(self.last_epoch)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.last_epoch += 1

    def state_dict(self):
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, d):
        self.last_epoch = d["last_epoch"]


# ============================================================
# MetricLogger
# ============================================================

class MetricLogger:
    def __init__(self, monitor="spearman", mode="max"):
        self.monitor = monitor
        self.mode = mode
        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.best_epoch = 0
        self.best_state = None
        self.no_improve = 0
        self.history = []

    def update(self, epoch, score, model):
        improved = (score > self.best_score) if self.mode == "max" else (score < self.best_score)
        if self.best_state is None or improved:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.no_improve = 0
        else:
            self.no_improve += 1
        self.history.append({"epoch": epoch, "score": score})

    def to_dict(self):
        return {"best_score": self.best_score, "best_epoch": self.best_epoch, "history": self.history}

    def load_from_dict(self, d):
        self.best_score = d.get("best_score", self.best_score)
        self.best_epoch = d.get("best_epoch", 0)
        self.history = d.get("history", [])


# ============================================================
# 单轮训练
# ============================================================

def _train_one_epoch(model, loader, optimizer, scheduler, scaler, config, device):
    model.train()
    accum = config.training.gradient_accumulation_steps
    use_amp = config.training.amp.enabled
    running = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)

    for i, (bag, clinical, y) in enumerate(loader):
        bag = bag.to(device)
        clinical = clinical.to(device)
        y = y.to(device)

        amp_ctx = autocast("cuda", enabled=use_amp) if _AMP_NEW else autocast(enabled=use_amp)
        with amp_ctx:
            out, _attn = model(bag, clinical)
            loss = mse_loss(out, y) / accum

        scaler.scale(loss).backward()
        if (i + 1) % accum == 0 or (i + 1) == len(loader):
            if config.training.gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        running += loss.item() * accum
        n += 1
    return running / max(n, 1)


# ============================================================
# 评估
# ============================================================

@torch.no_grad()
def _evaluate(model, loader, config, device):
    model.eval()
    preds, targets = [], []
    all_attn = []
    for bag, clinical, y in loader:
        bag = bag.to(device)
        clinical = clinical.to(device)
        out, attn = model(bag, clinical)
        preds.append(out.cpu().numpy())
        targets.append(y.cpu().numpy())
        if attn is not None:
            all_attn.append(attn.cpu().numpy())
    preds = np.concatenate(preds).reshape(-1) if preds else np.array([])
    targets = np.concatenate(targets).reshape(-1) if targets else np.array([])
    attn = np.concatenate(all_attn, axis=0) if all_attn else None
    return preds, targets, attn


# ============================================================
# 训练主函数
# ============================================================

def train(config, data: Dict, device: torch.device, fold_id: int = 1):
    logger = logging.getLogger("MucusMIL")
    if not logger.handlers:
        from src.utils.logger import setup_logger
        logger = setup_logger(level=config.logging.level, language=config.logging.language)

    from src.data_loading.dataset import build_dataloader
    need_patches = config.model.model_type != "clinical_mlp"
    train_loader = build_dataloader(config, data, split="train", shuffle=True, need_patches=need_patches)
    val_loader = build_dataloader(config, data, split="val", shuffle=False, need_patches=need_patches) if "val" in data else None

    model = build_model(config).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  [模型] model_type={config.model.model_type}, backbone={config.model.backbone}, "
                f"总参数={total:,}, 可训练={trainable:,}")

    opt_type = config.training.optimizer.type
    lr = config.training.optimizer.lr
    wd = config.training.optimizer.weight_decay
    betas = tuple(config.training.optimizer.betas)
    if opt_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    elif opt_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    n_epochs = config.training.scheduler.n_epochs
    scheduler = WarmupCosineScheduler(optimizer, config.training.scheduler.warmup_epochs,
                                      n_epochs, config.training.scheduler.min_lr)
    use_amp = config.training.amp.enabled
    scaler = GradScaler("cuda", enabled=use_amp) if _AMP_NEW else GradScaler(enabled=use_amp)

    monitor = config.training.early_stopping.monitor
    mode = "min" if monitor == "loss" else "max"
    metric_logger = MetricLogger(monitor=monitor, mode=mode)

    ckpt_dir = config.get_subdir("ckpt") / f"fold{fold_id}"
    start_epoch = 0
    if config.training.checkpoint.resume:
        from src.training.checkpoint import load_checkpoint
        start_epoch = load_checkpoint(ckpt_dir, fold_id, model, optimizer, scaler,
                                      scheduler, metric_logger)
        model = model.to(device)

    logger.info(f"  [训练] epochs={n_epochs}, batch={config.training.batch_size}×"
                f"{config.training.gradient_accumulation_steps}, monitor={monitor}")

    for epoch in range(start_epoch, n_epochs):
        train_loss = _train_one_epoch(model, train_loader, optimizer, scheduler, scaler, config, device)

        val_score = None
        val_metrics = None
        if val_loader is not None:
            vp, vt, _ = _evaluate(model, val_loader, config, device)
            if len(vt) > 1:
                val_metrics = regression_metrics(vp, vt)
                val_score = val_metrics.get(monitor, val_metrics.get("spearman"))
                if monitor == "loss":
                    val_score = float(np.mean((vp - vt) ** 2))

        if val_score is not None and not (isinstance(val_score, float) and np.isnan(val_score)):
            metric_logger.update(epoch + 1, val_score, model)
        else:
            metric_logger.update(epoch + 1, -train_loss, model)  # 无val时用train loss兜底

        log = f"  [Fold {fold_id}] Epoch {epoch+1}/{n_epochs} | loss={train_loss:.4f}"
        if val_score is not None and not np.isnan(val_score):
            log += f" | val_{monitor}={val_score:.4f}"
        logger.info(log)

        es = config.training.early_stopping
        if es.enabled and metric_logger.no_improve >= es.patience:
            logger.info(f"  [早停] 最佳={metric_logger.best_score:.4f} (第{metric_logger.best_epoch}轮)")
            break

        save_every = config.training.checkpoint.save_every
        if (epoch + 1) % save_every == 0 or (epoch + 1) == n_epochs:
            from src.training.checkpoint import save_checkpoint
            save_checkpoint(ckpt_dir, fold_id, epoch + 1, model, optimizer, scaler,
                            scheduler, metric_logger, config)

    if metric_logger.best_state is not None:
        model.load_state_dict(metric_logger.best_state)

    # 单独保存最佳模型 (供断点续训 / test模式直接加载)
    torch.save(model.state_dict(), ckpt_dir / f"fold{fold_id}_best.pth")
    return model, metric_logger


# ============================================================
# 测试
# ============================================================

def test(config, model, data: Dict, split: str, device: torch.device) -> Dict:
    from src.data_loading.dataset import build_dataloader
    need_patches = config.model.model_type != "clinical_mlp"
    loader = build_dataloader(config, data, split=split, shuffle=False, need_patches=need_patches)
    preds, targets, attn = _evaluate(model, loader, config, device)
    metrics = regression_metrics(preds, targets) if len(targets) > 1 else {}
    return {
        "preds": preds,
        "targets": targets,
        "attn": attn,
        "metrics": metrics,
        "patient_ids": data[split].get("patient_ids", []),
    }
