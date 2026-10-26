# -*- coding: utf-8 -*-
"""checkpoint.py - 断点保存/恢复 (跨fold独立)。"""

import torch
from pathlib import Path
from typing import Optional


def save_checkpoint(ckpt_dir: Path, fold_id: int, epoch: int, model, optimizer,
                    scaler, scheduler, metric_logger, config) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"fold{fold_id}_epoch{epoch}.pth"
    payload = {
        "epoch": epoch,
        "fold_id": fold_id,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "metric_logger": metric_logger.to_dict(),
        "config": config.to_dict(),
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, str(path))

    keep = config.training.checkpoint.keep_last_n
    if keep > 0:
        files = sorted(ckpt_dir.glob(f"fold{fold_id}_epoch*.pth"),
                       key=lambda p: int(p.stem.split("epoch")[-1]))
        while len(files) > keep:
            files.pop(0).unlink(missing_ok=True)
    return path


def load_checkpoint(ckpt_dir: Path, fold_id: int, model, optimizer, scaler,
                    scheduler, metric_logger) -> int:
    if not ckpt_dir.exists():
        return 0
    files = sorted(ckpt_dir.glob(f"fold{fold_id}_epoch*.pth"),
                   key=lambda p: int(p.stem.split("epoch")[-1]))
    if not files:
        return 0
    latest = files[-1]
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
    print(f"[Checkpoint] 从第{ckpt['epoch']}轮恢复: {latest.name}")
    return ckpt["epoch"]
