# -*- coding: utf-8 -*-
"""classification_loss.py - 二分类 BCEWithLogits 损失 (可加 pos_weight 处理类别不平衡)。"""

import torch
import torch.nn as nn


def bce_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor = None) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(
        logits.view(-1), target.view(-1), pos_weight=pos_weight)
