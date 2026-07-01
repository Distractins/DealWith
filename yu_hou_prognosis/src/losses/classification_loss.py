# -*- coding: utf-8 -*-
"""
classification_loss.py
============================================================================
分类损失函数模块。

包含三种分类损失:
    1. ClassificationLoss: CE/BCE损失 + label_smoothing + class_weight
    2. FocalLoss: 针对类别不平衡的Focal Loss
    3. 类别权重生成函数

使用场景:
    - N分期分类任务 (N0/N1/N2三分类): 使用CE损失
    - 中位风险二分类: 使用BCE损失
    - 类别严重不平衡时: 使用FocalLoss

使用示例:
    from src.losses.classification_loss import classification_loss, FocalLoss
    loss = classification_loss(logits, labels, loss_name="ce")
============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 类别权重生成
# ============================================================

def make_class_weights_from_labels(labels, device=None, beta=None):
    """
    根据训练标签计算类别权重（用于处理类别不平衡）。

    支持两种权重模式:
        1. 倒数频率权重: w_c = N / (C * n_c)（默认）
        2. Effective Number权重: w_c = (1-β) / (1-β^n_c)（Cui et al., CVPR 2019）

    参数:
        labels: [N] 整数类别标签
        device: torch设备
        beta: Effective Number参数（0<β<1），None时使用倒数频率权重

    返回:
        weight_t: [C] 类别权重张量

    使用示例:
        weights = make_class_weights_from_labels(train_labels, device="cuda")
        criterion = nn.CrossEntropyLoss(weight=weights)
    """
    labels = np.asarray(labels).reshape(-1).astype(int)
    if len(labels) == 0:
        return None

    uniq, cnt = np.unique(labels, return_counts=True)
    n_class = int(np.max(uniq)) + 1
    weights = np.ones(n_class, dtype=np.float32)

    if beta is not None and 0.0 < beta < 1.0:
        # Effective Number权重: 按样本数指数衰减
        for c, n in zip(uniq, cnt):
            effective_num = 1.0 - (beta ** int(n))
            weights[int(c)] = (1.0 - beta) / max(effective_num, 1e-8)
    else:
        # 倒数频率权重
        total = float(np.sum(cnt))
        for c, n in zip(uniq, cnt):
            weights[int(c)] = total / max(float(len(cnt)) * float(n), 1e-8)

    # 归一化（保持总均值≈1）
    weights = weights / max(np.mean(weights), 1e-8)

    weight_t = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        weight_t = weight_t.to(device)
    return weight_t


# ============================================================
# ClassificationLoss: 分类损失统一接口
# ============================================================

def classification_loss(logits, labels, loss_name="ce", class_weights=None,
                        label_smoothing=0.0, pos_weight=None):
    """
    分类损失统一接口。

    参数:
        logits: [B, C] 或 [B] 模型输出的logits
        labels: [B] 整数类别标签
        loss_name: 损失类型 ("ce" / "bce")
        class_weights: [C] 类别权重（CE使用）
        label_smoothing: 标签平滑系数（0.0-1.0，仅CE有效）
        pos_weight: [1] 正类权重（BCE使用）

    返回:
        loss: 标量损失值

    异常:
        ValueError: 不支持的损失类型
    """
    loss_name = str(loss_name).lower()

    if loss_name == "ce":
        # CrossEntropyLoss: 适用于多分类
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        labels = labels.long().view(-1)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(label_smoothing),
        )
        return criterion(logits, labels)

    elif loss_name == "bce":
        # BCEWithLogitsLoss: 适用于二分类
        if logits.dim() > 1 and logits.size(-1) == 1:
            logits = logits.view(-1)
        labels = labels.float().view(-1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return criterion(logits, labels)

    else:
        raise ValueError(f"不支持的分类损失类型: {loss_name}。支持: ce, bce")


# ============================================================
# FocalLoss: 针对类别不平衡的Focal Loss
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss - 针对类别不平衡的损失函数。

    通过在标准交叉熵损失基础上添加 (1-p_t)^γ 因子，
    降低易分类样本的损失权重，使模型更关注难分类样本。

    核心公式:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    参数:
        alpha: 类别权重因子（默认0.25，参考RetinaNet论文）
        gamma: 聚焦参数（默认2.0，越大越关注难样本）
        reduction: 归约方式 ("mean" / "sum" / "none")

    适用场景:
        - N分期分类中某些类别样本极少
        - 中位风险分类中高低风险不平衡

    参考:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

    使用示例:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        loss = criterion(logits, labels)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        计算Focal Loss。

        参数:
            logits: [B, C] 模型输出logits
            labels: [B] 整数类别标签

        返回:
            loss: 标量损失值
        """
        # 计算交叉熵
        ce_loss = F.cross_entropy(logits, labels.long(), reduction='none')

        # 计算p_t (正确类别的预测概率)
        p_t = torch.exp(-ce_loss)

        # Focal Loss = α * (1-p_t)^γ * CE
        focal_loss = self.alpha * ((1 - p_t) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("分类损失函数模块自测")
    print("=" * 60)

    torch.manual_seed(42)
    B, C = 16, 3
    logits = torch.randn(B, C)
    labels = torch.randint(0, C, (B,))

    # 测试CE损失
    loss_ce = classification_loss(logits, labels, loss_name="ce")
    print(f"\n  CE Loss: {loss_ce.item():.6f}")

    # 测试CE + label_smoothing
    loss_smooth = classification_loss(logits, labels, loss_name="ce", label_smoothing=0.1)
    print(f"  CE + LabelSmoothing(0.1): {loss_smooth.item():.6f}")

    # 测试BCE损失
    logits_b = torch.randn(B)
    labels_b = torch.randint(0, 2, (B,)).float()
    loss_bce = classification_loss(logits_b, labels_b, loss_name="bce")
    print(f"  BCE Loss: {loss_bce.item():.6f}")

    # 测试FocalLoss
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    loss_focal = focal(logits, labels)
    print(f"  Focal Loss (γ=2.0): {loss_focal.item():.6f}")

    # 测试类别权重生成
    imbalanced_labels = np.array([0] * 100 + [1] * 20 + [2] * 5)
    weights = make_class_weights_from_labels(imbalanced_labels)
    print(f"\n  不平衡标签的类别权重: {weights.numpy()}")

    print("\n所有测试通过!")
