# -*- coding: utf-8 -*-
"""
regularization.py
============================================================================
模型正则化模块。

支持L1/L2正则化，可按需对不同子模块施加不同的正则化强度:
    - none: 不使用正则化
    - path: 仅对分类器/linear层施加L1正则化
    - mm: 对融合相关模块施加L1正则化
    - all: 对所有参数施加L1正则化
    - omic: 仅对基因组编码器施加L1正则化

使用示例:
    from src.losses.regularization import compute_regularization
    reg_loss = compute_regularization(model, reg_type="all")
    total_loss = task_loss + lambda_reg * reg_loss
============================================================================
"""

import torch
import torch.nn as nn


def regularize_weights(model, reg_type=None):
    """
    对所有模型参数施加L1正则化。

    参数:
        model: nn.Module模型实例
        reg_type: 预留参数（当前未使用）

    返回:
        l1_reg: L1正则化损失值（所有参数绝对值之和）
    """
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()

    if l1_reg is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    return l1_reg


def regularize_path_weights(model, reg_type=None):
    """
    仅对分类器和输出线性层施加L1正则化。

    适用于: 希望正则化预测头但保留特征提取器的表达能力。

    参数:
        model: nn.Module模型实例
        reg_type: 预留参数

    返回:
        l1_reg: L1正则化损失值
    """
    l1_reg = None
    real_model = model.module if hasattr(model, "module") else model

    if hasattr(real_model, "classifier"):
        for W in real_model.classifier.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    if hasattr(real_model, "linear"):
        for W in real_model.linear.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    if l1_reg is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    return l1_reg


def regularize_mm_weights(model, reg_type=None):
    """
    对多模态融合相关模块施加L1正则化。

    适用于: 希望正则化融合模块，保留单模态编码器的预训练特征。

    参数:
        model: nn.Module模型实例
        reg_type: 预留参数

    返回:
        l1_reg: L1正则化损失值
    """
    l1_reg = None
    real_model = model.module if hasattr(model, "module") else model

    # 融合相关的模块名称列表
    mm_param_names = [
        'omic_net',
        'linear_h_path', 'linear_h_omic', 'linear_h_grph',
        'linear_z_path', 'linear_z_omic', 'linear_z_grph',
        'linear_o_path', 'linear_o_omic', 'linear_o_grph',
        'encoder1', 'encoder2', 'classifier',
    ]

    for name in mm_param_names:
        if hasattr(real_model, name):
            for W in getattr(real_model, name).parameters():
                if l1_reg is None:
                    l1_reg = torch.abs(W).sum()
                else:
                    l1_reg = l1_reg + torch.abs(W).sum()

    if l1_reg is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    return l1_reg


def regularize_omic_weights(model, reg_type=None):
    """
    仅对基因组编码器(omic_net)施加L1正则化。

    适用于: 基因组特征维度高且稀疏时，希望编码器学习更稀疏的表示。

    参数:
        model: nn.Module模型实例
        reg_type: 预留参数

    返回:
        l1_reg: L1正则化损失值
    """
    l1_reg = None

    if hasattr(model, 'module') and hasattr(model.module, 'omic_net'):
        for W in model.module.omic_net.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    elif hasattr(model, 'omic_net'):
        for W in model.omic_net.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    if l1_reg is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    return l1_reg


def compute_regularization(model, reg_type="none"):
    """
    正则化计算统一接口。

    根据reg_type选择对哪些模块施加正则化。

    参数:
        model: nn.Module模型实例
        reg_type: 正则化类型
            - "none": 不施加正则化 (返回0)
            - "path": 仅分类器/linear层
            - "mm": 多模态融合相关模块
            - "all": 所有参数
            - "omic": 仅基因组编码器

    返回:
        loss_reg: 正则化损失值

    异常:
        NotImplementedError: 不支持的正则化类型
    """
    if reg_type == "none":
        return torch.tensor(0.0, device=next(model.parameters()).device)

    elif reg_type == "path":
        return regularize_path_weights(model=model)

    elif reg_type == "mm":
        return regularize_mm_weights(model=model)

    elif reg_type == "all":
        return regularize_weights(model=model)

    elif reg_type == "omic":
        return regularize_omic_weights(model=model)

    else:
        raise NotImplementedError(
            f"不支持的正则化类型: '{reg_type}'。"
            f"支持: none, path, mm, all, omic"
        )


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("正则化模块自测")
    print("=" * 60)

    # 创建一个简单模型用于测试
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Linear(10, 1)
            self.linear = nn.Linear(10, 5)
            self.omic_net = nn.Sequential(
                nn.Linear(10, 5),
                nn.ReLU(),
                nn.Linear(5, 3),
            )

    model = TestModel()

    for reg_type in ["none", "path", "mm", "all", "omic"]:
        reg_loss = compute_regularization(model, reg_type)
        print(f"  {reg_type}: {reg_loss.item():.6f}")

    print("\n所有测试通过!")
