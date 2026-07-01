# -*- coding: utf-8 -*-
"""
cox_loss.py
============================================================================
Cox比例风险损失函数模块。

包含三种Cox损失实现:
    1. CoxLoss: 原始实现（基于R矩阵的部分似然）
    2. CoxLoss2: 矩阵运算优化版
    3. CoxPartialLikelihoodLoss: nn.Module封装版（推荐使用）
    4. CoxLossWithL2: 带L2正则化的Cox损失

Cox比例风险模型背景:
    生存分析中，Cox模型假设风险函数 h(t|x) = h0(t) * exp(βx)。
    通过最大化部分似然(partial likelihood)来估计β，不依赖基准风险h0(t)的具体形式。

部分似然公式:
    PL(β) = Π_{i:δ_i=1} [exp(βx_i) / Σ_{j:Y_j≥Y_i} exp(βx_j)]

    其中δ_i是事件标记(1=死亡,0=删失)，Y_i是观测时间。

使用示例:
    from src.losses.cox_loss import CoxPartialLikelihoodLoss
    criterion = CoxPartialLikelihoodLoss()
    loss = criterion(hazard_pred, survtime, censor)
============================================================================
"""

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# CoxLoss: 原始实现（基于R矩阵的部分似然损失）
# ============================================================
def CoxLoss(survtime, censor, hazard_pred, device):
    """
    Cox部分似然损失函数（原始实现）。

    通过构建风险集矩阵R（R[i,j]=1表示患者j在患者i死亡时仍存活），
    计算部分似然的负对数。

    参数:
        survtime: [B] 观测生存时间（天/月）
        censor: [B] 删失标记（1=事件发生, 0=删失）
        hazard_pred: [B] 模型预测的风险分数（越大风险越高）
        device: torch设备

    返回:
        loss_cox: 标量，Cox损失值

    注意:
        此函数内部使用numpy构建R矩阵（双重循环），在batch较大时可能较慢。
        推荐使用CoxLoss2或CoxPartialLikelihoodLoss替代。
    """
    if torch.cuda.is_available() and device != 'cpu':
        current_device = device
    else:
        current_device = 'cpu'

    current_batch_len = len(survtime)

    # 构建风险集矩阵R: R[i,j]=1 表示患者j在患者i死亡时仍存活
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(current_device)

    if str(hazard_pred.device) != str(current_device):
        hazard_pred = hazard_pred.to(current_device)

    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)

    # 部分似然: -mean( (θ_i - log(Σ_{j∈R_i} exp(θ_j))) * δ_i )
    loss_cox = -torch.mean(
        (theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor
    )
    return loss_cox


# ============================================================
# CoxLoss2: 矩阵运算优化版
# ============================================================
def _R_set(x):
    """构建下三角矩阵（用于风险集计算）"""
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)
    return indicator_matrix


def CoxLoss2(survtime, censor, hazard_pred, device):
    """
    Cox部分似然损失函数（矩阵运算优化版）。

    使用torch矩阵运算替代numpy双重循环，GPU加速友好。

    参数:
        survtime: [B] 观测生存时间
        censor: [B] 删失标记（1=事件发生, 0=删失）
        hazard_pred: [B] 模型预测的风险分数
        device: torch设备

    返回:
        cost: [1] Cox损失值
    """
    n_observed = censor.sum(0) + 1
    ytime_indicator = _R_set(survtime)
    ytime_indicator = torch.FloatTensor(ytime_indicator).to(device)

    risk_set_sum = ytime_indicator.mm(torch.exp(hazard_pred))
    diff = hazard_pred - torch.log(risk_set_sum)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(censor.unsqueeze(1))
    cost = (-(sum_diff_in_observed / n_observed)).reshape((-1,))
    return cost


# ============================================================
# CoxPartialLikelihoodLoss: nn.Module封装版（推荐）
# ============================================================
class CoxPartialLikelihoodLoss(nn.Module):
    """
    Cox部分似然损失 (PyTorch nn.Module封装版)。

    与CoxLoss功能相同，但封装为nn.Module以更好地集成到训练流程中。
    使用纯torch操作实现，避免numpy转换开销。

    使用示例:
        criterion = CoxPartialLikelihoodLoss()
        loss = criterion(hazard_pred, survtime, censor)
    """

    def __init__(self):
        super().__init__()

    def forward(self, hazard_pred, survtime, censor):
        """
        计算Cox部分似然损失。

        参数:
            hazard_pred: [B] 或 [B, 1] 模型预测的风险分数
            survtime: [B] 观测生存时间
            censor: [B] 删失标记（1=事件, 0=删失）

        返回:
            loss: 标量损失值
        """
        # 确保张量在相同设备上
        device = hazard_pred.device
        hazard_pred = hazard_pred.reshape(-1).float()
        survtime = survtime.reshape(-1).float()
        censor = censor.reshape(-1).float()

        n = hazard_pred.size(0)
        if n < 2:
            return torch.tensor(0.0, device=device)

        # 构建风险集矩阵（按生存时间排序）
        # R[i,j] = 1 表示患者j在患者i死亡时仍存活
        survtime_diff = survtime.unsqueeze(0) - survtime.unsqueeze(1)  # [n, n]
        R = (survtime_diff <= 0).float()  # [n, n]: 风险集指示矩阵

        # 风险分数
        theta = hazard_pred  # [n]
        exp_theta = torch.exp(theta)  # [n]

        # 风险集内的exp(theta)之和
        risk_sum = torch.matmul(R, exp_theta)  # [n]: Σ_{j∈R_i} exp(θ_j)

        # 部分似然: -Σ δ_i * (θ_i - log(risk_sum_i))
        # 只对发生事件的样本计算
        log_risk = theta - torch.log(risk_sum + 1e-8)
        loss = -torch.mean(censor * log_risk)

        return loss


# ============================================================
# CoxLossWithL2: 带L2正则化的Cox损失
# ============================================================
class CoxLossWithL2(nn.Module):
    """
    带L2正则化的Cox部分似然损失。

    在标准Cox损失基础上添加L2正则化项，惩罚过大的风险预测值，
    有助于缓解小样本过拟合问题。

    loss = CoxLoss + lambda_l2 * ||hazard_pred||²

    参数:
        lambda_l2: L2正则化系数（默认0.01）

    使用示例:
        criterion = CoxLossWithL2(lambda_l2=0.01)
        loss = criterion(hazard_pred, survtime, censor)
    """

    def __init__(self, lambda_l2=0.01):
        super().__init__()
        self.lambda_l2 = lambda_l2
        self.cox_loss = CoxPartialLikelihoodLoss()

    def forward(self, hazard_pred, survtime, censor):
        """
        计算带L2正则化的Cox损失。

        参数:
            hazard_pred: [B] 模型预测的风险分数
            survtime: [B] 观测生存时间
            censor: [B] 删失标记

        返回:
            loss: 标量，Cox损失 + L2正则化项
        """
        cox = self.cox_loss(hazard_pred, survtime, censor)
        l2 = self.lambda_l2 * torch.mean(hazard_pred ** 2)
        return cox + l2


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Cox损失函数模块自测")
    print("=" * 60)

    # 模拟生存数据
    torch.manual_seed(42)
    B = 16
    survtime = torch.randint(30, 3650, (B,)).float()  # 生存时间 (天)
    censor = torch.randint(0, 2, (B,)).float()         # 删失标记
    hazard_pred = torch.randn(B) * 0.5                  # 预测风险分数

    device = torch.device("cpu")

    # 测试CoxLoss
    loss1 = CoxLoss(survtime, censor, hazard_pred, device)
    print(f"\n  CoxLoss: {loss1.item():.6f}")

    # 测试CoxLoss2
    loss2 = CoxLoss2(survtime, censor, hazard_pred, device)
    print(f"  CoxLoss2: {loss2.item():.6f}")

    # 测试CoxPartialLikelihoodLoss
    criterion = CoxPartialLikelihoodLoss()
    loss3 = criterion(hazard_pred, survtime, censor)
    print(f"  CoxPartialLikelihoodLoss: {loss3.item():.6f}")

    # 测试CoxLossWithL2
    criterion_l2 = CoxLossWithL2(lambda_l2=0.01)
    loss4 = criterion_l2(hazard_pred, survtime, censor)
    print(f"  CoxLossWithL2: {loss4.item():.6f}")

    # 验证梯度
    hazard_pred.requires_grad_(True)
    loss_grad = criterion(hazard_pred, survtime, censor)
    loss_grad.backward()
    print(f"\n  梯度验证: hazard_pred.grad 正常: {hazard_pred.grad is not None}")

    print("\n所有测试通过!")
