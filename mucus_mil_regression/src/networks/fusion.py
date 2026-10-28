# -*- coding: utf-8 -*-
"""
fusion.py - 图像-临床多模态融合 (实验3的两种融合方式)。

1) LateConcatFusion   : 晚拼接融合 (基线)
2) AdaptiveGateFusion : 自适应门控融合 (本文提出的改进策略)
    gate = σ(W_g [proj_img; proj_clin])
    fused = gate ⊙ proj_img + (1 - gate) ⊙ proj_clin
"""

import torch
import torch.nn as nn


class LateConcatFusion(nn.Module):
    def __init__(self, img_dim: int, clin_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim + clin_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_img: torch.Tensor, z_clin: torch.Tensor):
        fused = torch.cat([z_img, z_clin], dim=1)
        return self.net(fused).squeeze(-1)


class AdaptiveGateFusion(nn.Module):
    """自适应门控融合: 逐维学习图像/临床两路权重。"""

    def __init__(self, img_dim: int, clin_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.clin_proj = nn.Linear(clin_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_img: torch.Tensor, z_clin: torch.Tensor):
        hi = self.img_proj(z_img)
        hc = self.clin_proj(z_clin)
        g = self.gate(torch.cat([hi, hc], dim=1))       # [B, hidden_dim]
        fused = g * hi + (1.0 - g) * hc
        return self.head(fused).squeeze(-1)


class AnchorResidualFusion(nn.Module):
    """锚定残差校正融合 (ARC: Anchor-Residual Correction)。

    非对称结构: 图像是主项(锚定排序), 临床只做残差校正(改校准),
    校正强度由 patch注意力熵(图像不确定性)通过轻量门控得到标量w。

    y_pred = y_img + w · res_clin
    w = sigmoid(MLP_gate(entropy)),  entropy = -Σ α_i ln α_i

    与旧对称门控对比:
        旧:  g·hi + (1-g)·hc  (零和凸组合, 逐维门控, 临床会挤占图像)
        新:  y_img + w·res_clin (图像永远主项, 临床只叠加残差, 单标量门控)
    """

    def __init__(self, img_dim: int, clin_dim: int, hidden_dim: int = 256,
                 dropout: float = 0.3, gate_dropout: float = 0.1):
        super().__init__()
        # hi/hc 各自 LayerNorm, 统一特征尺度 (约束2)
        self.img_norm = nn.LayerNorm(img_dim)
        self.clin_norm = nn.LayerNorm(clin_dim)

        # 图像主预测头 (产出 y_img)
        self.img_head = nn.Sequential(
            nn.Linear(img_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # 临床残差头 (只学校正量 res_clin, 不学最终预测)
        self.clin_head = nn.Sequential(
            nn.Linear(clin_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # 轻量门控: entropy -> 标量 w (约束3, 非逐维)
        self.gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(gate_dropout),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        # 用于训练日志统计 (每个batch的熵/w均值)
        self.last_entropy = 0.0
        self.last_w = 0.0

    def forward(self, z_img: torch.Tensor, z_clin: torch.Tensor, attn: torch.Tensor):
        hi = self.img_norm(z_img)                                  # 图像特征归一化
        hc = self.clin_norm(z_clin)                                # 临床特征归一化
        y_img = self.img_head(hi).squeeze(-1)                      # [B] 图像主预测
        res_clin = self.clin_head(hc).squeeze(-1)                  # [B] 临床残差校正

        # 注意力熵: 图像不确定性 (熵高=分散=不确定)
        entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=1, keepdim=True)  # [B,1]
        w = self.gate(entropy).squeeze(-1)                         # [B] ∈ [0,1]

        self.last_entropy = float(entropy.detach().mean().item())
        self.last_w = float(w.detach().mean().item())

        return y_img + w * res_clin                                # [B]
