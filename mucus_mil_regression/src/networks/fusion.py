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
