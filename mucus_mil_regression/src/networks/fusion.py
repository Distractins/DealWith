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

    故事: 病理图像是「主诊医生」, 从形态给出主要判断(锚定排序);
          临床信息是「会诊专家」, 只做残差校正(微调绝对数值);
          patch注意力集中度 = 主诊医生的「信心」——
          信心足(注意力集中)就独自判断, 拿不准(注意力分散)才请临床补充校正。

    公式:
        image_score = MLP_img(z_img)
        clin_residual = MLP_clin(z_clin)
        entropy = -Σ attn·log(attn)             # 越高越分散(不确定)
        weight = sigmoid(MLP_gate(entropy))      # 不确定 -> 权重高
        final = image_score + weight · clin_residual
    """

    def __init__(self, img_dim: int, clin_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.img_head = nn.Sequential(
            nn.Linear(img_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.clin_head = nn.Sequential(
            nn.Linear(clin_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # 注意力熵 -> 临床校正权重 (熵高=图像不确定=多用临床)
        self.gate = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, z_img: torch.Tensor, z_clin: torch.Tensor, attn: torch.Tensor):
        image_score = self.img_head(z_img).squeeze(-1)               # [B]
        clin_residual = self.clin_head(z_clin).squeeze(-1)           # [B]
        entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=1, keepdim=True)  # [B,1]
        clinical_weight = self.gate(entropy).squeeze(-1)             # [B]
        return image_score + clinical_weight * clin_residual
