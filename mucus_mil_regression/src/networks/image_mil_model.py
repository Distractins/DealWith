# -*- coding: utf-8 -*-
"""
image_mil_model.py - 纯图像ABMIL回归模型 (实验2)。

主干(ResNet/UNI)提取patch实例特征 -> GatedABMIL聚合 -> 回归头。
返回: 预测得分 [B] + 实例注意力权重 [B, N] (供可视化)。
"""

import torch
import torch.nn as nn

from src.networks.backbone import build_backbone
from src.networks.abmil import GatedABMIL


class ImageMILModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone, feat_dim = build_backbone(config)
        self.abmil = GatedABMIL(feat_dim, config.model.mil_hidden_dim, config.model.mil_dropout)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, config.model.mil_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.model.mil_dropout),
            nn.Linear(config.model.mil_hidden_dim, 1),
        )

    def forward(self, bag: torch.Tensor, clinical: torch.Tensor = None):
        B, N, C, H, W = bag.shape
        x = self.backbone(bag.view(B * N, C, H, W))   # [B*N, feat_dim]
        x = x.view(B, N, -1)                           # [B, N, feat_dim]
        z, attn = self.abmil(x)                        # z=[B, feat_dim]
        out = self.head(z).squeeze(-1)                 # [B]
        return out, attn
