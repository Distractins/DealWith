# -*- coding: utf-8 -*-
"""
clinical_mlp.py - 临床基线MLP + 临床特征编码器。

实验(1): 仅临床表格特征回归粘液分化得分, 评估图像带来的增量价值。
"""

import torch
import torch.nn as nn
from typing import List


def _mlp_block(in_dim: int, hidden_dims: List[int], dropout: float,
               out_dim: int = 1) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class ClinicalMLP(nn.Module):
    """临床基线: 仅临床表格特征 -> 回归得分。"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = (64, 32), dropout: float = 0.3):
        super().__init__()
        self.net = _mlp_block(input_dim, list(hidden_dims), dropout, out_dim=1)

    def forward(self, bag: torch.Tensor = None, clinical: torch.Tensor = None):
        out = self.net(clinical).squeeze(-1)
        return out, None   # 统一 (out, attn) 接口, 临床基线无注意力


class ClinicalFeatureEncoder(nn.Module):
    """临床特征编码器 (用于多模态融合, 输出 hidden_dims[-1] 维表征)。"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = (64, 32), dropout: float = 0.3):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, clinical: torch.Tensor):
        return self.net(clinical)
