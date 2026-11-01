# -*- coding: utf-8 -*-
"""
abmil.py - 门控注意力多实例聚合 (Gated Attention ABMIL)。

输入: patch实例特征 [B, N, feat_dim]
输出: bag全局表征 [B, feat_dim] + 实例注意力权重 [B, N]

公式 (Ilse et al., ABMIL):
    a_i = W · (tanh(V h_i) ⊙ sigmoid(U h_i))
    α = softmax(a)
    z = Σ α_i h_i
"""

import torch
import torch.nn as nn


class GatedABMIL(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.Sigmoid())
        self.w = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor):
        """
        h: [B, N, feat_dim]
        返回: (z, attn)  z=[B, feat_dim], attn=[B, N]
        """
        h = self.dropout(h)
        v = self.V(h)          # [B, N, hidden]
        u = self.U(h)          # [B, N, hidden]
        a = self.w(v * u).squeeze(-1)   # [B, N]
        attn = torch.softmax(a, dim=1)  # [B, N]
        z = torch.sum(attn.unsqueeze(-1) * h, dim=1)  # [B, feat_dim]
        return z, attn
