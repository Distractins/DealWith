# -*- coding: utf-8 -*-
"""
fusion_gmu.py
============================================================================
门控多模态单元 (Gated Multimodal Unit, GMU) 融合模块。

设计思想:
    受到LSTM/GRU门控机制的启发，GMU通过一个sigmoid门控来决定
    两个模态特征的混合比例。门控输出z ∈ [0,1]^H，每个维度独立
    控制两个模态的信息比例。

    核心公式:
        h1 = tanh(W1 * o1)         # 模态1的候选激活
        h2 = tanh(W2 * o2)         # 模态2的候选激活
        z  = sigmoid(Wz * [o1;o2]) # 门控信号
        fused = z * h1 + (1-z) * h2 # 门控加权混合

    这种设计允许模型在每个特征维度上动态决定更信任哪个模态。

参考论文:
    Arevalo et al., "Gated Multimodal Units for Information Fusion", ICLR 2017

输入:
    o1: [B, dim1]  - 病理图像特征
    o2: [B, dim2]  - 基因组特征

输出:
    fused: [B, output_dim]  - 融合后的多模态特征

使用示例:
    from src.networks.fusion_gmu import GMUFusion
    fusion = GMUFusion(dim1=32, dim2=32, hidden_dim=64, output_dim=64)
    fused = fusion(path_features, omic_features)
============================================================================
"""

import torch
import torch.nn as nn


class GMUFusion(nn.Module):
    """
    门控多模态单元 (GMU) 融合。

    通过sigmoid门控自适应混合两个模态的信息，
    每个特征维度独立决策两个模态的混合比例。

    参数:
        dim1: 模态1的特征维度 (默认32)
        dim2: 模态2的特征维度 (默认32)
        hidden_dim: 隐藏层维度 (默认64)
        output_dim: 输出维度 (默认64)
        dropout: Dropout比率 (默认0.25)
    """

    def __init__(self, dim1=32, dim2=32, hidden_dim=64, output_dim=64, dropout=0.25):
        super().__init__()
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)

        # 各模态的候选激活投影
        self.h1_proj = nn.Linear(self.dim1, self.hidden_dim)
        self.h2_proj = nn.Linear(self.dim2, self.hidden_dim)

        # 门控信号生成（基于两个模态的拼接）
        self.gate_proj = nn.Linear(self.dim1 + self.dim2, self.hidden_dim)

        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, o1, o2):
        """
        前向传播: 门控加权融合。

        参数:
            o1: [B, dim1] 病理图像特征
            o2: [B, dim2] 基因组特征

        返回:
            out: [B, output_dim] 融合后的多模态特征
        """
        if o1.dim() != 2 or o2.dim() != 2:
            raise ValueError(f"期望2D输入，但收到 {o1.shape}, {o2.shape}")

        # 维度适配
        d1 = min(o1.size(1), self.dim1)
        d2 = min(o2.size(1), self.dim2)
        x1 = o1[:, :d1]
        x2 = o2[:, :d2]

        if d1 < self.dim1:
            pad1 = torch.zeros(o1.size(0), self.dim1 - d1, device=o1.device, dtype=o1.dtype)
            x1 = torch.cat([x1, pad1], dim=1)
        if d2 < self.dim2:
            pad2 = torch.zeros(o2.size(0), self.dim2 - d2, device=o2.device, dtype=o2.dtype)
            x2 = torch.cat([x2, pad2], dim=1)

        # 候选激活: tanh非线性
        h1 = torch.tanh(self.h1_proj(x1))                       # [B, H]
        h2 = torch.tanh(self.h2_proj(x2))                       # [B, H]

        # 门控信号: sigmoid -> [0, 1]^H
        z = torch.sigmoid(self.gate_proj(torch.cat([x1, x2], dim=1)))  # [B, H]

        # 门控混合: z * h1 + (1-z) * h2
        fused = z * h1 + (1.0 - z) * h2

        return self.out_proj(fused)


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("GMUFusion 模块自测")
    B, D1, D2 = 8, 32, 32
    o1 = torch.randn(B, D1)
    o2 = torch.randn(B, D2)
    model = GMUFusion(dim1=D1, dim2=D2, hidden_dim=64, output_dim=64)
    out = model(o1, o2)
    print(f"  输入: o1={o1.shape}, o2={o2.shape}")
    print(f"  输出: fused={out.shape}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("测试通过!")
