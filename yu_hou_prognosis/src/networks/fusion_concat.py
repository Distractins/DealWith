# -*- coding: utf-8 -*-
"""
fusion_concat.py
============================================================================
简单拼接融合 (ConcatFusion) 基线模块。

设计思想:
    最基础的多模态融合策略：将两个模态的特征向量直接拼接，
    然后通过MLP投影到目标维度。作为融合策略对比实验的基线。

    虽然简单，但拼接融合在许多多模态任务中仍有竞争力，
    特别是当两个模态特征维度较低且已经过良好的单模态预训练时。

架构:
    concat([o1; o2]) -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout

输入:
    o1: [B, dim1]  - 病理图像特征
    o2: [B, dim2]  - 基因组特征

输出:
    fused: [B, output_dim]  - 融合后的多模态特征

使用示例:
    from src.networks.fusion_concat import ConcatFusion
    fusion = ConcatFusion(dim1=32, dim2=32, hidden_dim=64, output_dim=64)
    fused = fusion(path_features, omic_features)
============================================================================
"""

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """
    简单拼接融合基线。

    将两个模态特征直接拼接后通过两层MLP投影。

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
        self.output_dim = int(output_dim)

        # 拼接后通过两层MLP投影
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.dim1 + self.dim2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, o1, o2):
        """
        前向传播: 拼接 + MLP投影。

        自动处理维度不匹配：输入大于配置维度时裁切，不足时补零。
        """
        if o1.dim() != 2 or o2.dim() != 2:
            raise ValueError(f"期望2D输入，但收到 {o1.shape}, {o2.shape}")

        # 维度适配
        d1 = min(o1.size(1), self.dim1)
        d2 = min(o2.size(1), self.dim2)

        x1 = o1[:, :d1]
        x2 = o2[:, :d2]

        # 不足时补零
        if d1 < self.dim1:
            pad1 = torch.zeros(o1.size(0), self.dim1 - d1, device=o1.device, dtype=o1.dtype)
            x1 = torch.cat([x1, pad1], dim=1)
        if d2 < self.dim2:
            pad2 = torch.zeros(o2.size(0), self.dim2 - d2, device=o2.device, dtype=o2.dtype)
            x2 = torch.cat([x2, pad2], dim=1)

        fusion = torch.cat([x1, x2], dim=1)  # [B, dim1+dim2]
        return self.fusion_mlp(fusion)


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("ConcatFusion 模块自测")
    B, D1, D2 = 8, 32, 32
    o1 = torch.randn(B, D1)
    o2 = torch.randn(B, D2)
    model = ConcatFusion(dim1=D1, dim2=D2, hidden_dim=64, output_dim=64)
    out = model(o1, o2)
    print(f"  输入: o1={o1.shape}, o2={o2.shape}")
    print(f"  输出: fused={out.shape}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("测试通过!")
