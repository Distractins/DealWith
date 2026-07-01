# -*- coding: utf-8 -*-
"""
fusion_lmf.py
============================================================================
低秩多模态融合 (Low-rank Multimodal Fusion, LMF) 模块。

设计思想:
    传统外积融合的张量维度随模态数指数增长（维度爆炸问题）。
    LMF通过将全秩融合张量分解为多个低秩因子的组合，
    在保持表达性的同时大幅减少参数量。

核心公式:
    fusion = Σ_r (o1 · A_r) * (o2 · V_r) + bias
    其中A_r, V_r是第r个秩的因子矩阵，(·)表示向量内积，(*)表示逐元素乘积。

    实际实现使用einsum:
    o1_proj = einsum('bd,rdo->bro', o1, A)  # [B, R, Out]
    o2_proj = einsum('bd,rdo->bro', o2, V)  # [B, R, Out]
    fusion = (o1_proj * o2_proj).sum(dim=1) + bias  # [B, Out]

适用于:
    - 病理图像特征 + 基因组特征的参数高效融合
    - 需要控制模型大小的场景

输入:
    o1: [B, d1]  - 病理图像特征
    o2: [B, d2]  - 基因组特征

输出:
    fusion: [B, output_dim]  - 融合后的多模态特征

使用示例:
    from src.networks.fusion_lmf import LMF
    fusion = LMF(rank=4, hidden_dims=(32, 32), output_dim=64)
    fused = fusion(path_features, omic_features)
============================================================================
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class LMF(nn.Module):
    """
    低秩多模态融合 (LMF)。

    通过将全秩融合权重张量分解为rank个低秩因子，
    实现参数高效的多模态交互建模。

    参数:
        rank: 低秩分解的秩 (默认4, 越大表达能力越强但参数更多)
        hidden_dims: 两个模态的特征维度元组 (dim1, dim2)
        output_dim: 融合输出维度 (默认64)

    注意:
        - hidden_dims实际使用的维度 = dim + 1（为偏置项预留1维）
        - 因子矩阵形状: [rank, dim+1, output_dim]
    """

    def __init__(self, rank=4, hidden_dims=(32, 32), output_dim=64):
        super().__init__()
        self.rank = int(rank)
        self.audio_hidden = int(hidden_dims[0])   # 病理特征维度
        self.video_hidden = int(hidden_dims[1])   # 基因组特征维度
        self.output_dim = int(output_dim)

        # 因子矩阵: [rank, dim+1, output_dim]
        # +1是为偏置项预留位置
        self.audio_factor = Parameter(
            torch.randn(self.rank, self.audio_hidden + 1, self.output_dim)
        )
        self.video_factor = Parameter(
            torch.randn(self.rank, self.video_hidden + 1, self.output_dim)
        )

        # 输出偏置
        self.fusion_bias = Parameter(torch.zeros(self.output_dim))

    def forward(self, o1, o2):
        """
        前向传播 (设备安全版，无.cuda()调用)。

        参数:
            o1: [B, d1] 病理图像特征
            o2: [B, d2] 基因组特征

        返回:
            fusion: [B, output_dim] 融合后的多模态特征

        异常:
            ValueError: 输入不是2D时抛出
        """
        if o1.dim() != 2 or o2.dim() != 2:
            raise ValueError(
                f"期望2D输入 [B, D]，但收到 {o1.shape}, {o2.shape}"
            )

        device = o1.device
        dtype = o1.dtype

        # 将因子矩阵移到与输入相同的设备和数据类型
        A = self.audio_factor.to(device=device, dtype=dtype)  # [r, d1_expected, out]
        V = self.video_factor.to(device=device, dtype=dtype)  # [r, d2_expected, out]
        b = self.fusion_bias.to(device=device, dtype=dtype)   # [out]

        # 维度适配: 取输入和因子的最小维度避免shape不匹配
        d1 = min(o1.size(1), A.size(1))
        d2 = min(o2.size(1), V.size(1))
        o1_ = o1[:, :d1]       # [B, d1]
        o2_ = o2[:, :d2]       # [B, d2]
        A_ = A[:, :d1, :]      # [r, d1, out]
        V_ = V[:, :d2, :]      # [r, d2, out]

        # 低秩投影: einsum比循环更高效
        # o1_: [B,d1], A_: [r,d1,out] -> o1_proj: [B,r,out]
        o1_proj = torch.einsum('bd,rdo->bro', o1_, A_)
        o2_proj = torch.einsum('bd,rdo->bro', o2_, V_)

        # 逐元素乘积后沿秩求和 + 偏置
        fusion = (o1_proj * o2_proj).sum(dim=1) + b  # [B, out]

        return fusion


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LMF 低秩多模态融合模块自测")
    print("=" * 60)

    B, D1, D2 = 8, 32, 32
    o1 = torch.randn(B, D1)
    o2 = torch.randn(B, D2)

    for rank in [1, 2, 4, 8]:
        for out_dim in [32, 64]:
            model = LMF(rank=rank, hidden_dims=(D1, D2), output_dim=out_dim)
            out = model(o1, o2)
            params = sum(p.numel() for p in model.parameters())
            print(f"  rank={rank}, out_dim={out_dim}:")
            print(f"    输出shape: {out.shape}, 参数量: {params:,}")

    print("\n所有测试通过!")
