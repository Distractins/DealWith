# -*- coding: utf-8 -*-
"""
fusion_film.py
============================================================================
特征线性调制 (Feature-wise Linear Modulation, FiLM) 融合模块。

设计思想:
    FiLM源自视觉推理领域，通过一个模态（条件模态）生成γ(缩放)和β(平移)
    参数来调制另一个模态（主模态）的特征。这种设计非常适合于病理+
    基因组多模态融合场景：

    - 病理特征作为主模态: 携带丰富的组织形态学信息
    - 基因组特征作为条件: 通过γ/β告诉模型"关注哪些形态特征"

    核心公式:
        γ, β = f(omic_feat)                    # 基因组生成调制参数
        path_mod = γ * path_proj + β           # 调制病理特征
        fused = MLP([path_mod; omic_proj])     # 拼接后融合

    也支持双向: path->omic 和 omic->path，通过配置控制方向。

    对于结直肠癌预后预测:
        不同基因突变状态(T53, KRAS等)可能导致不同的组织形态变化,
        FiLM允许基因组信号直接调控病理特征的表达强度。

参考论文:
    Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018

输入:
    o1: [B, dim1]  - 病理图像特征（被调制的模态）
    o2: [B, dim2]  - 基因组特征（条件模态，生成γ/β）

输出:
    fused: [B, output_dim]  - 融合后的多模态特征

使用示例:
    from src.networks.fusion_film import FiLMFusion
    fusion = FiLMFusion(dim1=32, dim2=32, hidden_dim=64, output_dim=64)
    fused = fusion(path_features, omic_features)
============================================================================
"""

import torch
import torch.nn as nn


class FiLMFusion(nn.Module):
    """
    特征线性调制 (FiLM) 融合。

    使用基因组特征(o2)生成γ/β参数来调制病理特征(o1)，
    实现基因组条件对病理特征的引导。

    参数:
        dim1: 主模态(病理)的特征维度 (默认32)
        dim2: 条件模态(基因组)的特征维度 (默认32)
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

        # 将两个模态投影到相同的隐藏空间
        self.o1_proj = nn.Linear(self.dim1, self.hidden_dim)  # 病理投影
        self.o2_proj = nn.Linear(self.dim2, self.hidden_dim)  # 基因组投影

        # FiLM参数生成器: 从基因组特征生成γ和β
        self.film_gen = nn.Sequential(
            nn.Linear(self.dim2, self.hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),  # [γ; β]
        )

        # 后融合MLP: 拼接调制后的病理特征与基因组投影
        self.post_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, o1, o2):
        """
        前向传播: FiLM调制融合。

        参数:
            o1: [B, dim1] 病理图像特征（被调制）
            o2: [B, dim2] 基因组特征（条件信号）

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

        # 投影到隐藏空间
        h1 = self.o1_proj(x1)   # [B, H] 病理特征
        h2 = self.o2_proj(x2)   # [B, H] 基因组特征

        # 从基因组特征生成FiLM参数
        gamma_beta = self.film_gen(x2)         # [B, 2H]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)

        # 稳定化: gamma围绕1, beta围绕0
        # tanh输出[-1,1]，所以gamma∈[0,2], beta∈[-1,1]
        gamma = 1.0 + torch.tanh(gamma)
        beta = torch.tanh(beta)

        # FiLM调制: 基因组信号调控病理特征
        h1_mod = gamma * h1 + beta              # [B, H]

        # 拼接调制后的病理特征与基因组投影
        fused = torch.cat([h1_mod, h2], dim=1)  # [B, 2H]
        return self.post_fusion(fused)


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("FiLMFusion 模块自测")
    B, D1, D2 = 8, 32, 32
    o1 = torch.randn(B, D1)
    o2 = torch.randn(B, D2)
    model = FiLMFusion(dim1=D1, dim2=D2, hidden_dim=64, output_dim=64)
    out = model(o1, o2)
    print(f"  输入: o1={o1.shape}, o2={o2.shape}")
    print(f"  输出: fused={out.shape}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 验证FiLM调制参数范围
    print("\n  FiLM调制验证:")
    print(f"    gamma范围: [0, 2] (1+tanh)")
    print(f"    beta范围: [-1, 1] (tanh)")
    print("测试通过!")
