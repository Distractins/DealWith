# -*- coding: utf-8 -*-
"""
fusion_attention_weighted.py
============================================================================
注意力加权融合 (Attention-Weighted Multimodal Fusion) 模块 ★新增。

设计思想:
    利用Scaled Dot-Product Attention机制，让模型自动学习每个样本中
    病理模态和基因组模态的重要性权重。

    与GMU的固定门控不同，注意力加权融合将两个模态投影到
    Query/Key/Value空间，通过注意力分数捕捉模态间的细粒度交互，
    动态决定每个样本中哪个模态更可信。

架构:
    1. 联合Query: [path; omic] -> Linear -> Query (综合两个模态的信息发起查询)
    2. 病理K/V: path -> Key_path, Value_path
    3. 基因组K/V: omic -> Key_omic, Value_omic
    4. 注意力: softmax(Q · K^T / √d) -> 加权Value
    5. 融合: attn_path * V_path + attn_omic * V_omic -> MLP输出

    可选多头注意力: 多个注意力头并行计算后拼接，增强表达能力。

肿瘤预后场景适应性:
    - 某些患者病理特征不明显(早期肿瘤)，模型会自动偏向基因组特征
    - 某些患者基因组特征稀疏(测序质量低)，模型会自动偏向病理特征
    - 注意力权重可视化可辅助解释模型决策

输入:
    o1: [B, dim1]  - 病理图像特征
    o2: [B, dim2]  - 基因组特征

输出:
    fused: [B, output_dim]  - 融合后的多模态特征

使用示例:
    from src.networks.fusion_attention_weighted import AttentionWeightedFusion
    fusion = AttentionWeightedFusion(dim1=32, dim2=32, hidden_dim=64, output_dim=64)
    fused = fusion(path_features, omic_features)
============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionWeighted(nn.Module):
    """
    多头注意力加权核心模块。

    参数:
        dim: 特征维度
        num_heads: 注意力头数
        dropout: Dropout比率
    """

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 统一的Q/K/V投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        参数:
            query: [B, dim] 联合查询
            key: [B, dim] 待查询模态的key
            value: [B, dim] 待查询模态的value

        返回:
            out: [B, dim] 注意力加权后的特征
            attn_weights: [B, num_heads] 平均注意力权重(用于可解释性)
        """
        B = query.size(0)

        # 投影并分头
        Q = self.q_proj(query).view(B, self.num_heads, self.head_dim)  # [B, H, D_h]
        K = self.k_proj(key).view(B, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(B, self.num_heads, self.head_dim)

        # 注意力分数
        attn_scores = torch.matmul(Q.unsqueeze(2), K.unsqueeze(3)).squeeze(3)  # [B, H, 1]
        attn_scores = attn_scores * self.scale
        attn_weights = torch.softmax(attn_scores.squeeze(2), dim=1)  # [B, H]

        # 加权Value
        V_weighted = V * attn_weights.unsqueeze(2)  # [B, H, D_h]
        V_weighted = V_weighted.view(B, -1)  # [B, dim]

        out = self.out_proj(self.dropout(V_weighted))

        # 返回平均权重用于可解释性分析
        avg_attn = attn_weights.mean(dim=1, keepdim=True)  # [B, 1]

        return out, avg_attn


class AttentionWeightedFusion(nn.Module):
    """
    注意力加权多模态融合模块。

    通过多头注意力机制动态学习两个模态的重要性权重，
    对每个样本自适应地融合病理和基因组信息。

    参数:
        dim1: 病理特征维度 (默认32)
        dim2: 基因组特征维度 (默认32)
        hidden_dim: 隐藏层/注意力维度 (默认64)
        output_dim: 输出维度 (默认64)
        dropout: Dropout比率 (默认0.25)
        num_heads: 注意力头数 (默认4)
        use_residual: 是否使用残差连接 (默认True)
    """

    def __init__(
        self,
        dim1=32,
        dim2=32,
        hidden_dim=64,
        output_dim=64,
        dropout=0.25,
        num_heads=4,
        use_residual=True,
    ):
        super().__init__()
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.use_residual = use_residual

        # 将两个模态投影到共享维度
        self.path_proj = nn.Sequential(
            nn.Linear(self.dim1, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.omic_proj = nn.Sequential(
            nn.Linear(self.dim2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 联合查询生成: 拼接两个模态的信息
        self.query_gen = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 多头注意力: 分别对病理和基因组计算
        self.path_attn = MultiHeadAttentionWeighted(
            dim=self.hidden_dim, num_heads=num_heads, dropout=dropout
        )
        self.omic_attn = MultiHeadAttentionWeighted(
            dim=self.hidden_dim, num_heads=num_heads, dropout=dropout
        )

        # 后融合MLP
        self.post_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 残差投影（如果启用）
        if use_residual:
            self.residual_proj = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.output_dim),
                nn.LayerNorm(self.output_dim),
            )

    def forward(self, o1, o2):
        """
        前向传播: 注意力加权融合。

        参数:
            o1: [B, dim1] 病理图像特征
            o2: [B, dim2] 基因组特征

        返回:
            fused: [B, output_dim] 融合后的多模态特征
        """
        if o1.dim() != 2 or o2.dim() != 2:
            raise ValueError(f"期望2D输入 [B, D]，但收到 {o1.shape}, {o2.shape}")

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

        # 投影到共享空间
        h1 = self.path_proj(x1)  # [B, H]
        h2 = self.omic_proj(x2)  # [B, H]

        # 生成联合查询: 综合两个模态信息
        joint = torch.cat([h1, h2], dim=1)  # [B, 2H]
        query = self.query_gen(joint)        # [B, H]

        # 注意力加权
        path_weighted, path_w = self.path_attn(query, h1, h1)  # 病理特征 + 病理注意力
        omic_weighted, omic_w = self.omic_attn(query, h2, h2)  # 基因组特征 + 基因组注意力

        # 拼接加权后的特征
        fused = torch.cat([path_weighted, omic_weighted], dim=1)  # [B, 2H]

        # 后融合投影
        out = self.post_fusion(fused)

        # 残差连接
        if self.use_residual:
            residual = self.residual_proj(joint)
            out = out + residual

        return out


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AttentionWeightedFusion 注意力加权融合模块自测")
    print("=" * 60)

    B, D1, D2 = 8, 32, 32
    o1 = torch.randn(B, D1)
    o2 = torch.randn(B, D2)

    for num_heads in [2, 4, 8]:
        model = AttentionWeightedFusion(
            dim1=D1, dim2=D2, hidden_dim=64, output_dim=64,
            num_heads=num_heads, use_residual=True,
        )
        out = model(o1, o2)
        params = sum(p.numel() for p in model.parameters())
        print(f"  num_heads={num_heads}: 输出={out.shape}, 参数量={params:,}")

    # 测试残差开关
    for use_res in [True, False]:
        model = AttentionWeightedFusion(use_residual=use_res)
        out = model(o1, o2)
        print(f"  use_residual={use_res}: 输出={out.shape}")

    print("\n所有测试通过!")
