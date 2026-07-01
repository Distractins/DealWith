# -*- coding: utf-8 -*-
"""
fusion_cross_attention.py
============================================================================
交叉注意力多模态融合 (Cross-Attention Multimodal Fusion) 模块 ★新增。

设计思想:
    受到Transformer架构的启发，使用交叉注意力(Cross-Attention)实现
    病理和基因组两个模态之间的双向信息交互。

    - 病理图像作为Query查询基因组Key/Value: "哪些基因突变与当前形态有关?"
    - 基因组特征作为Query查询病理Key/Value: "哪些形态特征与当前突变有关?"

    双向交叉注意力后拼接，实现模态间深度信息交互。

架构:
    1. 模态投影: path/omic -> hidden_dim 共享空间
    2. 自注意力增强: 每个模态内部self-attention
    3. 交叉注意力: path(query) x omic(key/value) 双向
    4. 前馈网络: FFN(交叉注意力输出)
    5. 残差连接 + LayerNorm: 每层都有

    支持多层堆叠以增加交互深度。

肿瘤预后场景适应性:
    - 结直肠癌中特定基因突变(KRAS/BRAF/TP53)与组织形态密切相关
    - 交叉注意力允许模型显式学习基因-形态的对应关系
    - 多层堆叠可实现渐进式的跨模态推理

输入:
    o1: [B, dim1]  - 病理图像特征
    o2: [B, dim2]  - 基因组特征

输出:
    fused: [B, output_dim]  - 融合后的多模态特征

使用示例:
    from src.networks.fusion_cross_attention import CrossAttentionFusion
    fusion = CrossAttentionFusion(dim1=32, dim2=32, hidden_dim=64, output_dim=64,
                                   num_heads=4, num_layers=2)
    fused = fusion(path_features, omic_features)
============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    """
    单层交叉注意力模块。

    结构:
        Self-Attention(path) + Cross-Attention(path->omic) -> FFN
        Self-Attention(omic) + Cross-Attention(omic->path) -> FFN

    参数:
        dim: 特征维度
        num_heads: 注意力头数
        ff_expansion: 前馈网络扩展比例
        dropout: Dropout比率
    """

    def __init__(self, dim, num_heads=4, ff_expansion=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # ---- 病理分支 ----
        # Self-Attention (病理内部交互)
        self.path_self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.path_self_norm = nn.LayerNorm(dim)

        # Cross-Attention (病理->基因组)
        self.path_cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.path_cross_norm = nn.LayerNorm(dim)

        # Path FFN
        self.path_ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_expansion, dim),
            nn.Dropout(dropout),
        )
        self.path_ffn_norm = nn.LayerNorm(dim)

        # ---- 基因组分支 ----
        # Self-Attention (基因组内部交互)
        self.omic_self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.omic_self_norm = nn.LayerNorm(dim)

        # Cross-Attention (基因组->病理)
        self.omic_cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.omic_cross_norm = nn.LayerNorm(dim)

        # Omic FFN
        self.omic_ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_expansion, dim),
            nn.Dropout(dropout),
        )
        self.omic_ffn_norm = nn.LayerNorm(dim)

    def forward(self, path_feat, omic_feat):
        """
        参数:
            path_feat: [B, dim] 病理特征
            omic_feat: [B, dim] 基因组特征

        返回:
            path_out: [B, dim] 更新后的病理特征
            omic_out: [B, dim] 更新后的基因组特征
        """
        # 添加序列维度（MultiheadAttention需要[B, S, D]）
        p = path_feat.unsqueeze(1)  # [B, 1, dim]
        g = omic_feat.unsqueeze(1)  # [B, 1, dim]

        # ---- 病理分支更新 ----
        # Self-Attention
        p_self, _ = self.path_self_attn(p, p, p)
        p = self.path_self_norm(p + p_self)

        # Cross-Attention: path(query) -> omic(key/value)
        p_cross, _ = self.path_cross_attn(p, g, g)
        p = self.path_cross_norm(p + p_cross)

        # FFN
        p = self.path_ffn_norm(p + self.path_ffn(p))

        # ---- 基因组分支更新 ----
        # Self-Attention
        g_self, _ = self.omic_self_attn(g, g, g)
        g = self.omic_self_norm(g + g_self)

        # Cross-Attention: omic(query) -> path(key/value)
        g_cross, _ = self.omic_cross_attn(g, p, p)  # 使用更新后的p
        g = self.omic_cross_norm(g + g_cross)

        # FFN
        g = self.omic_ffn_norm(g + self.omic_ffn(g))

        # 移除序列维度
        return p.squeeze(1), g.squeeze(1)


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力多模态融合模块。

    通过多层交叉注意力实现病理与基因组特征的深度双向信息交互。

    参数:
        dim1: 病理特征维度 (默认32)
        dim2: 基因组特征维度 (默认32)
        hidden_dim: 隐藏层/注意力维度 (默认64)
        output_dim: 输出维度 (默认64)
        dropout: Dropout比率 (默认0.25)
        num_heads: 注意力头数 (默认4)
        num_layers: 交叉注意力层数 (默认2)
        ff_expansion: FFN扩展比例 (默认4)
    """

    def __init__(
        self,
        dim1=32,
        dim2=32,
        hidden_dim=64,
        output_dim=64,
        dropout=0.25,
        num_heads=4,
        num_layers=2,
        ff_expansion=4,
    ):
        super().__init__()
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)

        # 将两个模态投影到共享维度
        self.path_proj = nn.Sequential(
            nn.Linear(self.dim1, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.omic_proj = nn.Sequential(
            nn.Linear(self.dim2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # 堆叠多层交叉注意力
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(
                dim=self.hidden_dim,
                num_heads=num_heads,
                ff_expansion=ff_expansion,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # 最终融合MLP
        self.post_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, o1, o2):
        """
        前向传播: 多层交叉注意力融合。

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
        p = self.path_proj(x1)  # [B, H]
        g = self.omic_proj(x2)  # [B, H]

        # 逐层交叉注意力
        for layer in self.cross_attn_layers:
            p, g = layer(p, g)

        # 拼接最终的双向特征
        fused = torch.cat([p, g], dim=1)  # [B, 2H]

        return self.post_fusion(fused)


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CrossAttentionFusion 交叉注意力融合模块自测")
    print("=" * 60)

    B, D1, D2 = 8, 32, 32
    o1 = torch.randn(B, D1)
    o2 = torch.randn(B, D2)

    for num_layers in [1, 2, 4]:
        for num_heads in [2, 4]:
            model = CrossAttentionFusion(
                dim1=D1, dim2=D2, hidden_dim=64, output_dim=64,
                num_heads=num_heads, num_layers=num_layers,
            )
            out = model(o1, o2)
            params = sum(p.numel() for p in model.parameters())
            print(f"  layers={num_layers}, heads={num_heads}: "
                  f"输出={out.shape}, 参数量={params:,}")

    print("\n所有测试通过!")
