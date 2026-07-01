# -*- coding: utf-8 -*-
"""
fusion_tensor_concat.py
============================================================================
多层级特征图拼接融合 (Tensor Concat Fusion) 模块 ★新增。

设计思想:
    区别于简单拼接融合(ConcatFusion)仅在输入层级做一次拼接，
    本模块将两个模态的特征分别通过多层MLP生成浅/中/深三层表示，
    在每一层独立进行跨模态拼接融合，最后将所有层级的融合结果
    进行加权聚合。

    这种设计保留了不同抽象层次的信息:
    - 浅层: 原始特征的直接交互 (细粒度)
    - 中层: 经过一定变换后的特征交互 (中等粒度)
    - 深层: 高度抽象后的特征交互 (粗粒度)

    类似于FPN(Feature Pyramid Network)的思想，但用于多模态融合。

架构:
    1. 分层特征生成: path/omic各自通过3层投影生成L0/L1/L2
    2. 层级内融合: 每个层级独立拼接+MLP
    3. 层级间聚合: 可学习权重聚合三个层级的融合结果
    4. 残差连接: 原始拼接特征作为保底通道

肿瘤预后场景适应性:
    - 浅层: 基因突变与patch纹理的直接关联
    - 中层: 基因通路与组织架构的关联
    - 深层: 分子亚型与整体形态模式的关联
    - 多层级信息互补防止单一层级信息丢失

输入:
    o1: [B, dim1]  - 病理图像特征
    o2: [B, dim2]  - 基因组特征

输出:
    fused: [B, output_dim]  - 融合后的多模态特征

使用示例:
    from src.networks.fusion_tensor_concat import TensorConcatFusion
    fusion = TensorConcatFusion(dim1=32, dim2=32, hidden_dim=64, output_dim=64)
    fused = fusion(path_features, omic_features)
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TensorConcatFusion(nn.Module):
    """
    多层级特征图拼接融合模块。

    通过浅/中/深三层分别融合后加权聚合，
    保留不同抽象层次的多模态交互信息。

    参数:
        dim1: 病理特征维度 (默认32)
        dim2: 基因组特征维度 (默认32)
        hidden_dim: 隐藏层维度 (默认64)
        output_dim: 输出维度 (默认64)
        dropout: Dropout比率 (默认0.25)
        num_levels: 层级数量 (默认3: 浅/中/深)
    """

    def __init__(
        self,
        dim1=32,
        dim2=32,
        hidden_dim=64,
        output_dim=64,
        dropout=0.25,
        num_levels=3,
    ):
        super().__init__()
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_levels = int(num_levels)

        # ---- 分层特征生成器 ----
        # 每层有独立的path投影和omic投影
        self.path_levels = nn.ModuleList()
        self.omic_levels = nn.ModuleList()

        # 渐进式的特征变换: 越深层的变换越复杂
        for i in range(self.num_levels):
            # 病理特征投影
            self.path_levels.append(nn.Sequential(
                nn.Linear(self.dim1 if i == 0 else self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ))
            # 基因组特征投影
            self.omic_levels.append(nn.Sequential(
                nn.Linear(self.dim2 if i == 0 else self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ))

        # ---- 层级内融合模块 ----
        # 每一层将拼接后的 [path_level; omic_level] 融合为hidden_dim维
        self.level_fusions = nn.ModuleList()
        for i in range(self.num_levels):
            self.level_fusions.append(nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ))

        # ---- 层级聚合权重 ----
        # 可学习的层级重要性权重
        self.level_weights = nn.Parameter(
            torch.ones(self.num_levels) / self.num_levels
        )

        # ---- 全局聚合MLP ----
        # 将所有层级的融合结果与原始拼接特征聚合
        self.global_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_levels, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ---- 残差保底通道 ----
        # 直接拼接原始特征作为保底信号
        self.residual_proj = nn.Sequential(
            nn.Linear(self.dim1 + self.dim2, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

        # 残差融合权重
        self.main_scale = nn.Parameter(torch.tensor(0.8))
        self.res_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, o1, o2):
        """
        前向传播: 多层级特征图拼接融合。

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

        # 残差: 直接拼接原始特征
        residual = self.residual_proj(torch.cat([x1, x2], dim=1))

        # ---- 逐层生成特征并融合 ----
        level_outputs = []
        p_curr, g_curr = x1, x2

        for i in range(self.num_levels):
            # 投影到当前层级
            p_level = self.path_levels[i](p_curr)  # [B, H]
            g_level = self.omic_levels[i](g_curr)  # [B, H]

            # 层级内融合
            level_concat = torch.cat([p_level, g_level], dim=1)  # [B, 2H]
            level_fused = self.level_fusions[i](level_concat)    # [B, H]

            level_outputs.append(level_fused)

            # 传递到下一层（渐进变换）
            p_curr = p_level
            g_curr = g_level

        # ---- 层级加权聚合 ----
        # softmax归一化层级权重
        level_w = torch.softmax(self.level_weights, dim=0)  # [num_levels]

        # 加权求和
        all_levels = torch.stack(level_outputs, dim=1)  # [B, num_levels, H]
        weighted = all_levels * level_w.view(1, -1, 1)  # [B, num_levels, H]
        fused_levels = weighted.view(all_levels.size(0), -1)  # [B, num_levels*H]

        # 全局聚合
        fused_main = self.global_fusion(fused_levels)

        # 残差融合
        fused = self.main_scale * fused_main + self.res_scale * residual

        return fused


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TensorConcatFusion 多层级特征图拼接融合模块自测")
    print("=" * 60)

    B, D1, D2 = 8, 32, 32
    o1 = torch.randn(B, D1)
    o2 = torch.randn(B, D2)

    for num_levels in [2, 3, 4]:
        for hidden_dim in [32, 64]:
            model = TensorConcatFusion(
                dim1=D1, dim2=D2, hidden_dim=hidden_dim,
                output_dim=64, num_levels=num_levels,
            )
            out = model(o1, o2)
            params = sum(p.numel() for p in model.parameters())
            print(f"  levels={num_levels}, hidden={hidden_dim}: "
                  f"输出={out.shape}, 参数量={params:,}")

    # 验证层级权重
    model = TensorConcatFusion(num_levels=3)
    out = model(o1, o2)
    print(f"\n  层级权重: {torch.softmax(model.level_weights, dim=0).tolist()}")
    print(f"  主/残比例: main={model.main_scale.item():.3f}, "
          f"res={model.res_scale.item():.3f}")

    print("\n所有测试通过!")
