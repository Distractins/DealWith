# -*- coding: utf-8 -*-
"""
caugf.py - VectorCAUGF 多模态融合模块 (优化版)
============================================================================
核心创新模块: 三流自适应关系-残差融合 (VectorCAUGF)

设计思想:
    结直肠癌预后预测中，病理图像特征（组织形态）和基因组特征（分子突变）
    具有本质不同的信息内涵。本模块通过三流并行架构实现深度多模态融合:

    1. 病理特征流 (Path Stream):   保留病理形态学信息
    2. 基因组特征流 (Omic Stream): 保留分子突变信息
    3. 关系流 (Relation Stream):   显式建模模态间交互 (乘积/差异/余弦相似度)

    通过流级softmax自适应加权，模型可以针对每个样本动态调整三个流的
    重要性，避免某个模态在训练中被压制。

优化记录 (v2.0):
    [优化1] 取消dropout钳制，恢复用户对dropout的完全控制
    [优化2] 扩展relation类型: 新增cosine相似度，丰富模态关系建模
    [优化3] Sigmoid->Tanh门控: 缓解梯度消失，改善训练初期收敛
    [优化4] 添加LayerNorm: 每个projection后添加LN，稳定小batch训练(2-4)
    [优化5] 残差权重初始化修正: main_scale=res_scale=0.5初始平衡
    [优化6] 温度系数: stream权重softmax添加可学习temperature
    [优化7] 输出层可配置: post_fusion层数改为参数控制

输入:
    path_feat: [B, dim1]  - 病理ResNet50提取的32维特征
    omic_feat: [B, dim2]  - MLP编码器提取的32维基因组特征

输出:
    fused_feat: [B, output_dim]  - 融合后的64维多模态特征

使用示例:
    from src.networks.caugf import VectorCAUGF
    fusion = VectorCAUGF(dim1=32, dim2=32, hidden_dim=64, output_dim=64)
    fused = fusion(path_features, omic_features)
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorCAUGF(nn.Module):
    """
    三流自适应关系-残差融合模块 (优化版)

    架构流程:
        1. 单模态投影: path/omic特征分别投影到共享隐藏空间
        2. 关系建模: 通过乘积/差异/余弦相似度显式建模跨模态关系
        3. 流加权: 基于全局上下文的softmax自适应三流权重分配
        4. 特征校准: 门控机制温和校准各流特征
        5. 残差融合: 主分支(加权三流) + 残差分支(直接拼接投影)
        6. 输出投影: 多层MLP输出最终融合特征

    参数:
        dim1: 病理特征维度 (默认32)
        dim2: 基因组特征维度 (默认32)
        hidden_dim: 隐藏层维度 (默认64)
        output_dim: 输出维度 (默认64)
        dropout: Dropout比率 (默认0.25, 直接使用不再钳制)
        relation_types: 关系类型列表，可选 "product","difference","cosine"
        num_post_layers: 输出MLP层数 (默认2)
        use_layer_norm: 是否使用LayerNorm (默认True, 推荐小batch开启)
        min_stream_weight: 流权重下限 (默认0.12)
        temperature_init: 权重温度系数初始值 (默认1.0)
    """

    def __init__(
        self,
        dim1=32,
        dim2=32,
        hidden_dim=64,
        output_dim=64,
        dropout=0.25,
        relation_types=("product", "difference", "cosine"),
        num_post_layers=2,
        use_layer_norm=True,
        min_stream_weight=0.12,
        temperature_init=1.0,
    ):
        super().__init__()

        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)

        # [优化1] 直接使用外部dropout，不做钳制
        self.dropout_rate = float(dropout)

        # [优化2] 关系类型配置
        self.relation_types = list(relation_types)
        # 计算关系特征的维度
        self._n_relations = len(self.relation_types)
        # 每种关系产生hidden_dim维特征
        self._rel_input_dim = self.hidden_dim * self._n_relations

        # ============================================================
        # [优化4] LayerNorm层（在所有projection后使用，稳定训练）
        # ============================================================
        self._maybe_norm = lambda dim: nn.LayerNorm(dim) if use_layer_norm else nn.Identity()

        # ============================================================
        # 1) 单模态投影
        # ============================================================
        self.path_proj = nn.Sequential(
            nn.Linear(self.dim1, self.hidden_dim),
            self._maybe_norm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
        )

        self.omic_proj = nn.Sequential(
            nn.Linear(self.dim2, self.hidden_dim),
            self._maybe_norm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
        )

        # ============================================================
        # 2) 关系流: 显式建模模态间交互
        # [优化2] 支持product/difference/cosine三种关系
        # ============================================================
        self.relation_proj = nn.Sequential(
            nn.Linear(self._rel_input_dim, self.hidden_dim),
            self._maybe_norm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
        )

        # ============================================================
        # 3) 流级自适应加权
        # 三路: path / omic / relation
        # [优化6] 添加可学习温度系数控制权重锐度
        # ============================================================
        self.context_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            self._maybe_norm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
        )
        self.weight_head = nn.Linear(self.hidden_dim, 3)

        # [优化6] 可学习温度系数
        self.log_temperature = nn.Parameter(
            torch.tensor(temperature_init).log()
        )

        # ============================================================
        # 4) 特征校准门控
        # [优化3] 使用Tanh替代Sigmoid，缓解梯度消失
        # ============================================================
        self.path_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.omic_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.rel_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )

        # ============================================================
        # 5) 残差保底通道
        # [优化5] 初始化main_scale=res_scale=0.5，初始平衡主/残差分支
        # ============================================================
        self.residual_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            self._maybe_norm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
        )

        # [优化5] 初始平衡主分支和残差分支
        self.main_scale = nn.Parameter(torch.tensor(0.5))
        self.res_scale = nn.Parameter(torch.tensor(0.5))

        # ============================================================
        # 6) 输出投影层
        # [优化7] 可配置输出层数量
        # ============================================================
        post_layers = []
        in_dim = self.hidden_dim
        for i in range(num_post_layers):
            post_layers.extend([
                nn.Linear(in_dim, self.output_dim),
                self._maybe_norm(self.output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
            ])
            in_dim = self.output_dim
        self.post_fusion = nn.Sequential(*post_layers)

        # 防止某一路权重完全塌缩为0
        self.min_stream_weight = float(min_stream_weight)

    def _safe_pad_or_crop(self, x, target_dim):
        """
        安全的维度适配：将输入x的最后一维裁切或填充至target_dim。
        如果输入维度不足，用0填充；如果超出，取前target_dim个元素。
        """
        d = min(x.size(1), target_dim)
        out = x[:, :d]
        if d < target_dim:
            pad = torch.zeros(
                out.size(0), target_dim - d,
                device=out.device, dtype=out.dtype,
            )
            out = torch.cat([out, pad], dim=1)
        return out

    def _compute_relations(self, p, g):
        """
        [优化2] 计算多种模态间关系特征。

        支持的关系类型:
            - product:    p * g (逐元素乘积，捕获交互强度)
            - difference: |p - g| (绝对差，捕获模态差异)
            - cosine:     cos_sim(p, g) (余弦相似度，捕获方向一致性)

        参数:
            p: [B, H] 病理投影特征
            g: [B, H] 基因组投影特征

        返回:
            rel_feats: [B, H * n_relations] 拼接的关系特征
        """
        rel_list = []

        for rel_type in self.relation_types:
            if rel_type == "product":
                rel_list.append(p * g)
            elif rel_type == "difference":
                rel_list.append(torch.abs(p - g))
            elif rel_type == "cosine":
                # 余弦相似度: 逐元素计算后扩展为向量
                cos_sim = F.cosine_similarity(p, g, dim=1, eps=1e-8)
                rel_list.append(cos_sim.unsqueeze(1).expand(-1, self.hidden_dim))
            else:
                raise ValueError(f"不支持的关系类型: {rel_type}")

        return torch.cat(rel_list, dim=1)  # [B, H * n_relations]

    def forward(self, path_feat, omic_feat):
        """
        前向传播。

        参数:
            path_feat: [B, dim1] 病理图像特征
            omic_feat: [B, dim2] 基因组特征

        返回:
            fused_feat: [B, output_dim] 融合后的多模态特征

        异常:
            ValueError: 输入张量不是2D时抛出
        """
        if path_feat.dim() != 2 or omic_feat.dim() != 2:
            raise ValueError(
                f"期望2D输入 [B, D]，但收到 {path_feat.shape}, {omic_feat.shape}"
            )

        # --------------------------------------------------
        # 0) 输入维度兼容（自动pad/crop到配置维度）
        # --------------------------------------------------
        p = self._safe_pad_or_crop(path_feat, self.dim1)
        g = self._safe_pad_or_crop(omic_feat, self.dim2)

        # --------------------------------------------------
        # 1) 单模态编码
        # --------------------------------------------------
        p = self.path_proj(p)   # [B, H]
        g = self.omic_proj(g)   # [B, H]

        # --------------------------------------------------
        # 2) 关系特征计算
        # [优化2] 支持多种关系类型的组合
        # --------------------------------------------------
        rel_raw = self._compute_relations(p, g)
        rel = self.relation_proj(rel_raw)  # [B, H]

        # --------------------------------------------------
        # 3) 全局上下文 -> 三路自适应权重
        # [优化6] 添加温度系数控制权重锐度
        # --------------------------------------------------
        joint = torch.cat([p, g], dim=1)           # [B, 2H]
        context = self.context_proj(joint)          # [B, H]

        # [优化6] 使用可学习温度
        temperature = torch.exp(self.log_temperature)
        alpha_logits = self.weight_head(context) / temperature  # [B, 3]
        alpha = torch.softmax(alpha_logits, dim=1)              # [B, 3]

        # 应用下限防止某一路权重完全塌缩
        floor = self.min_stream_weight
        alpha = floor + (1.0 - 3.0 * floor) * alpha
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)

        a_p = alpha[:, 0:1]  # 病理权重
        a_g = alpha[:, 1:2]  # 基因组权重
        a_r = alpha[:, 2:3]  # 关系权重

        # --------------------------------------------------
        # 4) 特征校准
        # [优化3] Tanh门控替代Sigmoid，梯度流动更均匀
        # --------------------------------------------------
        p_hat = p * self.path_gate(context)   # 病理特征校准
        g_hat = g * self.omic_gate(context)   # 基因组特征校准
        r_hat = rel * self.rel_gate(context)  # 关系特征校准

        # --------------------------------------------------
        # 5) 主融合分支: 三流加权和
        # --------------------------------------------------
        fused_main = a_p * p_hat + a_g * g_hat + a_r * r_hat

        # --------------------------------------------------
        # 6) 残差保底分支: 直接拼接投影
        # [优化5] 可学习权重平衡主/残差分支
        # --------------------------------------------------
        fused_res = self.residual_proj(joint)

        fused = self.main_scale * fused_main + self.res_scale * fused_res

        # --------------------------------------------------
        # 7) 输出投影
        # [优化7] 可配置层数的输出MLP
        # --------------------------------------------------
        fused = self.post_fusion(fused)

        return fused


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("VectorCAUGF 融合模块自测")
    print("=" * 60)

    # 测试参数
    B, D1, D2 = 8, 32, 32
    path_feat = torch.randn(B, D1)
    omic_feat = torch.randn(B, D2)

    # 测试各种relation组合
    for rel_types in [
        ["product", "difference"],
        ["product", "difference", "cosine"],
        ["cosine"],
    ]:
        model = VectorCAUGF(
            dim1=D1, dim2=D2, hidden_dim=64, output_dim=64,
            dropout=0.25, relation_types=rel_types,
            num_post_layers=2, use_layer_norm=True,
        )
        out = model(path_feat, omic_feat)
        print(f"  relation_types={rel_types}")
        print(f"    输入: path={path_feat.shape}, omic={omic_feat.shape}")
        print(f"    输出: fused={out.shape}")
        print(f"    总参数量: {sum(p.numel() for p in model.parameters()):,}")
        print()

    # 测试LayerNorm开关
    for use_ln in [True, False]:
        model = VectorCAUGF(use_layer_norm=use_ln)
        out = model(path_feat, omic_feat)
        print(f"  use_layer_norm={use_ln}, 输出shape: {out.shape}")

    # 测试维度不匹配时的自动处理
    path_feat_odd = torch.randn(B, 40)   # 大于配置的32
    omic_feat_odd = torch.randn(B, 20)   # 小于配置的32
    model = VectorCAUGF()
    out = model(path_feat_odd, omic_feat_odd)
    print(f"\n  维度不匹配测试:")
    print(f"    输入: path={path_feat_odd.shape}, omic={omic_feat_odd.shape}")
    print(f"    输出: fused={out.shape}")

    print("\n所有测试通过!")
