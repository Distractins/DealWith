# -*- coding: utf-8 -*-
"""
fusion_bilinear.py
============================================================================
门控双线性融合 (BilinearFusion / POFusion) 模块。

设计思想:
    通过外积 (outer product) 捕获两个模态特征之间的全部二阶交互，
    使用门控机制 (sigmoid gating) 对各模态进行自适应信息过滤，
    最后通过MLP将高维外积特征压缩到融合输出维度。

架构:
    1. 门控单元: 对每个模态独立计算 sigmoid(z) * h 的门控表示
    2. 外积融合: o1 ⊗ o2 生成捕捉所有二阶交互的双线性特征
    3. Skip连接: 保留门控后的原始模态特征
    4. 后融合: MLP压缩高维外积特征

输入:
    vec1: [B, dim1]  - 病理图像特征
    vec2: [B, dim2]  - 基因组特征

输出:
    fused: [B, mmhid]  - 融合后的多模态特征

使用示例:
    from src.networks.fusion_bilinear import BilinearFusion
    fusion = BilinearFusion(dim1=32, dim2=32, mmhid=64)
    fused = fusion(path_features, omic_features)
============================================================================
"""

import torch
import torch.nn as nn


class BilinearFusion(nn.Module):
    """
    门控双线性融合模块。

    参数:
        skip: 是否使用skip连接 (默认1)
        use_bilinear: 是否使用bilinear层计算门控z，否则用拼接+Linear (默认1)
        gate1: 是否对模态1使用门控 (默认1)
        gate2: 是否对模态2使用门控 (默认1)
        dim1: 模态1的特征维度 (默认32)
        dim2: 模态2的特征维度 (默认32)
        scale_dim1: 模态1的降维比例 (默认1)
        scale_dim2: 模态2的降维比例 (默认1)
        mmhid: 融合输出维度 (默认64)
        dropout_rate: Dropout比率 (默认0.25)
    """

    def __init__(
        self,
        skip=1,
        use_bilinear=1,
        gate1=1,
        gate2=1,
        dim1=32,
        dim2=32,
        scale_dim1=1,
        scale_dim2=1,
        mmhid=64,
        dropout_rate=0.25,
    ):
        super().__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og = dim1, dim2
        dim1 = dim1 // scale_dim1
        dim2 = dim2 // scale_dim2
        self.dim1 = dim1
        self.dim2 = dim2

        # skip拼接后的额外维度
        skip_dim = (dim1 + 1) + (dim2 + 1) if skip else 0

        # ---- 模态1门控单元 ----
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = (
            nn.Bilinear(dim1_og, dim2_og, dim1)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim1))
        )
        self.linear_o1 = nn.Sequential(
            nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate),
        )

        # ---- 模态2门控单元 ----
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = (
            nn.Bilinear(dim1_og, dim2_og, dim2)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim2))
        )
        self.linear_o2 = nn.Sequential(
            nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate),
        )

        # ---- 后融合MLP ----
        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)

        # encoder1: 将外积(dim1+1)*(dim2+1)=1089维压缩到mmhid=64维
        self.encoder1 = nn.Sequential(
            nn.Linear((dim1 + 1) * (dim2 + 1), mmhid),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

        # encoder2: 处理skip拼接后的特征
        self.encoder2 = nn.Sequential(
            nn.Linear(mmhid + skip_dim, mmhid),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

        # 初始化权重（使用self-normalizing初始化）
        self._init_max_weights()

        # skip加权网络: 控制skip特征的混合比例
        self.skip_weight_net = nn.Sequential(
            nn.Linear(mmhid + (dim1 + 1) + (dim2 + 1), 75),
            nn.ReLU(),
            nn.Linear(75, mmhid + (dim1 + 1) + (dim2 + 1)),
            nn.Sigmoid(),
        )

    def _init_max_weights(self):
        """Self-Normalizing权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                import math
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.normal_(0, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, vec1, vec2):
        """
        前向传播。

        参数:
            vec1: [B, dim1_og] 病理图像特征
            vec2: [B, dim2_og] 基因组特征

        返回:
            out: [B, mmhid] 融合后的多模态特征
        """
        # 保证二维 [B, D]
        if vec1.dim() > 2:
            vec1 = vec1.view(vec1.size(0), -1)
        if vec2.dim() > 2:
            vec2 = vec2.view(vec2.size(0), -1)

        # ---- 模态1门控 ----
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = (
                self.linear_z1(vec1, vec2)
                if self.use_bilinear
                else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            )
            o1 = self.linear_o1(torch.sigmoid(z1) * h1)
        else:
            o1 = self.linear_o1(vec1)

        # ---- 模态2门控 ----
        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = (
                self.linear_z2(vec1, vec2)
                if self.use_bilinear
                else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            )
            o2 = self.linear_o2(torch.sigmoid(z2) * h2)
        else:
            o2 = self.linear_o2(vec2)

        # ---- 添加偏置1（用于外积捕获偏置交互） ----
        ones1 = torch.ones(o1.size(0), 1, device=o1.device, dtype=o1.dtype)
        ones2 = torch.ones(o2.size(0), 1, device=o2.device, dtype=o2.dtype)
        o1 = torch.cat((o1, ones1), dim=1)  # [B, dim1+1]
        o2 = torch.cat((o2, ones2), dim=1)  # [B, dim2+1]

        # ---- 外积: [B,33] ⊗ [B,33] -> [B,1089] ----
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)

        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)  # [B, mmhid]

        # ---- Skip连接: 加权混合外积特征与原始门控特征 ----
        if self.skip:
            cat_all = torch.cat((out, o1, o2), dim=1)  # [B, mmhid+dim1+1+dim2+1]
            weight = self.skip_weight_net(cat_all)

            mmhid_dim = out.size(1)
            d1p1 = o1.size(1)
            d2p1 = o2.size(1)

            out = torch.cat(
                (
                    out * weight[:, :mmhid_dim],
                    o1 * weight[:, mmhid_dim:mmhid_dim + d1p1],
                    o2 * weight[:, mmhid_dim + d1p1:mmhid_dim + d1p1 + d2p1],
                ),
                dim=1,
            )

        out = self.encoder2(out)  # [B, mmhid]
        return out


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BilinearFusion 模块自测")
    print("=" * 60)

    B, D1, D2 = 8, 32, 32
    vec1 = torch.randn(B, D1)
    vec2 = torch.randn(B, D2)

    for mmhid in [32, 64, 128]:
        model = BilinearFusion(dim1=D1, dim2=D2, mmhid=mmhid)
        out = model(vec1, vec2)
        print(f"  mmhid={mmhid}:")
        print(f"    输入: vec1={vec1.shape}, vec2={vec2.shape}")
        print(f"    输出: fused={out.shape}")
        print(f"    总参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n所有测试通过!")
