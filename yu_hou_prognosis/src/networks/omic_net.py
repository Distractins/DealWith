# -*- coding: utf-8 -*-
"""
omic_net.py
============================================================================
基因组特征编码器 (MaxNet) 模块。

基于多层感知机(MLP)的基因组特征编码器，将高维基因组特征
（如356维基因突变标记）压缩为低维密集表示。

架构:
    Linear(input_dim, 64) -> SiLU -> AlphaDropout ->
    Linear(64, 48) -> SiLU -> AlphaDropout ->
    Linear(48, 32) -> SiLU -> AlphaDropout ->
    Linear(32, omic_dim) -> classifier: Linear(omic_dim, label_dim)

设计考虑:
    1. SiLU (Swish) 激活函数: 比ReLU更平滑，适合高维稀疏特征
    2. AlphaDropout: 与SELU配合的自归一化dropout，保持输入分布
    3. 可选的预训练权重加载（例如从单模态预训练的模型加载）

使用示例:
    from src.networks.omic_net import MaxNet
    omic_net = MaxNet(input_dim=356, omic_dim=32)
    features, _ = omic_net(x_omic=genomic_batch)  # genomic_batch: [B, 356]
============================================================================
"""

import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import Parameter


class MaxNet(nn.Module):
    """
    基因组特征编码器 (MLP-based)。

    通过三层渐进降维的MLP将高维稀疏基因组特征编码为低维密集表示。

    参数:
        input_dim: 输入基因组特征维度 (默认356，COAD基因突变特征数)
        omic_dim: 输出基因组特征维度 (默认32)
        dropout_rate: AlphaDropout比率 (默认0.25)
        act: 输出激活函数（训练融合模型时通常为None）
        label_dim: 分类器输出维度 (默认1)
        init_max: 是否使用self-normalizing权重初始化 (默认True)
    """

    def __init__(
        self,
        input_dim=356,
        omic_dim=32,
        dropout_rate=0.25,
        act=None,
        label_dim=1,
        init_max=True,
    ):
        super(MaxNet, self).__init__()

        # 三层渐进降维: 356 -> 64 -> 48 -> 32
        hidden = [64, 48, 32]
        self.act = act

        self.encoder = nn.Sequential(
            # 第1层: 大幅降维 (356 -> 64)
            nn.Linear(input_dim, hidden[0]),
            nn.SiLU(),  # Swish激活: 平滑、非单调、无界
            nn.AlphaDropout(p=dropout_rate, inplace=False),

            # 第2层: 渐进降维 (64 -> 48)
            nn.Linear(hidden[0], hidden[1]),
            nn.SiLU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False),

            # 第3层: 进一步降维 (48 -> 32)
            nn.Linear(hidden[1], hidden[2]),
            nn.SiLU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False),
        )

        # 输出投影（如果hidden[-1] != omic_dim）
        if hidden[-1] != omic_dim:
            self.proj = nn.Linear(hidden[-1], omic_dim)
        else:
            self.proj = nn.Identity()

        # 分类器头
        self.classifier = nn.Linear(omic_dim, label_dim)

        # Self-Normalizing权重初始化（针对SiLU+AlphaDropout组合）
        if init_max:
            self._init_max_weights()

        # 输出范围参数（Sigmoid激活时的输出变换）
        self.output_range = Parameter(torch.tensor([6.0]), requires_grad=False)
        self.output_shift = Parameter(torch.tensor([-3.0]), requires_grad=False)

    def _init_max_weights(self):
        """Self-Normalizing权重初始化（参考SELU论文）"""
        import math
        for m in self.modules():
            if isinstance(m, nn.Linear):
                stdv = 1.0 / math.sqrt(m.weight.size(1))
                m.weight.data.normal_(0, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, **kwargs):
        """
        前向传播。

        参数:
            x_omic: [B, input_dim] 基因组特征向量

        返回:
            features: [B, omic_dim] 基因组编码特征
            out: [B, label_dim] 预测值
        """
        x = kwargs["x_omic"].float()

        # 编码
        features = self.encoder(x)
        features = self.proj(features)

        # 分类预测
        out = self.classifier(features)

        # 输出激活
        if self.act is not None:
            out = self.act(out)
            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift

        return features, out


# ============================================================
# 便捷构造函数
# ============================================================

def create_omic_net(input_dim=356, omic_dim=32, dropout_rate=0.25,
                    act=None, label_dim=1, init_max=True):
    """
    创建MaxNet基因组编码器的便捷函数。

    参数:
        input_dim: 输入基因组特征维度
        omic_dim: 输出特征维度
        dropout_rate: Dropout比率
        act: 输出激活函数
        label_dim: 预测输出维度
        init_max: 是否使用self-normalizing初始化

    返回:
        MaxNet实例
    """
    return MaxNet(
        input_dim=input_dim,
        omic_dim=omic_dim,
        dropout_rate=dropout_rate,
        act=act,
        label_dim=label_dim,
        init_max=init_max,
    )


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MaxNet 基因组特征编码器自测")
    print("=" * 60)

    for input_dim in [356, 200, 100]:
        for omic_dim in [32, 64]:
            model = MaxNet(input_dim=input_dim, omic_dim=omic_dim)
            x = torch.randn(4, input_dim)
            features, out = model(x_omic=x)
            print(f"  input_dim={input_dim}, omic_dim={omic_dim}:")
            print(f"    特征输出: features={features.shape}")
            print(f"    预测输出: out={out.shape}")
            total = sum(p.numel() for p in model.parameters())
            print(f"    参数量: {total:,}")

    print("\n所有测试通过!")
