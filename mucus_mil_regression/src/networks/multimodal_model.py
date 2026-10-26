# -*- coding: utf-8 -*-
"""
multimodal_model.py - 图像-临床多模态融合MIL模型 (实验3)。

图像分支: 主干 -> ABMIL -> 全局表征 z_img
临床分支: MLP编码 -> 临床表征 z_clin
融合:  late_concat / adaptive_gate -> 回归得分

返回: 预测得分 [B] + 实例注意力权重 [B, N]。
"""

import torch
import torch.nn as nn

from src.networks.backbone import build_backbone
from src.networks.abmil import GatedABMIL
from src.networks.clinical_mlp import ClinicalFeatureEncoder
from src.networks.fusion import LateConcatFusion, AdaptiveGateFusion


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone, feat_dim = build_backbone(config)
        self.abmil = GatedABMIL(feat_dim, config.model.mil_hidden_dim, config.model.mil_dropout)
        self.clinical_encoder = ClinicalFeatureEncoder(
            config.model.clinical_input_dim,
            config.model.clinical_hidden_dims,
            config.model.clinical_dropout,
        )
        clin_dim = self.clinical_encoder.output_dim

        if config.model.fusion_type == "adaptive_gate":
            self.fusion = AdaptiveGateFusion(feat_dim, clin_dim,
                                             config.model.fusion_hidden_dim,
                                             config.model.clinical_dropout)
        else:
            self.fusion = LateConcatFusion(feat_dim, clin_dim,
                                           config.model.fusion_hidden_dim,
                                           config.model.clinical_dropout)

    def forward(self, bag: torch.Tensor, clinical: torch.Tensor):
        B, N, C, H, W = bag.shape
        x = self.backbone(bag.view(B * N, C, H, W))
        x = x.view(B, N, -1)
        z_img, attn = self.abmil(x)                    # [B, feat_dim]
        z_clin = self.clinical_encoder(clinical)       # [B, clin_dim]
        out = self.fusion(z_img, z_clin)               # [B]
        return out, attn
