# -*- coding: utf-8 -*-
"""
pathomic_net.py
============================================================================
Histomolecular主网络 (PathomicNet)。

任务: 用WSI patch图像预测肠分化/粘液相关RNA-seq基因表达。

架构 (保留原有PathNet/OmicNet/9种融合模块，输入用途改变):
    1. PathNet2: 从WSI patch提取病理特征 (N个patch均值池化 -> 病人级)
    2. OmicNet : 编码可选协变量 (临床/其他分子特征, 可选)
    3. Fusion  : 多模态融合 (仅 pathomic 模式)
    4. Classifier: 输出基因标签
        - classification: label_dim=2 (高/低表达二分类, 输出logits)
        - regression    : label_dim=1 (连续表达量)

运行模式:
    - path    : 仅图像 -> 分类器
    - omic    : 仅协变量 -> 分类器
    - pathomic: 图像+协变量 -> 融合 -> 分类器

使用示例:
    model = PathomicNet(config)
    features, logits = model(x_path=patches, x_omic=covariates)
============================================================================
"""

import os
import torch
import torch.nn as nn

from src.networks.path_net import create_path_net
from src.networks.omic_net import create_omic_net
from src.networks.fusion_factory import create_fusion


class PathomicNet(nn.Module):
    """Histomolecular基因表达预测主网络。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mode = config.model.mode
        self.task_type = str(config.model.task_type).lower()

        print("=" * 70)
        print("[PathomicNet] 构建Histomolecular基因表达预测模型")
        print(f"  运行模式     : {self.mode}")
        print(f"  任务类型     : {self.task_type}")
        print(f"  融合策略     : {config.model.fusion.type}")
        print(f"  病理特征维度 : {config.model.path.dim}")
        print(f"  协变量维度   : {config.model.omic.input_dim}")
        print(f"  目标基因     : {config.model.target_gene}")
        print("=" * 70)

        # ---- 病理图像编码器 (path / pathomic 模式) ----
        if self.mode in ("path", "pathomic"):
            pretrained_path = config.resolve_path(config.model.path.pretrained)
            self.path_net = create_path_net(
                path_dim=config.model.path.dim,
                act=None,
                label_dim=1,
                pretrained_path=str(pretrained_path),
                freeze_backbone=config.model.path.freeze_backbone,
                freeze_bn=config.model.path.freeze_bn,
            )
        else:
            self.path_net = None

        # ---- 协变量编码器 (omic / pathomic 模式) ----
        if self.mode in ("omic", "pathomic"):
            omic_input_dim = config.model.omic.input_dim
            if omic_input_dim <= 0:
                raise ValueError(
                    "mode 含 'omic' 但协变量维度为0。"
                    "请配置 data.genomic.covariate_cols，或改用 mode='path'。"
                )
            self.omic_net = create_omic_net(
                input_dim=omic_input_dim,
                omic_dim=config.model.omic.dim,
                dropout_rate=config.model.fusion.dropout,
                act=None,
                label_dim=1,
                init_max=True,
            )
        else:
            self.omic_net = None

        # ---- 多模态融合 (仅 pathomic) ----
        if self.mode == "pathomic":
            self.fusion = create_fusion(
                fusion_type=config.model.fusion.type,
                dim1=config.model.path.dim,
                dim2=config.model.omic.dim,
                hidden_dim=config.model.fusion.hidden_dim,
                output_dim=config.model.fusion.output_dim,
                dropout=config.model.fusion.dropout,
                caugf_config=config.model.fusion.caugf,
                cross_attention_config=config.model.fusion.cross_attention,
                attention_weighted_config=config.model.fusion.attention_weighted,
                lmf_config=config.model.fusion.lmf,
                pofusion_config=config.model.fusion.pofusion,
            )
        else:
            self.fusion = None

        # ---- 分类器头 ----
        if self.mode == "pathomic":
            head_in_dim = config.model.fusion.output_dim
        elif self.mode == "path":
            head_in_dim = config.model.path.dim
        else:
            head_in_dim = config.model.omic.dim

        if self.task_type == "regression":
            label_dim = 1
        else:
            # 二分类高/低表达
            label_dim = 2

        classifier_dropout = getattr(config.model.fusion, "dropout", 0.25)
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(head_in_dim, label_dim)

        self._print_param_summary()

    def _print_param_summary(self):
        for name, module in [("path_net", self.path_net),
                             ("omic_net", self.omic_net),
                             ("fusion", self.fusion),
                             ("classifier", self.classifier)]:
            if module is None:
                continue
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  [{name}] 总参数: {total:,}, 可训练: {trainable:,}")

        total_all = sum(p.numel() for p in self.parameters())
        trainable_all = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [总计] 总参数: {total_all:,}, 可训练: {trainable_all:,}")

    def _extract_path_feat(self, x_path):
        """从多patch图像提取病人级病理特征 (均值池化)。"""
        if x_path.dim() == 5:
            B, N, C, H, W = x_path.shape
            x_path_flat = x_path.view(B * N, C, H, W)
            patch_feat, _ = self.path_net(x_path=x_path_flat)
            feat_dim = patch_feat.size(-1)
            patch_feat = patch_feat.view(B, N, feat_dim)
            return patch_feat.mean(dim=1)  # [B, path_dim]
        else:
            patient_path_feat, _ = self.path_net(x_path=x_path)
            return patient_path_feat

    def forward(self, **kwargs):
        x_path = kwargs["x_path"]
        x_omic = kwargs.get("x_omic", None)

        # ---- 提取各模态特征 ----
        path_feat = None
        omic_feat = None

        if self.mode in ("path", "pathomic"):
            path_feat = self._extract_path_feat(x_path)

        if self.mode in ("omic", "pathomic"):
            omic_feat, _ = self.omic_net(x_omic=x_omic)

        # ---- 融合 / 单模态 ----
        if self.mode == "pathomic":
            fused = self.fusion(path_feat, omic_feat)
        elif self.mode == "path":
            fused = path_feat
        else:
            fused = omic_feat

        # ---- 分类/回归头 ----
        fused = self.classifier_dropout(fused)
        logits = self.classifier(fused)  # [B, label_dim]

        return fused, logits


def create_pathomic_net(config):
    return PathomicNet(config)


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    from config.config_loader import load_config
    config = load_config()
    config.training.device = "cpu"
    config.model.mode = "path"
    config.upstream.num_patches_per_patient = 3
    config.model.path.input_size = 256

    model = PathomicNet(config)
    B, N = 2, 3
    x_path = torch.randn(B, N, 3, 256, 256)
    x_omic = torch.randn(B, config.model.omic.input_dim) if config.model.omic.input_dim > 0 else torch.zeros(B, 1)

    features, logits = model(x_path=x_path, x_omic=x_omic)
    print(f"  features={features.shape}, logits={logits.shape}")
    print("测试完成!")
