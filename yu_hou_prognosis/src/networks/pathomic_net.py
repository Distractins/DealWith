# -*- coding: utf-8 -*-
"""
pathomic_net.py
============================================================================
多模态预后预测主网络 (PathomicNet) 模块。

整合病理图像编码器(PathNet2)、基因组编码器(MaxNet)和多模态融合模块，
构成完整的多模态预后预测模型。

架构流程:
    1. PathNet2: 从WSI patch图像提取病理特征 [B, path_dim]
    2. MaxNet: 从基因组数据提取分子特征 [B, omic_dim]
    3. Fusion: 多模态特征融合 [B, mmhid]
    4. Classifier: 生存风险预测 [B, label_dim]

多patch处理:
    每个病人有N个patch ([B, N, C, H, W])，处理流程:
    1. 将patch展开为 [B*N, C, H, W]
    2. PathNet2提取每个patch的特征 [B*N, path_dim]
    3. 重塑为 [B, N, path_dim]
    4. 在N维度上均值池化 -> [B, path_dim] (病人级表示)

使用示例:
    from src.networks.pathomic_net import PathomicNet
    from config.config_loader import load_config

    config = load_config("config/default_config.yaml")
    model = PathomicNet(config)
    features, hazard = model(x_path=patches, x_omic=genomics)
============================================================================
"""

import os
from collections import OrderedDict

import torch
import torch.nn as nn

from src.networks.path_net import create_path_net
from src.networks.omic_net import create_omic_net
from src.networks.fusion_factory import create_fusion


class PathomicNet(nn.Module):
    """
    多模态预后预测主网络。

    整合病理+基因组编码器+融合模块的完整预测模型。

    参数:
        config: ConfigBundle配置对象，包含model.path, model.omic, model.fusion等配置
    """

    def __init__(self, config):
        super(PathomicNet, self).__init__()

        # 保存配置引用
        self.config = config

        print("=" * 70)
        print("[PathomicNet] 构建多模态预后预测模型")
        print(f"  融合策略     : {config.model.fusion.type}")
        print(f"  病理特征维度 : {config.model.path.dim}")
        print(f"  基因组特征维度: {config.model.omic.dim}")
        print(f"  融合隐藏维度 : {config.model.fusion.hidden_dim}")
        print(f"  融合输出维度 : {config.model.fusion.output_dim}")
        print(f"  任务类型     : {config.model.task}")
        print(f"  冻结ResNet   : {config.model.path.freeze_backbone}")
        print("=" * 70)

        # ---- 1) 病理图像编码器 ----
        pretrained_path = config.resolve_path(config.model.path.pretrained)
        self.path_net = create_path_net(
            path_dim=config.model.path.dim,
            act=None,  # 不激活，只用作特征提取
            label_dim=1,
            pretrained_path=str(pretrained_path),
            freeze_backbone=config.model.path.freeze_backbone,
            freeze_bn=config.model.path.freeze_bn,
        )

        # ---- 2) 基因组特征编码器 ----
        omic_input_dim = config.model.omic.input_dim
        self.omic_net = create_omic_net(
            input_dim=omic_input_dim,
            omic_dim=config.model.omic.dim,
            dropout_rate=config.model.fusion.dropout,
            act=None,
            label_dim=1,
            init_max=True,
        )

        # ---- 3) 多模态融合模块 ----
        self.fusion = create_fusion(
            fusion_type=config.model.fusion.type,
            dim1=config.model.path.dim,
            dim2=config.model.omic.dim,
            hidden_dim=config.model.fusion.hidden_dim,
            output_dim=config.model.fusion.output_dim,
            dropout=config.model.fusion.dropout,
            # 各融合策略的专用参数
            caugf_config=config.model.fusion.caugf,
            cross_attention_config=config.model.fusion.cross_attention,
            attention_weighted_config=config.model.fusion.attention_weighted,
            lmf_config=config.model.fusion.lmf,
            pofusion_config=config.model.fusion.pofusion,
        )

        # ---- 4) 风险预测分类器 ----
        label_dim = 1 if config.model.task == "surv" else config.model.omic.input_dim
        # 根据任务类型确定label_dim
        if config.model.task == "surv":
            label_dim = 1
        elif config.model.task == "ncls":
            label_dim = 3  # N0/N1/N2
        else:
            label_dim = 1
        self.classifier = nn.Linear(config.model.fusion.output_dim, label_dim)

        # ---- 5) 输出范围参数 ----
        self.act = None  # surv任务不激活，直接输出风险分数
        self.output_range = nn.Parameter(torch.tensor([6.0]), requires_grad=False)
        self.output_shift = nn.Parameter(torch.tensor([-3.0]), requires_grad=False)

        # 打印参数统计
        self._print_param_summary()

    def _print_param_summary(self):
        """打印各子模块的参数统计"""
        for name, module in [
            ("path_net", self.path_net),
            ("omic_net", self.omic_net),
            ("fusion", self.fusion),
            ("classifier", self.classifier),
        ]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  [{name}] 总参数: {total:,}, 可训练: {trainable:,}")

        total_all = sum(p.numel() for p in self.parameters())
        trainable_all = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [总计] 总参数: {total_all:,}, 可训练: {trainable_all:,}")

    def forward(self, **kwargs):
        """
        前向传播。

        参数:
            x_path: [B, N, 3, H, W] 或 [B, 3, H, W]
                    每个病人N个patch的批次（N=num_patches_per_patient）
            x_omic: [B, input_dim] 基因组特征向量

        返回:
            features: [B, fusion_output_dim] 融合后的多模态特征
            hazard: [B, label_dim] 预测风险分数
        """
        x_path = kwargs["x_path"]
        x_omic = kwargs["x_omic"]

        # ---- 处理多patch图像 ----
        if x_path.dim() == 5:
            # [B, N, C, H, W] -> [B*N, C, H, W]
            B, N, C, H, W = x_path.shape
            x_path_flat = x_path.view(B * N, C, H, W)

            # 提取每个patch的特征
            patch_feat, _ = self.path_net(x_path=x_path_flat)  # [B*N, path_dim]

            # 重塑为病人级: [B*N, path_dim] -> [B, N, path_dim] -> [B, path_dim]
            feat_dim = patch_feat.size(-1)
            patch_feat = patch_feat.view(B, N, feat_dim)

            # 均值池化: 将N个patch聚合为病人级别表示
            patient_path_feat = patch_feat.mean(dim=1)  # [B, path_dim]
        else:
            # 单patch: [B, C, H, W]
            patient_path_feat, _ = self.path_net(x_path=x_path)

        # ---- 提取基因组特征 ----
        patient_omic_feat, _ = self.omic_net(x_omic=x_omic)  # [B, omic_dim]

        # ---- 多模态融合 ----
        fused_feat = self.fusion(patient_path_feat, patient_omic_feat)  # [B, mmhid]

        # ---- 风险预测 ----
        hazard = self.classifier(fused_feat)  # [B, label_dim]

        # 生存分析输出处理
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return fused_feat, hazard


# ============================================================
# 便捷构造函数
# ============================================================

def create_pathomic_net(config):
    """
    根据配置创建PathomicNet的便捷函数。

    参数:
        config: ConfigBundle配置对象

    返回:
        PathomicNet实例
    """
    return PathomicNet(config)


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PathomicNet 多模态预测网络自测")
    print("=" * 60)

    # 使用最小配置测试（需要config模块）
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    try:
        from config.config_loader import load_config
        config = load_config()

        # 覆盖设备为CPU
        config.training.device = "cpu"

        model = PathomicNet(config)

        # 模拟输入
        B, N = 2, 6  # 2个病人，每个6个patch
        x_path = torch.randn(B, N, 3, 1024, 1024)
        x_omic = torch.randn(B, 356)

        print(f"\n  输入: x_path={x_path.shape}, x_omic={x_omic.shape}")

        features, hazard = model(x_path=x_path, x_omic=x_omic)
        print(f"  融合特征: features={features.shape}")
        print(f"  风险预测: hazard={hazard.shape}")

    except ImportError:
        print("  (跳过完整测试: 需要config模块)")

    print("\n测试完成!")
