# -*- coding: utf-8 -*-
"""
path_net.py
============================================================================
病理图像编码器 (PathNet2) 模块。

基于ResNet50骨干网络，用于从WSI patch图像中提取病理形态学特征。
支持ImageNet预训练权重加载、骨干网络冻结、BatchNorm冻结等训练策略。

架构:
    ResNet50 (预训练) -> fc: Linear(2048, path_dim) -> linear: Linear(path_dim, num_classes)

特征提取模式:
    设置act=None时，forward返回:
        features: [B, path_dim]  - 病理特征向量（用于后续融合）
        hazard: [B, num_classes] - 原始数值预测（不使用）

微调策略:
    - freeze_backbone=True: 冻结除fc层外的所有参数，仅训练fc层
    - freeze_bn=True: 将BatchNorm层设为eval模式，不更新统计量
    推荐初期训练时同时开启两个冻结，后期微调时逐步解冻。

使用示例:
    from src.networks.path_net import PathNet2
    path_net = PathNet2(path_dim=32, pretrained_path="weights/resnet50-0676ba61.pth")
    features, _ = path_net(x_path=patch_batch)  # patch_batch: [B, 3, 1024, 1024]
============================================================================
"""

import os
from collections import OrderedDict

import torch
import torch.nn as nn

from src.networks.resnet_backbone import resnet50


# ============================================================
# 工具函数
# ============================================================

def _count_params(model):
    """统计模型的总参数量、可训练参数量、冻结参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def _print_param_status(model, prefix="[PathNet]"):
    """打印模型参数状态"""
    total, trainable, frozen = _count_params(model)
    print(f"{prefix} 总参数量     : {total:,}")
    print(f"{prefix} 可训练参数   : {trainable:,}")
    print(f"{prefix} 冻结参数     : {frozen:,}")


def _extract_state_dict(ckpt):
    """从多种checkpoint格式中提取state_dict"""
    if isinstance(ckpt, OrderedDict):
        return ckpt
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "net", "model"]:
            if key in ckpt and isinstance(ckpt[key], (dict, OrderedDict)):
                return ckpt[key]
    return ckpt


def _clean_state_dict(state_dict):
    """去掉'module.'前缀（处理DataParallel保存的权重）"""
    if not isinstance(state_dict, (dict, OrderedDict)):
        return state_dict
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd


# ============================================================
# PathNet2: 病理图像编码器
# ============================================================

class PathNet2(nn.Module):
    """
    病理图像编码器 (基于ResNet50骨干)。

    参数:
        path_dim: 输出病理特征维度 (默认32)
        act: 激活函数层（为None时不激活，仅输出特征和原始logit）
        num_classes: 输出类别数（生存分析通常设为1）
        pretrained_path: ImageNet预训练权重路径
        freeze_backbone: 是否冻结ResNet骨干 (默认True)
        freeze_bn: 是否冻结BatchNorm统计量 (默认True)
    """

    def __init__(
        self,
        path_dim=32,
        act=None,
        num_classes=1,
        pretrained_path=None,
        freeze_backbone=True,
        freeze_bn=True,
    ):
        super(PathNet2, self).__init__()

        self.act = act

        # 构建ResNet50骨干网络
        self.net = resnet50()

        print("[PathNet2] 初始化ResNet50病理图像编码器")

        # 替换最后的全连接层（ImageNet 1000类 -> path_dim维特征）
        in_channel = self.net.fc.in_features  # 2048
        self.net.fc = nn.Linear(in_channel, path_dim)

        # 加载预训练权重
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained(pretrained_path)
        else:
            if pretrained_path:
                print(f"[PathNet2] 警告: 预训练权重不存在: {pretrained_path}")
            print("[PathNet2] 使用随机初始化")

        # 冻结骨干网络
        if freeze_backbone:
            for name, param in self.net.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False
            print("[PathNet2] ResNet骨干已冻结，仅fc层可训练")
        else:
            print("[PathNet2] ResNet骨干可训练")

        # 冻结BatchNorm
        if freeze_bn:
            self._set_bn_eval(self.net)
            print("[PathNet2] BatchNorm层已设为eval模式并冻结")
        else:
            print("[PathNet2] BatchNorm层保持可训练")

        # 输出层
        self.linear = nn.Linear(path_dim, num_classes)

        # 输出范围调整参数（用于Sigmoid激活时的输出缩放）
        self.output_range = nn.Parameter(torch.tensor([6.0]), requires_grad=False)
        self.output_shift = nn.Parameter(torch.tensor([-3.0]), requires_grad=False)

        _print_param_status(self, prefix="[PathNet2]")

    def _load_pretrained(self, ckpt_path):
        """加载预训练权重（自动跳过shape不匹配的层）"""
        print(f"[PathNet2] 加载预训练权重: {ckpt_path}")

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = _extract_state_dict(ckpt)
            state_dict = _clean_state_dict(state_dict)

            model_dict = self.net.state_dict()
            matched = {}
            skipped = []

            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    matched[k] = v
                elif k in model_dict:
                    skipped.append((k, tuple(v.shape), tuple(model_dict[k].shape)))

            model_dict.update(matched)
            self.net.load_state_dict(model_dict, strict=False)

            print(f"[PathNet2] 加载成功: {len(matched)}个层匹配, {len(skipped)}个层shape不匹配")

        except Exception as e:
            print(f"[PathNet2] 加载预训练权重失败: {e}")

    @staticmethod
    def _set_bn_eval(module):
        """递归地将所有BatchNorm层设为eval模式并冻结参数"""
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        """覆盖train方法：即使设为train模式，冻结的BN层仍保持eval"""
        super().train(mode)
        return self

    def forward(self, **kwargs):
        """
        前向传播。

        参数:
            x_path: [B, 3, H, W] 病理patch图像批次

        返回:
            features: [B, path_dim] 病理特征向量
            hazard: [B, num_classes] 预测值（如不用于预测则忽略）
        """
        x = kwargs["x_path"].float()

        features = self.net(x)       # [B, path_dim]
        hazard = self.linear(features)  # [B, num_classes]

        # 激活函数处理（survival任务通常为none）
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return features, hazard


# ============================================================
# 便捷构造函数
# ============================================================

def create_path_net(path_dim=32, act=None, label_dim=1,
                    pretrained_path=None, freeze_backbone=True, freeze_bn=True):
    """
    创建PathNet2病理图像编码器的便捷函数。

    参数:
        path_dim: 输出特征维度
        act: 激活函数
        label_dim: 输出预测维度
        pretrained_path: 预训练权重路径
        freeze_backbone: 是否冻结骨干
        freeze_bn: 是否冻结BN

    返回:
        PathNet2实例
    """
    return PathNet2(
        path_dim=path_dim,
        act=act,
        num_classes=label_dim,
        pretrained_path=pretrained_path,
        freeze_backbone=freeze_backbone,
        freeze_bn=freeze_bn,
    )


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PathNet2 病理图像编码器自测")
    print("=" * 60)

    # 测试随机初始化（不加载预训练权重）
    model = PathNet2(path_dim=32, pretrained_path=None)
    x = torch.randn(2, 3, 1024, 1024)
    features, hazard = model(x_path=x)
    print(f"\n  输入: x_path={x.shape}")
    print(f"  特征输出: features={features.shape}")
    print(f"  预测输出: hazard={hazard.shape}")

    # 测试不同path_dim
    for dim in [32, 64, 128]:
        model = PathNet2(path_dim=dim, pretrained_path=None)
        features, _ = model(x_path=torch.randn(1, 3, 1024, 1024))
        print(f"  path_dim={dim}: features={features.shape}")

    print("\n所有测试通过!")
