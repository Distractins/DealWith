# -*- coding: utf-8 -*-
"""
resnet_backbone.py
============================================================================
ResNet骨干网络模块，包含CBAM通道-空间注意力和CA坐标注意力机制。

支持的模型变体:
    - ResNet34 / ResNet50 / ResNet101
    - ResNeXt50_32x4d / ResNeXt101_32x8d

注意力模块:
    - CBAMLayer: 通道注意力 + 空间注意力 (Convolutional Block Attention Module)
    - CA_Block: 坐标注意力 (Coordinate Attention)，增强位置感知能力

用于:
    病理图像编码器 PathNet2 的骨干网络，提取WSI patch的层次化视觉特征。

使用示例:
    from src.networks.resnet_backbone import resnet50
    backbone = resnet50(num_classes=32, include_top=True)  # 输出32维病理特征
============================================================================
"""

import torch
import torch.nn as nn


# ============================================================
# CBAM: 卷积块注意力模块
# ============================================================
class CBAMLayer(nn.Module):
    """
    CBAM (Convolutional Block Attention Module) 注意力层。

    在通道和空间两个维度上依次进行注意力加权：
    1. 通道注意力: 自适应学习"哪些通道重要"
    2. 空间注意力: 自适应学习"哪些空间位置重要"

    参数:
        channel: 输入特征通道数
        reduction: 通道注意力MLP的降维比例（默认1，即不降维）
        spatial_kernel: 空间注意力的卷积核大小
    """

    def __init__(self, channel, reduction=1, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # ---- 通道注意力: 压缩H,W为1x1 ----
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 共享MLP（使用1x1卷积实现，比Linear更方便操作4D张量）
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),  # inplace节省显存
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )

        # ---- 空间注意力 ----
        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=spatial_kernel,
            padding=spatial_kernel // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """前向传播: 输入[B,C,H,W] -> 输出[B,C,H,W]（形状不变）"""
        # 通道注意力
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        # 空间注意力
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x

        return x


# ============================================================
# CA: 坐标注意力模块
# ============================================================
class CA_Block(nn.Module):
    """
    坐标注意力 (Coordinate Attention) 模块。

    与CBAM不同，CA将空间注意力分解为两个1D编码过程：
    - 沿水平方向 (宽度) 编码
    - 沿垂直方向 (高度) 编码

    这样做的好处：
    1. 保留了精确的位置信息（普通全局池化会丢失）
    2. 计算量更小
    3. 对于需要空间定位的病理图像任务特别有用

    参数:
        channel: 输入特征通道数
        reduction: 降维比例（默认16）
    """

    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        # 1x1卷积降维
        self.conv_1x1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1, stride=1, bias=False,
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        # 分别生成高度和宽度方向的注意力权重
        self.F_h = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1, stride=1, bias=False,
        )
        self.F_w = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1, stride=1, bias=False,
        )

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        """前向传播: 输入[B,C,H,W] -> 输出[B,C,H,W]（形状不变）"""
        _, _, h, w = x.size()

        # 沿高度方向池化: [B,C,H,W] -> [B,C,H,1] -> [B,C,1,H]
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        # 沿宽度方向池化: [B,C,H,W] -> [B,C,1,W]
        x_w = torch.mean(x, dim=2, keepdim=True)

        # 拼接两个方向的池化结果: [B,C,1,W+H]
        # 降维: [B,C,1,W+H] -> [B,C/r,1,W+H]
        x_cat_conv_relu = self.relu(
            self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3)))
        )

        # 拆分回两个方向
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        # 生成注意力权重
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        # 应用注意力
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


# ============================================================
# ResNet基础模块: BasicBlock (ResNet34使用)
# ============================================================
class BasicBlock(nn.Module):
    """
    ResNet基础残差块 (适用于ResNet18/34)。

    结构: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> 残差连接 -> ReLU
    """
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel, out_channels=out_channel,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channel, out_channels=out_channel,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# ============================================================
# ResNet基础模块: Bottleneck (ResNet50/101使用)
# ============================================================
class Bottleneck(nn.Module):
    """
    ResNet瓶颈残差块 (适用于ResNet50/101/152)。

    结构: Conv1x1(降维) -> Conv3x3 -> Conv1x1(升维) -> 残差连接 -> ReLU

    支持分组卷积 (用于ResNeXt变体)。
    """
    expansion = 4  # 输出通道数是中间通道的4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        # 1x1卷积降维
        self.conv1 = nn.Conv2d(
            in_channels=in_channel, out_channels=width,
            kernel_size=1, stride=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(width)

        # 3x3卷积（支持分组）
        self.conv2 = nn.Conv2d(
            in_channels=width, out_channels=width, groups=groups,
            kernel_size=3, stride=stride, bias=False, padding=1,
        )
        self.bn2 = nn.BatchNorm2d(width)

        # 1x1卷积升维
        self.conv3 = nn.Conv2d(
            in_channels=width, out_channels=out_channel * self.expansion,
            kernel_size=1, stride=1, bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


# ============================================================
# ResNet主类
# ============================================================
class ResNet(nn.Module):
    """
    ResNet通用实现，可配置为ResNet/ResNeXt系列模型。

    内嵌CA坐标注意力和CBAM通道-空间注意力模块，
    在layer4输出后应用注意力增强。

    参数:
        block: 残差块类型 (BasicBlock 或 Bottleneck)
        blocks_num: 每层的残差块数量，如 [3,4,6,3] 表示ResNet50
        num_classes: 输出类别数（ImageNet默认1000）
        include_top: 是否包含全局池化和全连接层
        groups: 分组卷积的组数（ResNeXt=32）
        width_per_group: 每组的基础宽度（ResNeXt=4或8）
    """

    def __init__(self, block, blocks_num, num_classes=1000,
                 include_top=True, groups=1, width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group

        # 输入层: 7x7卷积 + 最大池化（与标准ResNet一致）
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4个残差层
        self.layer1 = self._make_layer(block, 64, blocks_num[0])     # 256通道
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)  # 512通道
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)  # 1024通道
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)  # 2048通道

        # 注意力增强模块（可选择性启用）
        self.ca = CA_Block(2048)     # 坐标注意力
        self.cbam = CBAMLayer(2048)  # CBAM注意力

        # 分类头
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化（Kaiming正态分布）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        """
        构建一个残差层。

        参数:
            block: 残差块类型
            channel: 输出通道基数
            block_num: 该层包含的残差块数量
            stride: 第一个块的步长
        """
        downsample = None
        # 当输入/输出维度不匹配时需要降采样
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion),
            )

        layers = []
        layers.append(block(
            self.in_channel, channel,
            downsample=downsample, stride=stride,
            groups=self.groups, width_per_group=self.width_per_group,
        ))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(
                self.in_channel, channel,
                groups=self.groups, width_per_group=self.width_per_group,
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播。

        输入:  [B, 3, H, W]  - RGB病理patch图像
        输出:  [B, num_classes] - 如果include_top=True
               [B, 2048, H/32, W/32] - 如果include_top=False
        """
        # 初始卷积 + 池化: [B,3,H,W] -> [B,64,H/4,W/4]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # -> [B,64,H/8,W/8]

        # 残差层: 逐步降采样
        x = self.layer1(x)      # -> [B,256,H/8,W/8]
        x = self.layer2(x)      # -> [B,512,H/16,W/16]
        x = self.layer3(x)      # -> [B,1024,H/32,W/32]
        x = self.layer4(x)      # -> [B,2048,H/32,W/32]

        # 注意: 默认不启用CA和CBAM（避免额外计算开销）
        # 如需启用，取消以下注释:
        # x = self.ca(x)
        # x = self.cbam(x)

        if self.include_top:
            x = self.avgpool(x)       # -> [B,2048,1,1]
            x = torch.flatten(x, 1)   # -> [B,2048]
            x = self.fc(x)            # -> [B,num_classes]

        return x


# ============================================================
# 模型工厂函数
# ============================================================

def resnet34(num_classes=1000, include_top=True):
    """构建ResNet34模型"""
    return ResNet(BasicBlock, [3, 4, 6, 3],
                  num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    """构建ResNet50模型（本项目默认使用）"""
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    """构建ResNet101模型"""
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    """构建ResNeXt50_32x4d模型"""
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes, include_top=include_top,
                  groups=groups, width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    """构建ResNeXt101_32x8d模型"""
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes, include_top=include_top,
                  groups=groups, width_per_group=width_per_group)
