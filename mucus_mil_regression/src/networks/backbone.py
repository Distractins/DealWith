# -*- coding: utf-8 -*-
"""
backbone.py - 图像主干 (ImageNet ResNet50 / UNI病理自监督 ViT-L/16)。

输出: patch实例特征 [B, feat_dim]
  - resnet50 -> 2048
  - uni(ViT-L/16) -> 1024 (CLS token)

UNI 采用纯PyTorch实现的ViT-L/16 (无timm依赖), 直接加载UNI checkpoint。
"""

import os
from collections import OrderedDict

import torch
import torch.nn as nn


# ============================================================
# ResNet50 (ImageNet预训练)
# ============================================================

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained_path: str = "", feat_dim: int = 2048):
        super().__init__()
        import torchvision.models as models
        try:
            self.net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            self.net = models.resnet50(weights=None)

        # 若提供本地权重则加载 (兼容多种checkpoint格式)
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_local(pretrained_path)

        self.feat_dim = feat_dim
        # 去掉最后的fc, 只保留特征
        self.net.fc = nn.Identity()

    def _load_local(self, path: str):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[ResNetBackbone] 本地权重加载失败: {e}")
            return
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state.items())
        try:
            self.net.load_state_dict(state, strict=False)
            print(f"[ResNetBackbone] 已加载本地权重: {path}")
        except Exception as e:
            print(f"[ResNetBackbone] 权重加载异常: {e}")

    def forward(self, x):
        return self.net(x)  # [B, 2048]


# ============================================================
# ViT-L/16 (UNI病理自监督, 纯PyTorch实现)
# ============================================================

class _Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _Mlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class UNIBackbone(nn.Module):
    """ViT-L/16 (UNI结构): img_size=224, patch=16, embed=1024, depth=24, heads=16。"""

    def __init__(self, pretrained_path: str = "", img_size: int = 224,
                 patch_size: int = 16, embed_dim: int = 1024,
                 depth: int = 24, num_heads: int = 16):
        super().__init__()
        self.feat_dim = embed_dim
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([_Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Identity()  # 只取CLS特征

        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained(pretrained_path)
        else:
            print(f"[UNIBackbone] 未找到UNI权重: {pretrained_path}, 使用随机初始化")

    def _load_pretrained(self, path: str):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[UNIBackbone] 加载失败: {e}")
            return
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state.items())
        # 去掉不匹配的 head
        state = OrderedDict((k, v) for k, v in state.items() if not k.startswith("head."))
        try:
            self.load_state_dict(state, strict=False)
            print(f"[UNIBackbone] 已加载UNI权重: {path}")
        except Exception as e:
            print(f"[UNIBackbone] 权重加载异常: {e}")

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # [B, embed_dim]


# ============================================================
# 工厂
# ============================================================

def build_backbone(config):
    """根据配置构建主干, 返回 (backbone, feat_dim)。"""
    backbone_type = config.model.backbone
    pretrained = config.model.backbone_pretrained
    if backbone_type == "resnet":
        net = ResNetBackbone(pretrained_path=pretrained, feat_dim=2048)
        return net, 2048
    elif backbone_type == "uni":
        net = UNIBackbone(pretrained_path=pretrained)
        return net, net.feat_dim
    raise ValueError(f"不支持的backbone: {backbone_type}")
