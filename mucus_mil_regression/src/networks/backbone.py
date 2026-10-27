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
import torch.nn.functional as F


# ============================================================
# ResNet50 (ImageNet预训练)
# ============================================================

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained_path: str = "", feat_dim: int = 2048, freeze: bool = True):
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

        if freeze:
            for p in self.net.parameters():
                p.requires_grad = False
            print("[ResNetBackbone] 主干已冻结 (仅做特征提取)")

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


class _LayerScale(nn.Module):
    """timm ViT的LayerScale (init_values=1e-5), 键名 ls1.gamma / ls2.gamma。"""

    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class _Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, init_values=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads)
        # init_values=None 表示不带LayerScale (CONCH/Phikon等ViT-B用)
        self.ls1 = _LayerScale(dim, init_values) if init_values is not None else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = _LayerScale(dim, init_values) if init_values is not None else nn.Identity()

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class UNIBackbone(nn.Module):
    """ViT-L/16 (UNI结构): img_size=224, patch=16, embed=1024, depth=24, heads=16。"""

    def __init__(self, pretrained_path: str = "", img_size: int = 224,
                 patch_size: int = 16, embed_dim: int = 1024,
                 depth: int = 24, num_heads: int = 16, freeze: bool = True):
        super().__init__()
        self.feat_dim = embed_dim
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([_Block(embed_dim, num_heads, init_values=1e-5) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Identity()  # 只取CLS特征

        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained(pretrained_path)
        else:
            print(f"[UNIBackbone] 未找到UNI权重: {pretrained_path}, 使用随机初始化")

        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            print("[UNIBackbone] 主干已冻结 (仅做特征提取)")

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
# ViT-B/16 (CONCH / Phikon 等, 纯PyTorch实现)
# ============================================================

def _strip_prefixes(state: OrderedDict) -> OrderedDict:
    """去掉checkpoint中常见的前缀, 兼容多种保存格式。"""
    prefixes = [
        "module.", "model.", "backbone.", "image_encoder.", "encoder.",
        "visual.trunk.", "trunk.", "net.",
    ]
    out = OrderedDict()
    for k, v in state.items():
        k2 = k
        for p in prefixes:
            if k2.startswith(p):
                k2 = k2[len(p):]
                break
        out[k2] = v
    return out


class ViTB16Backbone(nn.Module):
    """ViT-B/16 (timm/OpenCLIP风格), 无LayerScale。

    用于 CONCH(ViT-B/16, 768) / Phikon(ViT-B/16, 768) 等主干。
    img_size=224, patch=16, embed=768, depth=12, heads=12。
    """

    def __init__(self, pretrained_path: str = "", img_size: int = 224,
                 patch_size: int = 16, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, freeze: bool = True):
        super().__init__()
        self.feat_dim = embed_dim
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([_Block(embed_dim, num_heads, init_values=None) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Identity()

        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained(pretrained_path)
        else:
            print(f"[ViTB16Backbone] 未找到权重: {pretrained_path}, 使用随机初始化")

        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            print("[ViTB16Backbone] 主干已冻结 (仅做特征提取)")

    def _load_pretrained(self, path: str):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[ViTB16Backbone] 加载失败: {e}")
            return
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state = _strip_prefixes(OrderedDict(state.items()))
        state = OrderedDict((k, v) for k, v in state.items() if not k.startswith("head."))
        try:
            self.load_state_dict(state, strict=False)
            print(f"[ViTB16Backbone] 已加载权重: {path}")
        except Exception as e:
            print(f"[ViTB16Backbone] 权重加载异常: {e}")

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


# ============================================================
# CTransPath (ViT-Hybrid: ResNet-50 stem + Google-ViT encoder)
# ============================================================

class _GoogleAttention(nn.Module):
    """Google ViT 注意力 (分离的 query/key/value/out, 非融合qkv)。"""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.query(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(x)


class _GoogleBlock(nn.Module):
    """Google ViT encoder块: attention_norm + attn, ffn_norm + ffn (fc1/fc2)。"""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attn = _GoogleAttention(dim, num_heads)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Module()
        self.ffn.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.ffn.fc2 = nn.Linear(int(dim * mlp_ratio), dim)

    def forward(self, x):
        x = x + self.attn(self.attention_norm(x))
        x = x + self.ffn.fc2(F.gelu(self.ffn.fc1(self.ffn_norm(x))))
        return x


class _Bottleneck(nn.Module):
    """Google ResNetV2 Bottleneck (GroupNorm, pre-activation), 键名 gn1/conv1/.../downsample/gn_proj。"""

    def __init__(self, nin, nout, stride=1):
        super().__init__()
        groups = 32 if nin % 32 == 0 else max(1, nin)
        self.gn1 = nn.GroupNorm(groups, nin)
        self.conv1 = nn.Conv2d(nin, nout // 4, 1, bias=False)
        groups2 = 32 if (nout // 4) % 32 == 0 else max(1, nout // 4)
        self.gn2 = nn.GroupNorm(groups2, nout // 4)
        self.conv2 = nn.Conv2d(nout // 4, nout // 4, 3, stride, 1, bias=False)
        self.gn3 = nn.GroupNorm(groups2, nout // 4)
        self.conv3 = nn.Conv2d(nout // 4, nout, 1, bias=False)
        self.use_proj = (nin != nout or stride != 1)
        if self.use_proj:
            self.downsample = nn.Conv2d(nin, nout, 1, stride, bias=False)
            groups_p = 32 if nout % 32 == 0 else max(1, nout)
            self.gn_proj = nn.GroupNorm(groups_p, nout)
        else:
            self.downsample = None
            self.gn_proj = None

    def forward(self, x):
        identity = x
        x = F.relu(self.gn1(x))
        x = self.conv1(x)
        x = F.relu(self.gn2(x))
        x = self.conv2(x)
        x = F.relu(self.gn3(x))
        x = self.conv3(x)
        if self.use_proj:
            identity = self.gn_proj(self.downsample(identity))
        return x + identity


class _ResNetV2Stem(nn.Module):
    """Google ResNetV2 stem (root + body.block1~3), 输出 1/16 分辨率。"""

    def __init__(self):
        super().__init__()
        self.root = nn.Module()
        self.root.conv = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.root.gn = nn.GroupNorm(32, 64)
        self.root.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.body = nn.Module()
        self.body.block1 = self._make_block(64, 256, 3, stride=1)
        self.body.block2 = self._make_block(256, 512, 4, stride=2)
        self.body.block3 = self._make_block(512, 1024, 6, stride=2)

    def _make_block(self, nin, nout, units, stride):
        block = nn.Module()
        block.unit1 = _Bottleneck(nin, nout, stride)
        for i in range(2, units + 1):
            setattr(block, f"unit{i}", _Bottleneck(nout, nout, 1))
        return block

    def forward(self, x):
        x = self.root.conv(x)
        x = self.root.gn(x)
        x = F.relu(x)
        x = self.root.maxpool(x)
        for block in [self.body.block1, self.body.block2, self.body.block3]:
            for unit in block.children():
                x = unit(x)
        return x  # [B, 1024, H/16, W/16]


class CTransPathBackbone(nn.Module):
    """CTransPath: ResNet-50 hybrid stem + Google-ViT encoder (embed=768, 12层)。

    键名前缀: module.net.transformer.embeddings.* / encoder.layer.*
    原生 256×256 输入 → 16×16 patch (257 = 1 cls + 256 patch)。
    """

    def __init__(self, pretrained_path: str = "", embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, freeze: bool = True):
        super().__init__()
        self.feat_dim = embed_dim
        self.stem = _ResNetV2Stem()
        # 1024 -> embed_dim 的 1x1 投影 (对应 embeddings.patch_embeddings)
        self.patch_embeddings = nn.Conv2d(1024, embed_dim, 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, 257, embed_dim))
        self.encoder = nn.ModuleList([_GoogleBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained(pretrained_path)
        else:
            print(f"[CTransPathBackbone] 未找到权重: {pretrained_path}, 使用随机初始化")

        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            print("[CTransPathBackbone] 主干已冻结 (仅做特征提取)")

    def _load_pretrained(self, path: str):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[CTransPathBackbone] 加载失败: {e}")
            return
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        remap = OrderedDict()
        for k, v in state.items():
            k2 = k
            for p in ["module.", "net."]:
                if k2.startswith(p):
                    k2 = k2[len(p):]
                    break
            # transformer.embeddings.hybrid_model -> stem, patch_embeddings -> patch_embeddings
            k2 = k2.replace("transformer.embeddings.hybrid_model", "stem")
            k2 = k2.replace("transformer.embeddings.position_embeddings", "position_embeddings")
            k2 = k2.replace("transformer.embeddings.cls_token", "cls_token")
            k2 = k2.replace("transformer.embeddings.patch_embeddings", "patch_embeddings")
            # encoder.layer.N.xxx -> encoder.N.xxx
            import re
            k2 = re.sub(r"transformer\.encoder\.layer\.(\d+)\.", r"encoder.\1.", k2)
            if k2.startswith("head."):
                continue
            remap[k2] = v

        try:
            self.load_state_dict(remap, strict=False)
            matched = sum(1 for k in self.state_dict() if k in remap)
            total = len(self.state_dict())
            print(f"[CTransPathBackbone] 权重加载: 匹配 {matched}/{total} 个参数")
        except Exception as e:
            print(f"[CTransPathBackbone] 权重加载异常: {e}")

    def forward(self, x):
        B = x.shape[0]
        x = self.stem(x)                          # [B, 1024, h, w]
        x = self.patch_embeddings(x)              # [B, 768, h, w]
        x = x.flatten(2).transpose(1, 2)          # [B, N, 768]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)            # [B, 1+N, 768]

        # position_embeddings 原生257, 若实际patch数不同则插值
        pos = self.position_embeddings
        if x.shape[1] != pos.shape[1]:
            pos = _interpolate_pos_embed(pos, x.shape[1])
        x = x + pos

        for blk in self.encoder:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # [B, 768]


def _interpolate_pos_embed(pos_embed: torch.Tensor, target_len: int) -> torch.Tensor:
    """把 (1, L, D) 的位置编码插值到 target_len。"""
    if pos_embed.shape[1] == target_len:
        return pos_embed
    cls = pos_embed[:, :1, :]
    patch = pos_embed[:, 1:, :]  # [1, L-1, D]
    patch = patch.permute(0, 2, 1)  # [1, D, L-1]
    patch = torch.nn.functional.interpolate(patch, size=target_len - 1, mode="linear")
    patch = patch.permute(0, 2, 1)
    return torch.cat([cls, patch], dim=1)


# ============================================================
# 工厂
# ============================================================

def build_backbone(config):
    """根据配置构建主干, 返回 (backbone, feat_dim)。

    支持: resnet(ImageNet) / retccl(ResNet50病理权重) / uni(ViT-L/16) /
          phikon(ViT-B/16) / conch(ViT-B/16)
    """
    backbone_type = config.model.backbone
    pretrained = config.model.backbone_pretrained
    freeze = getattr(config.model, "freeze_backbone", True)

    if backbone_type in ("resnet", "retccl"):
        # retccl = ResNet50 换病理预训练权重 (retccl_weights.pth)
        if backbone_type == "retccl" and (not pretrained):
            pretrained = "weights/retccl/retccl_weights.pth"
        net = ResNetBackbone(pretrained_path=pretrained, feat_dim=2048, freeze=freeze)
        return net, 2048

    if backbone_type == "uni":
        net = UNIBackbone(pretrained_path=pretrained, freeze=freeze)
        return net, net.feat_dim

    if backbone_type in ("phikon", "conch"):
        net = ViTB16Backbone(pretrained_path=pretrained, freeze=freeze)
        return net, net.feat_dim

    if backbone_type == "ctranspath":
        net = CTransPathBackbone(pretrained_path=pretrained, freeze=freeze)
        return net, net.feat_dim

    raise ValueError(f"不支持的backbone: {backbone_type}。可选: resnet/retccl/uni/phikon/conch/ctranspath")
