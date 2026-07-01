# -*- coding: utf-8 -*-
"""
fusion_factory.py
============================================================================
多模态融合策略工厂模块。

统一管理和创建9种多模态融合策略，支持通过配置字符串切换融合方式。

已注册的融合策略:
    | 策略名               | 模块                          | 说明                         |
    |---------------------|------------------------------|------------------------------|
    | pofusion            | BilinearFusion               | 门控双线性外积融合             |
    | lmf                 | LMF                          | 低秩多模态融合(参数高效)        |
    | concat              | ConcatFusion                 | 简单拼接+MLP基线              |
    | gmu                 | GMUFusion                    | 门控多模态单元                |
    | film                | FiLMFusion                   | 特征线性调制(基因组调病理)      |
    | caugf               | VectorCAUGF                  | 三流自适应加权融合 ★核心创新    |
    | attention_weighted  | AttentionWeightedFusion      | 注意力加权融合 ★新增           |
    | cross_attention     | CrossAttentionFusion         | 交叉注意力多模态融合 ★新增      |
    | tensor_concat       | TensorConcatFusion           | 多层级特征图拼接融合 ★新增      |

使用示例:
    from src.networks.fusion_factory import create_fusion

    # 通过字符串创建融合模块
    fusion = create_fusion("caugf", dim1=32, dim2=32, hidden_dim=64, output_dim=64)

    # 通过配置创建
    fusion = create_fusion(config.model.fusion.type, **config_params)
============================================================================
"""

import torch.nn as nn
from typing import Dict, Any, Optional, Type

# 导入所有融合策略
from src.networks.fusion_bilinear import BilinearFusion
from src.networks.fusion_lmf import LMF
from src.networks.fusion_concat import ConcatFusion
from src.networks.fusion_gmu import GMUFusion
from src.networks.fusion_film import FiLMFusion
from src.networks.caugf import VectorCAUGF
from src.networks.fusion_attention_weighted import AttentionWeightedFusion
from src.networks.fusion_cross_attention import CrossAttentionFusion
from src.networks.fusion_tensor_concat import TensorConcatFusion


# ============================================================
# 融合策略注册表
# ============================================================

# 核心融合策略注册表: 策略名 -> (模块类, 描述)
FUSION_REGISTRY: Dict[str, Type[nn.Module]] = {
    "pofusion": BilinearFusion,
    "lmf": LMF,
    "concat": ConcatFusion,
    "gmu": GMUFusion,
    "film": FiLMFusion,
    "caugf": VectorCAUGF,
    "attention_weighted": AttentionWeightedFusion,
    "cross_attention": CrossAttentionFusion,
    "tensor_concat": TensorConcatFusion,
}


# 融合策略中文描述
FUSION_DESCRIPTIONS = {
    "pofusion": "门控双线性外积融合 - 通过外积捕获全部二阶模态交互，门控机制自适应过滤",
    "lmf": "低秩多模态融合 - 将全秩融合张量分解为多个低秩因子，参数高效",
    "concat": "简单拼接融合基线 - 直接拼接两个模态特征后MLP投影",
    "gmu": "门控多模态单元 - sigmoid门控在每个特征维度独立控制模态混合比例",
    "film": "特征线性调制 - 基因组特征生成γ/β参数调制病理特征表达",
    "caugf": "VectorCAUGF三流自适应融合 ★核心创新 - 病理流+基因组流+关系流+残差保底",
    "attention_weighted": "注意力加权融合 ★新增 - Scaled Dot-Product Attention动态学习模态重要性",
    "cross_attention": "交叉注意力融合 ★新增 - Transformer cross-attention双向模态交互",
    "tensor_concat": "多层级特征图拼接融合 ★新增 - 浅/中/深三层特征各自融合后加权拼接",
}


def get_available_fusions() -> Dict[str, str]:
    """
    获取所有可用的融合策略及其描述。

    返回:
        Dict[str, str]: 策略名 -> 描述 的映射
    """
    return {
        name: FUSION_DESCRIPTIONS.get(name, "未描述")
        for name in FUSION_REGISTRY.keys()
    }


def register_fusion(name: str, module_class: Type[nn.Module], description: str = ""):
    """
    注册新的融合策略到注册表。

    参数:
        name: 策略名称（唯一标识）
        module_class: 融合模块的nn.Module子类
        description: 策略描述（可选）

    使用示例:
        register_fusion("my_fusion", MyFusionModule, "自定义融合策略")
    """
    FUSION_REGISTRY[name] = module_class
    if description:
        FUSION_DESCRIPTIONS[name] = description
    print(f"[FusionFactory] 已注册新融合策略: '{name}'")


def create_fusion(
    fusion_type: str,
    dim1: int = 32,
    dim2: int = 32,
    hidden_dim: int = 64,
    output_dim: int = 64,
    dropout: float = 0.25,
    **kwargs,
) -> nn.Module:
    """
    根据融合策略类型创建对应的融合模块。

    参数:
        fusion_type: 融合策略名称（见FUSION_REGISTRY）
        dim1: 模态1的特征维度（病理, 默认32）
        dim2: 模态2的特征维度（基因组, 默认32）
        hidden_dim: 隐藏层维度（默认64）
        output_dim: 输出维度（默认64）
        dropout: Dropout比率（默认0.25）
        **kwargs: 各融合策略的专用参数

    返回:
        nn.Module: 融合模块实例

    异常:
        ValueError: 不支持的融合策略类型

    使用示例:
        # 使用CAUGF
        fusion = create_fusion("caugf", dim1=32, dim2=32, hidden_dim=64, output_dim=64)

        # 使用LMF，指定rank
        fusion = create_fusion("lmf", dim1=32, dim2=32, output_dim=64, lmf_rank=4)

        # 使用交叉注意力
        fusion = create_fusion("cross_attention", dim1=32, dim2=32,
                               hidden_dim=64, output_dim=64,
                               num_heads=4, num_layers=2)
    """
    if fusion_type not in FUSION_REGISTRY:
        available = list(FUSION_REGISTRY.keys())
        raise ValueError(
            f"不支持的融合策略: '{fusion_type}'。"
            f"可用的策略: {available}"
        )

    fusion_class = FUSION_REGISTRY[fusion_type]

    # 根据融合策略类型构建参数
    if fusion_type == "pofusion":
        pofusion_cfg = kwargs.get("pofusion_config", None)
        return fusion_class(
            skip=getattr(pofusion_cfg, "skip", 1) if pofusion_cfg else 1,
            use_bilinear=getattr(pofusion_cfg, "use_bilinear", 1) if pofusion_cfg else 1,
            gate1=getattr(pofusion_cfg, "gate1", 1) if pofusion_cfg else 1,
            gate2=getattr(pofusion_cfg, "gate2", 1) if pofusion_cfg else 1,
            dim1=dim1,
            dim2=dim2,
            mmhid=output_dim,
            dropout_rate=dropout,
        )

    elif fusion_type == "lmf":
        lmf_cfg = kwargs.get("lmf_config", None)
        rank = getattr(lmf_cfg, "rank", 4) if lmf_cfg else kwargs.get("lmf_rank", 4)
        return fusion_class(
            rank=rank,
            hidden_dims=(dim1, dim2),
            output_dim=output_dim,
        )

    elif fusion_type == "caugf":
        caugf_cfg = kwargs.get("caugf_config", None)
        if caugf_cfg:
            return fusion_class(
                dim1=dim1,
                dim2=dim2,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
                relation_types=caugf_cfg.relation_types,
                num_post_layers=caugf_cfg.num_post_layers,
                use_layer_norm=caugf_cfg.use_layer_norm,
                min_stream_weight=caugf_cfg.min_stream_weight,
                temperature_init=caugf_cfg.temperature_init,
            )
        else:
            return fusion_class(
                dim1=dim1, dim2=dim2,
                hidden_dim=hidden_dim, output_dim=output_dim,
                dropout=dropout,
            )

    elif fusion_type == "cross_attention":
        ca_cfg = kwargs.get("cross_attention_config", None)
        return fusion_class(
            dim1=dim1, dim2=dim2,
            hidden_dim=hidden_dim, output_dim=output_dim,
            dropout=dropout,
            num_heads=getattr(ca_cfg, "num_heads", 4) if ca_cfg else 4,
            num_layers=getattr(ca_cfg, "num_layers", 2) if ca_cfg else 2,
            ff_expansion=getattr(ca_cfg, "ff_expansion", 4) if ca_cfg else 4,
        )

    elif fusion_type == "attention_weighted":
        aw_cfg = kwargs.get("attention_weighted_config", None)
        return fusion_class(
            dim1=dim1, dim2=dim2,
            hidden_dim=hidden_dim, output_dim=output_dim,
            dropout=dropout,
            num_heads=getattr(aw_cfg, "num_heads", 4) if aw_cfg else 4,
            use_residual=getattr(aw_cfg, "use_residual", True) if aw_cfg else True,
        )

    else:
        # concat / gmu / film: 通用构造
        return fusion_class(
            dim1=dim1, dim2=dim2,
            hidden_dim=hidden_dim, output_dim=output_dim,
            dropout=dropout,
        )


def print_fusion_registry():
    """打印所有已注册的融合策略"""
    print("=" * 70)
    print("已注册的融合策略")
    print("=" * 70)
    for name in FUSION_REGISTRY:
        desc = FUSION_DESCRIPTIONS.get(name, "无描述")
        print(f"  [{name}] {desc}")
    print("=" * 70)


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    import torch

    print_fusion_registry()

    B, D1, D2 = 8, 32, 32
    o1 = torch.randn(B, D1)
    o2 = torch.randn(B, D2)

    print("\n测试所有已注册的融合策略:")
    for name in FUSION_REGISTRY:
        try:
            model = create_fusion(name, dim1=D1, dim2=D2, hidden_dim=64, output_dim=64)
            out = model(o1, o2)
            params = sum(p.numel() for p in model.parameters())
            print(f"  [{name}] 输出: {out.shape}, 参数量: {params:,} ✓")
        except Exception as e:
            print(f"  [{name}] 创建失败: {e} ✗")

    print("\n所有测试完成!")
