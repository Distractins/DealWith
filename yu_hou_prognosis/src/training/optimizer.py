# -*- coding: utf-8 -*-
"""
optimizer.py
============================================================================
优化器工厂模块 —— 根据配置创建优化器实例。

支持的优化器类型:
    - adam     : torch.optim.Adam
    - adamw    : torch.optim.AdamW (推荐用于带权重衰减的训练)
    - adagrad  : torch.optim.Adagrad
    - adabound : AdaBound (可选，需 pip install adabound)

参数来源:
    config.training.optimizer
        - type          : str  = "adam"     优化器类型
        - lr            : float = 0.0001    初始学习率
        - weight_decay  : float = 0.0003   权重衰减 (L2正则化系数)
        - betas         : list  = [0.9, 0.999]  Adam/AdamW/AdaBound 的动量系数
        - final_lr      : float = 0.1       AdaBound 最终学习率

使用示例:
    from config.config_loader import load_config
    from src.training.optimizer import create_optimizer

    config = load_config("config/default_config.yaml")
    optimizer = create_optimizer(model, config)
============================================================================
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
import torch.optim as optim
from torch.nn import Module
from torch.optim import Optimizer

from config.config_loader import ConfigBundle
from src.utils.logger import get_logger


# ============================================================
# 模块级日志器
# ============================================================
_logger: Optional[logging.Logger] = None


def _get_logger() -> logging.Logger:
    """获取或创建模块级日志器 (懒加载)。"""
    global _logger
    if _logger is None:
        _logger = get_logger("YuHou.optimizer")
    return _logger


# ============================================================
# 参数提取
# ============================================================

def _extract_optim_params(config: ConfigBundle) -> Dict[str, Any]:
    """
    从配置中提取优化器参数，返回标准化字典。

    处理配置中可能缺少的字段，使用安全的默认值。

    参数:
        config: ConfigBundle 实例

    返回:
        dict: 包含所有优化器相关参数的字典
    """
    opt_cfg = config.training.optimizer

    params: Dict[str, Any] = {
        "type": getattr(opt_cfg, "type", "adam"),
        "lr": float(getattr(opt_cfg, "lr", 0.0001)),
        "weight_decay": float(getattr(opt_cfg, "weight_decay", 0.0003)),
        "betas": tuple(getattr(opt_cfg, "betas", [0.9, 0.999])),
        "final_lr": float(getattr(opt_cfg, "final_lr", 0.1)),
    }

    # 确保 betas 是二元组
    if not isinstance(params["betas"], (list, tuple)) or len(params["betas"]) != 2:
        params["betas"] = (0.9, 0.999)
    else:
        params["betas"] = tuple(params["betas"])

    return params


# ============================================================
# 参数分组 (微分学习率)
# ============================================================

def _group_params(
    model: Module,
    base_lr: float,
    weight_decay: float,
    finetune_groups: bool = False,
    backbone_lr_ratio: float = 0.1,
    no_decay_keywords: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    将模型参数分组，支持微分学习率与选择性权重衰减。

    分组策略:
        1. 如果 finetune_groups=True，将参数分为 backbone (低学习率)
           与 head (全学习率) 两组。
        2. 对于 BatchNorm / LayerNorm / bias 等参数，禁用 weight_decay，
           避免正则化破坏归一化层的行为。

    参数:
        model:          PyTorch 模型
        base_lr:        基础学习率
        weight_decay:   全局权重衰减系数
        finetune_groups:是否启用微分学习率分组
        backbone_lr_ratio:backbone 参数的学习率系数 (默认 0.1)
        no_decay_keywords: 不施加 weight_decay 的参数名关键字集合。
                           默认: ["bias", "LayerNorm", "BatchNorm", "norm"]

    返回:
        List[Dict]: 参数分组列表，每个元素包含 "params" 和选项
    """
    if no_decay_keywords is None:
        no_decay_keywords = ["bias", "LayerNorm", "BatchNorm", "norm"]

    # 分离需要 / 不需要 weight_decay 的参数
    decay_params = []
    no_decay_params = []
    decay_names = []
    no_decay_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 判定是否需要 weight_decay
        if any(keyword in name for keyword in no_decay_keywords):
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            decay_params.append(param)
            decay_names.append(name)

    param_groups = []

    # 带 weight_decay 的参数组
    if decay_params:
        param_groups.append({
            "params": decay_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
        })

    # 不带 weight_decay 的参数组 (bias / norm 层)
    if no_decay_params:
        param_groups.append({
            "params": no_decay_params,
            "lr": base_lr,
            "weight_decay": 0.0,
        })

    logger = _get_logger()
    logger.debug(
        "参数分组完成: 带weight_decay参数=%d个, 无weight_decay参数=%d个 (bias/norm)",
        len(decay_params), len(no_decay_params),
    )

    # 如果启用微调分组，进一步修改 backbone 参数组的学习率
    if finetune_groups:
        backbone_lr = base_lr * backbone_lr_ratio
        for group in param_groups:
            # 遍历并标记 backbone 参数
            backbone_params = []
            other_params = []
            for p in group["params"]:
                # 简单启发式: 参数名含 "backbone" 或 "path" 且不含 "fusion" 或 "head"
                found = False
                for n, mp in model.named_parameters():
                    if mp is p:
                        if ("backbone" in n or "path_net" in n or "resnet" in n) and \
                           "fusion" not in n and "head" not in n:
                            backbone_params.append(p)
                        else:
                            other_params.append(p)
                        found = True
                        break
                if not found:
                    other_params.append(p)

            group["params"] = other_params
            if backbone_params:
                param_groups.append({
                    "params": backbone_params,
                    "lr": backbone_lr,
                    "weight_decay": group.get("weight_decay", weight_decay),
                })

        logger.info(
            "微分学习率已启用: head组 lr=%.2e, backbone组 lr=%.2e (ratio=%.2f)",
            base_lr, backbone_lr, backbone_lr_ratio,
        )

    return param_groups


# ============================================================
# 各优化器工厂函数
# ============================================================

def _create_adam(
    model: Module,
    params: Dict[str, Any],
    **kwargs,
) -> optim.Adam:
    """
    创建 Adam 优化器。

    参数:
        model:  PyTorch 模型
        params: 提取后的优化器参数字典
        **kwargs: 额外参数 (如 finetune_groups)

    返回:
        torch.optim.Adam
    """
    lr = params["lr"]
    betas = params["betas"]
    weight_decay = params["weight_decay"]

    param_groups = _group_params(
        model, lr, weight_decay,
        finetune_groups=kwargs.get("finetune_groups", False),
        backbone_lr_ratio=kwargs.get("backbone_lr_ratio", 0.1),
    )

    optimizer = optim.Adam(
        param_groups,
        lr=lr,
        betas=betas,
        weight_decay=0.0,  # 已在 _group_params 中手动处理
        eps=1e-8,
        amsgrad=False,
    )

    # 重新设置 param_groups 中的 weight_decay (Adam 原生不支持 per-group weight_decay)
    # 实际上 PyTorch >= 1.11 支持 per-group weight_decay，此处确保兼容
    _get_logger().info(
        "优化器: Adam | lr=%.2e | betas=%s | weight_decay=%.2e (分组控制)",
        lr, betas, weight_decay,
    )
    return optimizer


def _create_adamw(
    model: Module,
    params: Dict[str, Any],
    **kwargs,
) -> optim.AdamW:
    """
    创建 AdamW 优化器。

    AdamW 将权重衰减与自适应学习率解耦，是带正则化训练的推荐选择。
    比 Adam + L2 正则化更正确地实现了权重衰减。

    参数:
        model:  PyTorch 模型
        params: 提取后的优化器参数字典
        **kwargs: 额外参数

    返回:
        torch.optim.AdamW
    """
    lr = params["lr"]
    betas = params["betas"]
    weight_decay = params["weight_decay"]

    param_groups = _group_params(
        model, lr, weight_decay,
        finetune_groups=kwargs.get("finetune_groups", False),
        backbone_lr_ratio=kwargs.get("backbone_lr_ratio", 0.1),
    )

    optimizer = optim.AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        weight_decay=0.0,  # 已在 _group_params 中手动分组控制
        eps=1e-8,
        amsgrad=False,
    )

    _get_logger().info(
        "优化器: AdamW | lr=%.2e | betas=%s | weight_decay=%.2e (分组控制)",
        lr, betas, weight_decay,
    )
    return optimizer


def _create_adagrad(
    model: Module,
    params: Dict[str, Any],
    **kwargs,
) -> optim.Adagrad:
    """
    创建 AdaGrad 优化器。

    适用于稀疏特征场景 (如基因组特征)。自适应地为低频特征赋予更高学习率。

    注意:
        AdaGrad 会累积历史梯度平方和，学习率单调递减，可能在训练后期过早停滞。
        对于深度学习模型，通常推荐 Adam / AdamW。

    参数:
        model:  PyTorch 模型
        params: 提取后的优化器参数字典
        **kwargs: 额外参数

    返回:
        torch.optim.Adagrad
    """
    lr = params["lr"]
    weight_decay = params["weight_decay"]

    param_groups = _group_params(
        model, lr, weight_decay,
        finetune_groups=kwargs.get("finetune_groups", False),
        backbone_lr_ratio=kwargs.get("backbone_lr_ratio", 0.1),
    )

    optimizer = optim.Adagrad(
        param_groups,
        lr=lr,
        weight_decay=0.0,
        lr_decay=0.0,
        initial_accumulator_value=0.0,
        eps=1e-10,
    )

    _get_logger().info(
        "优化器: AdaGrad | lr=%.2e | weight_decay=%.2e (分组控制)",
        lr, weight_decay,
    )
    return optimizer


def _create_adabound(
    model: Module,
    params: Dict[str, Any],
    **kwargs,
) -> Optimizer:
    """
    创建 AdaBound 优化器 (可选)。

    AdaBound 在训练初期行为类似 Adam，后期平滑过渡到 SGD，
    兼具 Adam 的快速收敛与 SGD 的良好泛化能力。

    依赖:
        pip install adabound
        import adabound

    参数:
        model:  PyTorch 模型
        params: 提取后的优化器参数字典
        **kwargs: 额外参数

    返回:
        adabound.AdaBound 实例

    异常:
        ImportError: 如果 adabound 包未安装
    """
    try:
        import adabound
    except ImportError:
        raise ImportError(
            "AdaBound 优化器需要安装 adabound 包。\n"
            "请运行: pip install adabound\n"
            "或使用其他优化器类型 (adam / adamw / adagrad)。"
        )

    lr = params["lr"]
    betas = params["betas"]
    final_lr = params["final_lr"]
    weight_decay = params["weight_decay"]

    param_groups = _group_params(
        model, lr, weight_decay,
        finetune_groups=kwargs.get("finetune_groups", False),
        backbone_lr_ratio=kwargs.get("backbone_lr_ratio", 0.1),
    )

    optimizer = adabound.AdaBound(
        param_groups,
        lr=lr,
        betas=betas,
        final_lr=final_lr,
        gamma=1e-3,
        weight_decay=0.0,
        eps=1e-8,
        amsbound=False,
    )

    _get_logger().info(
        "优化器: AdaBound | lr=%.2e -> final_lr=%.2e | betas=%s | weight_decay=%.2e (分组控制)",
        lr, final_lr, betas, weight_decay,
    )
    return optimizer


# ============================================================
# 优化器类型到工厂函数的映射
# ============================================================
_OPTIMIZER_REGISTRY: Dict[str, Any] = {
    "adam": _create_adam,
    "adamw": _create_adamw,
    "adagrad": _create_adagrad,
    "adabound": _create_adabound,
}


def _get_available_optimizers() -> List[str]:
    """返回当前环境下可用的优化器类型列表。"""
    available = ["adam", "adamw", "adagrad"]
    try:
        import adabound  # noqa: F401
        available.append("adabound")
    except ImportError:
        pass
    return available


# ============================================================
# 主入口：create_optimizer
# ============================================================

def create_optimizer(
    model: Module,
    config: ConfigBundle,
    **kwargs,
) -> Optimizer:
    """
    根据配置创建优化器 (工厂函数主入口)。

    自动检测配置中的 optimizer.type 字段，调用对应的工厂函数。
    支持通过 **kwargs 传入额外参数 (如 finetune_groups, backbone_lr_ratio)。

    参数:
        model:      PyTorch 模型实例，需包含可训练参数
        config:     ConfigBundle 配置实例
        **kwargs:   额外参数，包括:
            - finetune_groups (bool):   是否启用微分学习率分组 (默认 False)
            - backbone_lr_ratio (float):backbone 学习率系数 (默认 0.1)

    返回:
        torch.optim.Optimizer: 配置好的优化器实例

    异常:
        ValueError: 不支持的优化器类型
        ImportError: AdaBound 未安装但被选用

    使用示例:
        >>> from config.config_loader import load_config
        >>> from src.training.optimizer import create_optimizer
        >>>
        >>> config = load_config("config/default_config.yaml")
        >>> optimizer = create_optimizer(model, config)
        >>>
        >>> # 微调模式: backbone 使用低学习率
        >>> optimizer = create_optimizer(
        ...     model, config,
        ...     finetune_groups=True,
        ...     backbone_lr_ratio=0.1,
        ... )
    """
    logger = _get_logger()
    params = _extract_optim_params(config)
    opt_type = params["type"].lower()

    if opt_type not in _OPTIMIZER_REGISTRY:
        available = _get_available_optimizers()
        raise ValueError(
            f"不支持的优化器类型: '{opt_type}'。\n"
            f"当前可用的类型: {available}\n"
            f"请检查 config.training.optimizer.type 的配置。"
        )

    factory_fn = _OPTIMIZER_REGISTRY[opt_type]
    logger.debug("创建优化器: type=%s", opt_type)

    try:
        optimizer = factory_fn(model, params, **kwargs)
    except ImportError:
        raise
    except Exception as e:
        logger.error("优化器创建失败: %s", str(e))
        raise RuntimeError(f"优化器 '{opt_type}' 创建失败: {e}") from e

    return optimizer


def get_optimizer_info(optimizer: Optimizer) -> Dict[str, Any]:
    """
    获取优化器的元信息摘要。

    参数:
        optimizer: PyTorch 优化器实例

    返回:
        dict: 包含 type, lr, weight_decay, betas 等信息的字典
    """
    info: Dict[str, Any] = {
        "type": type(optimizer).__name__,
        "param_groups": len(optimizer.param_groups),
        "total_params": sum(
            p.numel() for group in optimizer.param_groups for p in group["params"]
        ),
    }

    for i, group in enumerate(optimizer.param_groups):
        info[f"group_{i}_lr"] = group.get("lr", "N/A")
        info[f"group_{i}_weight_decay"] = group.get("weight_decay", "N/A")
        info[f"group_{i}_params"] = sum(
            p.numel() for p in group["params"]
        )

    return info


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("优化器工厂模块自测")
    print("=" * 60)

    # 构建最小化测试模型
    class _DummyModel(torch.nn.Module):
        """用于自测的简单模型"""
        def __init__(self):
            super().__init__()
            self.backbone_conv = torch.nn.Conv2d(3, 16, 3)
            self.backbone_bn = torch.nn.BatchNorm2d(16)
            self.head_fc = torch.nn.Linear(16 * 30 * 30, 10)
            self.head_bias_param = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            x = self.backbone_conv(x)
            x = self.backbone_bn(x)
            x = x.view(x.size(0), -1)
            return self.head_fc(x) + self.head_bias_param

    model = _DummyModel()

    # ---- 模拟 ConfigBundle ----
    # 由于完整的 ConfigBundle 依赖 YAML，这里使用 mock 对象模拟最小配置
    from unittest.mock import Mock

    mock_config = Mock(spec=ConfigBundle)
    mock_opt = Mock()
    mock_opt.type = "adam"
    mock_opt.lr = 0.001
    mock_opt.weight_decay = 0.0001
    mock_opt.betas = [0.9, 0.999]
    mock_opt.final_lr = 0.1

    mock_training = Mock()
    mock_training.optimizer = mock_opt
    mock_config.training = mock_training

    # ---- 测试各优化器 ----
    print("\n[1/4] 测试 Adam ...")
    mock_opt.type = "adam"
    opt_adam = create_optimizer(model, mock_config)
    print(f"  创建成功: {type(opt_adam).__name__}")
    print(f"  参数组数: {len(opt_adam.param_groups)}")
    info = get_optimizer_info(opt_adam)
    print(f"  总参数量: {info['total_params']:,}")

    print("\n[2/4] 测试 AdamW ...")
    mock_opt.type = "adamw"
    opt_adamw = create_optimizer(model, mock_config)
    print(f"  创建成功: {type(opt_adamw).__name__}")

    print("\n[3/4] 测试 AdaGrad ...")
    mock_opt.type = "adagrad"
    opt_adagrad = create_optimizer(model, mock_config)
    print(f"  创建成功: {type(opt_adagrad).__name__}")

    print("\n[4/4] 测试微分学习率分组 ...")
    mock_opt.type = "adam"
    opt_finetune = create_optimizer(
        model, mock_config,
        finetune_groups=True,
        backbone_lr_ratio=0.1,
    )
    print(f"  创建成功: {type(opt_finetune).__name__}")
    for i, g in enumerate(opt_finetune.param_groups):
        print(f"  组{i}: lr={g['lr']:.2e}, weight_decay={g.get('weight_decay', 'N/A')}, "
              f"params={sum(p.numel() for p in g['params'])}")

    # ---- 测试 AdaBound 可选 ----
    print("\n[可选] 测试 AdaBound ...")
    mock_opt.type = "adabound"
    try:
        opt_ab = create_optimizer(model, mock_config)
        print(f"  创建成功: {type(opt_ab).__name__}")
    except ImportError:
        print("  跳过: adabound 未安装 (可选依赖)")

    # ---- 测试错误处理 ----
    print("\n[错误处理] 测试不支持的类型 ...")
    mock_opt.type = "sgd_momentum"
    try:
        create_optimizer(model, mock_config)
        print("  错误: 应该抛出异常!")
    except ValueError as e:
        print(f"  正确捕获 ValueError: {str(e)[:80]}...")

    # ---- 梯度验证 ----
    print("\n[梯度验证] ...")
    x = torch.randn(2, 3, 32, 32)
    mock_opt.type = "adam"
    optimizer = create_optimizer(model, mock_config)
    optimizer.zero_grad()
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    print("  优化器 step() 执行成功, 梯度回传正常")

    print("\n" + "=" * 60)
    print("所有自测通过!")
    print("=" * 60)
