# -*- coding: utf-8 -*-
"""支持 YAML 和命令行覆盖的配置加载器。

加载 default_config.yaml 并通过命令行参数覆盖配置项。
无硬编码路径 —— 所有路径均来自配置文件。
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Any

from common.dataclasses import ConfigBundle

logger = logging.getLogger(__name__)

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False
    logger.warning("pyyaml not installed; falling back to default config only.")


def _load_yaml_config(yaml_path: Path) -> dict:
    """从 YAML 文件加载配置。

    Args:
        yaml_path: YAML 配置文件的路径。

    Returns:
        配置值的字典。
    """
    if not _HAS_YAML:
        raise ImportError("pyyaml is required to load YAML config files.")

    with open(yaml_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if data else {}


def _flatten_nested_config(raw: dict) -> dict:
    """将嵌套的配置节拍平为与 ConfigBundle 字段匹配的扁平字典。

    例如 {'stratified': {'n_bins_x': 3}} -> {'n_bins_x': 3}
    """
    flat = {}
    nested_keys = {"stratified", "splice", "yottixel", "sdm", "visualization", "logging"}
    for key, value in raw.items():
        if key in nested_keys and isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[sub_key] = sub_value
        else:
            flat[key] = value
    return flat


def _apply_cli_overrides(config: ConfigBundle, args: argparse.Namespace) -> ConfigBundle:
    """将命令行参数覆盖应用到配置上。

    只有非 None 的 CLI 参数才会覆盖配置值。

    Args:
        config: 来自 YAML 的基础 ConfigBundle。
        args: 已解析的命令行参数。

    Returns:
        更新后的 ConfigBundle。
    """
    overrides = {}
    for field_name in vars(args):
        value = getattr(args, field_name)
        if value is not None and hasattr(config, field_name):
            overrides[field_name] = value

    if overrides:
        logger.info(f"Applying CLI overrides: {overrides}")
        for key, value in overrides.items():
            setattr(config, key, value)

    return config


def _validate_config(config: ConfigBundle) -> ConfigBundle:
    """校验配置并创建输出目录。

    Args:
        config: 待校验的 ConfigBundle。

    Returns:
        校验后的 ConfigBundle。

    Raises:
        ValueError: 如果必需的路径缺失或无效。
    """
    # 确保输出目录存在
    output_root = Path(config.output_root)
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "patches").mkdir(parents=True, exist_ok=True)
    (output_root / "csv").mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir(parents=True, exist_ok=True)

    # 确保日志目录存在
    log_file = Path(config.log_file)
    if not log_file.is_absolute():
        log_file = output_root / config.log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # 展开用户路径
    if config.wsi_root:
        config.wsi_root = str(Path(config.wsi_root).expanduser())

    return config


def load_config(
    config_path: Optional[str] = None,
    cli_args: Optional[List[str]] = None,
) -> ConfigBundle:
    """从 YAML 文件加载配置，并支持 CLI 覆盖。

    优先级：CLI 参数 > YAML 配置 > 默认值。

    Args:
        config_path: YAML 配置文件的路径。若为 None，则使用
            相对于项目根目录的 configs/default_config.yaml。
        cli_args: 命令行参数列表。若为 None，则使用 sys.argv[1:]。

    Returns:
        经过校验、可直接使用的 ConfigBundle。
    """
    # 从默认值开始
    config = ConfigBundle()

    # 如有 YAML 则加载
    if config_path:
        yaml_path = Path(config_path)
    else:
        # 默认：相对于项目根目录查找 configs/default_config.yaml
        project_root = Path(__file__).resolve().parent.parent
        yaml_path = project_root / "configs" / "default_config.yaml"

    if yaml_path.exists() and _HAS_YAML:
        raw = _load_yaml_config(yaml_path)
        flat = _flatten_nested_config(raw)
        for key, value in flat.items():
            if hasattr(config, key):
                setattr(config, key, value)
        logger.info(f"Loaded config from {yaml_path}")
    else:
        if not yaml_path.exists():
            logger.warning(f"Config file not found: {yaml_path}. Using defaults.")
        if not _HAS_YAML:
            logger.warning("pyyaml not installed. Using defaults.")

    # 解析 CLI 参数
    parser = _build_arg_parser()
    if cli_args is not None:
        args = parser.parse_args(cli_args)
    else:
        args = parser.parse_args([])  # 自动化模式下不传 CLI 参数

    config = _apply_cli_overrides(config, args)
    config = _validate_config(config)

    logger.info(
        f"Configuration: patch_size={config.patch_size}, "
        f"K={config.patches_per_case}, seed={config.seed}, "
        f"samplers={config.enabled_samplers}"
    )

    return config


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建用于 CLI 覆盖的参数解析器。

    Returns:
        配置好的 ArgumentParser。
    """
    parser = argparse.ArgumentParser(
        description="WSI Patch Selection Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 路径
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--wsi-root", type=str, default=None,
                        help="Root directory containing WSI (.svs) files")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Root directory for all outputs")

    # 核心参数
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit to first N cases (0=all, default: 0)")
    parser.add_argument("--patch-size", type=int, default=None,
                        help="Side length of square patches (default: 1024)")
    parser.add_argument("--patches-per-case", type=int, default=None,
                        help="Target patches per case, K (default: 6)")
    parser.add_argument("--ds-mask", type=int, default=None,
                        help="Downsample factor for tissue mask (default: 32)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Grid stride for candidate scanning (default: 1024)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Global random seed (default: 2025)")

    # 候选池
    parser.add_argument("--preselect-topk", type=int, default=None,
                        help="Top tissue-ratio candidates to preselect (default: 96)")
    parser.add_argument("--candidate-pool-size", type=int, default=None,
                        help="Candidate pool size per slide (default: 72)")
    parser.add_argument("--max-tries", type=int, default=None,
                        help="Max random sampling attempts (default: 5000)")

    # 评分
    parser.add_argument("--innovation-weight", type=float, default=None,
                        help="Innovation score weight (default: 0.25)")

    # 多样性
    parser.add_argument("--min-center-distance-ratio", type=float, default=None,
                        help="Spatial diversity threshold ratio (default: 0.65)")
    parser.add_argument("--min-feature-distance", type=float, default=None,
                        help="Feature diversity threshold (default: 0.10)")

    # 算法特定参数
    parser.add_argument("--redundancy-lambda", type=float, default=None,
                        help="Redundancy penalty weight for SPLICE (default: 2.0)")

    # 可视化
    parser.add_argument("--dpi", type=int, default=None,
                        help="Figure DPI (default: 300)")

    # 日志
    parser.add_argument("--log-level", type=str, default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")

    return parser


class ConfigLoader:
    """加载配置的便捷包装器。

    用法:
        loader = ConfigLoader("configs/default_config.yaml")
        config = loader.load()
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._config: Optional[ConfigBundle] = None

    def load(self, cli_args: Optional[List[str]] = None) -> ConfigBundle:
        self._config = load_config(self.config_path, cli_args)
        return self._config

    @property
    def config(self) -> ConfigBundle:
        if self._config is None:
            raise RuntimeError("Config not loaded. Call load() first.")
        return self._config
