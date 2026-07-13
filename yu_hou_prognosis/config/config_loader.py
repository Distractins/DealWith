# -*- coding: utf-8 -*-
"""
config_loader.py
============================================================================
配置加载器：从YAML配置文件加载所有参数，支持命令行覆盖。

设计模式参考:
    wsi_patch_selection_benchmark/configs/config_loader.py

功能:
    1. 从YAML文件加载默认配置
    2. 命令行参数覆盖YAML中的配置项
    3. 配置验证（路径存在性、参数合法性）
    4. 返回ConfigBundle数据类实例

使用示例:
    from config.config_loader import load_config
    config = load_config("config/default_config.yaml")
    print(config.model.fusion.type)  # -> "caugf"
============================================================================
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

# 尝试导入yaml，如果不存在则提示安装
try:
    import yaml
except ImportError:
    print("错误: 需要安装PyYAML. 请运行: pip install pyyaml")
    sys.exit(1)

# ============================================================
# 项目根目录（yu_hou_prognosis/）
# ============================================================
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_project_root() -> Path:
    """获取项目根目录的绝对路径"""
    return _PROJECT_ROOT


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class UpstreamConfig:
    """上游WSI patch数据配置"""
    enabled: bool = True
    patch_root: str = "../wsi_patch_selection_benchmark/outputs/patches"
    algorithm_name: str = "grid"
    file_pattern: str = "{slide_base}_{idx}.png"
    num_patches_per_patient: int = 6
    patch_size: int = 1024


@dataclass
class GenomicConfig:
    """基因组数据配置"""
    input_dim: int = 356
    survtime_col: str = "OS.time"
    censor_col: str = "OS"
    patient_id_col: str = "PatientID"
    max_rnaseq_features: int = 500
    max_other_features: int = 50


@dataclass
class FeatureSelectionConfig:
    """特征选择配置"""
    method: str = "variance_threshold"
    variance_threshold: float = 0.01
    n_features_mi: int = 200
    pca_variance_ratio: float = 0.95
    n_features_final: int = 100


@dataclass
class DataConfig:
    """数据路径配置"""
    root: str = "data/TCGA_COAD"
    genomic_csv: str = "data/TCGA_COAD/COAD_all_dataset.csv"
    split_file: str = "data/TCGA_COAD/splits/coad_allst_patient5fold.pkl"
    genomic: GenomicConfig = field(default_factory=GenomicConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)


@dataclass
class DenoiseConfig:
    """图像去噪配置"""
    method: str = "gaussian"
    kernel_size: int = 3
    sigma: float = 0.0


@dataclass
class ContrastConfig:
    """对比度增强配置"""
    method: str = "clahe"
    clip_limit: float = 2.0
    tile_grid_size: List[int] = field(default_factory=lambda: [8, 8])


@dataclass
class ColorNormConfig:
    """颜色归一化配置"""
    method: str = "reinhard"
    target_mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    target_std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])


@dataclass
class BlurFilterConfig:
    """模糊图像过滤配置"""
    enabled: bool = True
    method: str = "laplacian_variance"
    threshold: float = 100.0
    strategy: str = "mark_only"
    max_blur_ratio_per_patient: float = 0.5


@dataclass
class ImagePreprocessingConfig:
    """图像预处理配置"""
    enabled: bool = True
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    contrast: ContrastConfig = field(default_factory=ContrastConfig)
    color_normalization: ColorNormConfig = field(default_factory=ColorNormConfig)
    blur_filter: BlurFilterConfig = field(default_factory=BlurFilterConfig)


@dataclass
class PathConfig:
    """病理图像分支配置"""
    backbone: str = "resnet50"
    pretrained: str = "weights/resnet50-0676ba61.pth"
    dim: int = 32
    freeze_backbone: bool = True
    freeze_bn: bool = True
    input_size: int = 1024


@dataclass
class OmicConfig:
    """基因组分支配置"""
    input_dim: int = 356
    dim: int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [64, 48, 32])
    pretrained: str = ""


@dataclass
class CAUGFConfig:
    """CAUGF融合模块专用配置"""
    min_stream_weight: float = 0.12
    use_relation: bool = True
    relation_types: List[str] = field(default_factory=lambda: ["product", "difference", "cosine"])
    num_post_layers: int = 2
    use_layer_norm: bool = True
    temperature_init: float = 1.0


@dataclass
class CrossAttentionConfig:
    """交叉注意力融合专用配置"""
    num_heads: int = 4
    num_layers: int = 2
    ff_expansion: int = 4


@dataclass
class AttentionWeightedConfig:
    """注意力加权融合专用配置"""
    num_heads: int = 4
    use_residual: bool = True


@dataclass
class LMFConfig:
    """LMF融合专用配置"""
    rank: int = 4


@dataclass
class POFusionConfig:
    """双线性融合专用配置"""
    skip: int = 1
    use_bilinear: int = 1
    gate1: int = 1
    gate2: int = 1


@dataclass
class FusionConfig:
    """多模态融合配置"""
    type: str = "caugf"
    hidden_dim: int = 64
    output_dim: int = 64
    dropout: float = 0.25
    caugf: CAUGFConfig = field(default_factory=CAUGFConfig)
    cross_attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)
    attention_weighted: AttentionWeightedConfig = field(default_factory=AttentionWeightedConfig)
    lmf: LMFConfig = field(default_factory=LMFConfig)
    pofusion: POFusionConfig = field(default_factory=POFusionConfig)


@dataclass
class ModelConfig:
    """模型配置"""
    mode: str = "pathomic"
    name: str = "YuHou"
    task: str = "surv"
    seed: int = 2026
    path: PathConfig = field(default_factory=PathConfig)
    omic: OmicConfig = field(default_factory=OmicConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)


@dataclass
class OptimizerConfig:
    """优化器配置"""
    type: str = "adam"
    lr: float = 0.0001
    weight_decay: float = 0.0003
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    final_lr: float = 0.1


@dataclass
class SchedulerConfig:
    """学习率调度器配置"""
    type: str = "cosine"
    n_epochs: int = 8
    n_epochs_decay: int = 4
    warmup_epochs: int = 2
    min_lr: float = 1.0e-6
    lr_decay_iters: int = 10


@dataclass
class AMPConfig:
    """混合精度训练配置"""
    enabled: bool = True
    dtype: str = "float16"


@dataclass
class LossConfig:
    """损失函数配置"""
    lambda_cox: float = 1.0
    lambda_task: float = 1.0
    lambda_reg: float = 0.0003
    reg_type: str = "none"


@dataclass
class ClassificationLossConfig:
    """分类损失配置"""
    loss_type: str = "ce"
    use_class_weights: bool = True
    label_smoothing: float = 0.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass
class EarlyStoppingConfig:
    """早停策略配置"""
    enabled: bool = True
    monitor: str = "cindex"
    patience: int = 5
    min_delta: float = 0.001


@dataclass
class CheckpointConfig:
    """断点管理配置"""
    save_every: int = 1
    resume: bool = True
    keep_last_n: int = 3


@dataclass
class GPUMonitorConfig:
    """GPU监控配置"""
    enabled: bool = True
    oom_protection: bool = True
    memory_warning_threshold: float = 0.85
    log_interval: int = 10


@dataclass
class TrainingConfig:
    """训练配置"""
    device: str = "auto"
    gpu_id: int = 0
    gpu_memory_fraction: float = 0.0
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    non_blocking: bool = True
    prefetch_factor: int = 2
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    amp: AMPConfig = field(default_factory=AMPConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    classification: ClassificationLossConfig = field(default_factory=ClassificationLossConfig)
    gradient_clip_norm: float = 1.0
    finetune: bool = False
    act_type: str = "none"
    init_type: str = "none"
    init_gain: float = 0.02
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    gpu_monitor: GPUMonitorConfig = field(default_factory=GPUMonitorConfig)


@dataclass
class CVConfig:
    """交叉验证配置"""
    n_folds: int = 5
    fold_ids: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    shuffle: bool = True


@dataclass
class EvalMetricsConfig:
    """评估指标列表配置"""
    survival: List[str] = field(default_factory=lambda: [
        "c_index", "time_dependent_auc", "log_rank_pvalue",
        "hazard_ratio", "brier_score"
    ])
    classification: List[str] = field(default_factory=lambda: [
        "accuracy", "balanced_accuracy", "precision", "recall",
        "f1_score", "auc", "average_precision", "mcc"
    ])


@dataclass
class EvaluationConfig:
    """评估配置"""
    eval_times: List[int] = field(default_factory=lambda: [12, 24, 36])
    n_bootstrap: int = 100
    average: str = "macro"
    per_fold_details: bool = True
    metrics: EvalMetricsConfig = field(default_factory=EvalMetricsConfig)


@dataclass
class KMCurveConfig:
    """KM曲线可视化配置"""
    xlabel: str = "时间 (月)"
    ylabel: str = "总体生存率 (OS)"
    title: str = "Kaplan-Meier生存曲线"
    show_ci: bool = True
    show_risk_table: bool = True
    color_high_risk: str = "#e74c3c"
    color_low_risk: str = "#2ecc71"


@dataclass
class ROCCurveConfig:
    """ROC曲线可视化配置"""
    title: str = "时间依赖ROC曲线"


@dataclass
class CalibrationCurveConfig:
    """校准曲线可视化配置"""
    title: str = "校准曲线"
    n_bins: int = 10


@dataclass
class RiskHistogramConfig:
    """风险直方图可视化配置"""
    title: str = "预测风险分布"
    n_bins: int = 30


@dataclass
class VisualizationConfig:
    """可视化配置"""
    enabled: bool = True
    dpi: int = 300
    formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    km_curve: KMCurveConfig = field(default_factory=KMCurveConfig)
    roc_curve: ROCCurveConfig = field(default_factory=ROCCurveConfig)
    calibration_curve: CalibrationCurveConfig = field(default_factory=CalibrationCurveConfig)
    risk_histogram: RiskHistogramConfig = field(default_factory=RiskHistogramConfig)


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    language: str = "zh_CN"
    format: str = "%(asctime)s [%(levelname)s] %(message)s"
    file: str = "logs/train.log"
    console: bool = True
    log_model_arch: bool = True
    log_gpu_info: bool = True


@dataclass
class ExperimentConfig:
    """实验管理配置"""
    name: str = "default"
    output_root: str = "experiments"
    description: str = "默认实验配置"
    subdirs: Dict[str, str] = field(default_factory=lambda: {
        "ckpt": "ckpt",
        "logs": "logs",
        "resume": "resume",
        "preds": "preds",
        "results": "results",
        "figures": "figures",
    })
    save_predictions: bool = True
    save_epoch_predictions: bool = False


@dataclass
class ProjectConfig:
    """项目基本信息配置"""
    name: str = "YuHou_Multimodal_Prognosis"
    version: str = "2.0.0"
    description: str = "结直肠癌COAD多模态预后生存预测"


@dataclass
class ConfigBundle:
    """
    全局配置聚合类

    所有配置子模块的聚合，提供统一访问接口。
    """
    project: ProjectConfig
    upstream: UpstreamConfig
    data: DataConfig
    image_preprocessing: ImagePreprocessingConfig
    model: ModelConfig
    training: TrainingConfig
    cross_validation: CVConfig
    evaluation: EvaluationConfig
    visualization: VisualizationConfig
    logging: LoggingConfig
    experiment: ExperimentConfig

    # 运行时注入的额外属性
    project_root: Path = field(default_factory=get_project_root)
    config_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典（用于序列化/日志记录）"""
        return asdict(self)

    def get_exp_dir(self) -> Path:
        """获取实验输出目录的绝对路径"""
        exp_dir = self.project_root / self.experiment.output_root / self.experiment.name
        return exp_dir.resolve()

    def get_subdir(self, key: str) -> Path:
        """获取实验子目录的绝对路径"""
        exp_dir = self.get_exp_dir()
        subdir_name = self.experiment.subdirs.get(key, key)
        subdir = exp_dir / subdir_name
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir

    def resolve_path(self, path: str) -> Path:
        """
        将配置中的相对路径解析为绝对路径

        规则:
            1. 如果已经是绝对路径，直接返回
            2. 否则，相对于项目根目录解析
        """
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()


# ============================================================
# YAML配置加载
# ============================================================

def _deep_merge(base: Dict, override: Dict) -> Dict:
    """深度合并两个字典，override中的值覆盖base中的值"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_dataclass(cls, data: Dict):
    """递归地将字典转换为嵌套的数据类实例"""
    if data is None:
        return None

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key not in field_types:
            continue

        field_type = field_types[key]

        # 检查是否是嵌套的数据类
        if hasattr(field_type, '__dataclass_fields__') and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(field_type, value)
        # 检查是否是 List[dataclass] 类型（暂不处理，当前配置无此情况）
        elif isinstance(value, dict) and hasattr(field_type, '__origin__'):
            # 处理 Optional / Union 类型
            origin = getattr(field_type, '__origin__', None)
            if origin is Union:
                # 尝试找到数据类类型
                for arg in getattr(field_type, '__args__', []):
                    if hasattr(arg, '__dataclass_fields__') and isinstance(value, dict):
                        kwargs[key] = _dict_to_dataclass(arg, value)
                        break
                else:
                    kwargs[key] = value
            else:
                kwargs[key] = value
        else:
            kwargs[key] = value

    return cls(**kwargs)


def _parse_cli_overrides(args: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    解析命令行参数，返回需要覆盖的配置字典。

    支持的格式:
        --training.batch_size 8
        --model.fusion.type caugf
        --experiment.name my_exp

    也支持短参数:
        --config / -c: 指定配置文件路径
        --mode: 运行模式 (eda / train / test / ablation)
    """
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="YuHou多模态预后预测 - 命令行参数",
        add_help=False,
    )

    parser.add_argument("--config", "-c", type=str, default=None,
                        help="配置文件路径 (默认: config/default_config.yaml)")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["eda", "train", "test", "ablation"],
                        help="运行模式")
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="GPU卡号")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="批次大小")
    parser.add_argument("--fusion_type", type=str, default=None,
                        help="融合策略类型")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="实验名称")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率")
    parser.add_argument("--n_epochs", type=int, default=None,
                        help="训练轮数")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")

    # 支持任意 --key.subkey value 格式的参数
    parsed, unknown = parser.parse_known_args(args)

    # 处理key.subkey格式的参数
    overrides = {}

    # 标准参数映射
    if parsed.gpu_id is not None:
        overrides["training.gpu_id"] = parsed.gpu_id
    if parsed.batch_size is not None:
        overrides["training.batch_size"] = parsed.batch_size
    if parsed.fusion_type is not None:
        overrides["model.fusion.type"] = parsed.fusion_type
    if parsed.exp_name is not None:
        overrides["experiment.name"] = parsed.exp_name
    if parsed.lr is not None:
        overrides["training.optimizer.lr"] = parsed.lr
    if parsed.n_epochs is not None:
        overrides["training.scheduler.n_epochs"] = parsed.n_epochs
    if parsed.seed is not None:
        overrides["model.seed"] = parsed.seed

    return overrides, parsed


def _apply_dot_path_override(config_dict: Dict, dot_path: str, value: Any) -> Dict:
    """
    将 "training.batch_size" 形式的覆盖路径应用到嵌套字典中。

    参数:
        config_dict: 原始配置字典
        dot_path: 点号分隔的路径，如 "model.fusion.type"
        value: 要设置的值

    返回:
        修改后的配置字典
    """
    keys = dot_path.split(".")
    current = config_dict

    for i, key in enumerate(keys[:-1]):
        if key not in current:
            current[key] = {}
        if not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    # 尝试类型转换
    last_key = keys[-1]
    if last_key in current and current[last_key] is not None:
        try:
            orig_type = type(current[last_key])
            if orig_type == bool and isinstance(value, str):
                value = value.lower() in ("true", "1", "yes")
            else:
                value = orig_type(value)
        except (ValueError, TypeError):
            pass

    current[last_key] = value
    return config_dict


def load_config(config_path: Optional[str] = None,
                cli_args: Optional[List[str]] = None) -> ConfigBundle:
    """
    加载配置文件并返回ConfigBundle实例。

    参数:
        config_path: YAML配置文件路径（相对于项目根目录或绝对路径）。
                     如果为None，使用默认路径 config/default_config.yaml
        cli_args: 命令行参数列表。如果为None，使用 sys.argv[1:]

    返回:
        ConfigBundle: 包含所有配置的数据类实例

    异常:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML格式错误
        ValueError: 配置验证失败
    """
    # 解析命令行参数
    cli_overrides, parsed_args = _parse_cli_overrides(cli_args)

    # 确定配置文件路径
    if config_path is None:
        config_path = parsed_args.config or "config/default_config.yaml"

    config_abs_path = _PROJECT_ROOT / config_path
    if not config_abs_path.is_file():
        raise FileNotFoundError(
            f"配置文件不存在: {config_abs_path}\n"
            f"请确保配置文件路径正确，或使用 --config 参数指定配置文件。"
        )

    # 加载YAML
    with open(config_abs_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raise ValueError(f"配置文件为空: {config_abs_path}")

    # 应用命令行覆盖
    for dot_path, value in cli_overrides.items():
        raw_config = _apply_dot_path_override(raw_config, dot_path, value)

    # 构建ConfigBundle
    config = ConfigBundle(
        project=_dict_to_dataclass(ProjectConfig, raw_config.get("project", {})),
        upstream=_dict_to_dataclass(UpstreamConfig, raw_config.get("upstream", {})),
        data=_dict_to_dataclass(DataConfig, raw_config.get("data", {})),
        image_preprocessing=_dict_to_dataclass(
            ImagePreprocessingConfig, raw_config.get("image_preprocessing", {})
        ),
        model=_dict_to_dataclass(ModelConfig, raw_config.get("model", {})),
        training=_dict_to_dataclass(TrainingConfig, raw_config.get("training", {})),
        cross_validation=_dict_to_dataclass(CVConfig, raw_config.get("cross_validation", {})),
        evaluation=_dict_to_dataclass(EvaluationConfig, raw_config.get("evaluation", {})),
        visualization=_dict_to_dataclass(VisualizationConfig, raw_config.get("visualization", {})),
        logging=_dict_to_dataclass(LoggingConfig, raw_config.get("logging", {})),
        experiment=_dict_to_dataclass(ExperimentConfig, raw_config.get("experiment", {})),
        config_path=str(config_abs_path),
    )

    # 验证配置
    _validate_config(config)

    return config


def _validate_config(config: ConfigBundle) -> None:
    """
    验证配置的合法性。

    检查项:
        1. 融合策略类型是否在支持列表中
        2. 必要的路径是否存在
        3. 参数范围是否合法
    """
    # 验证融合策略类型
    valid_fusion_types = [
        "pofusion", "lmf", "concat", "gmu", "film",
        "caugf", "attention_weighted", "cross_attention", "tensor_concat"
    ]
    if config.model.fusion.type not in valid_fusion_types:
        raise ValueError(
            f"不支持的融合策略类型: '{config.model.fusion.type}'。"
            f"支持的类型: {valid_fusion_types}"
        )

    # 验证任务类型
    valid_tasks = ["surv", "ncls", "n_binary"]
    if config.model.task not in valid_tasks:
        raise ValueError(
            f"不支持的任务类型: '{config.model.task}'。"
            f"支持的类型: {valid_tasks}"
        )

    # 验证上游patch路径（仅当enabled时）
    if config.upstream.enabled:
        patch_root = config.resolve_path(config.upstream.patch_root)
        if not patch_root.exists():
            print(f"警告: 上游patch根目录不存在: {patch_root}")
            print(f"  如果尚未运行上游WSI切块任务，请先执行: cd ../wsi_patch_selection_benchmark && python run_patch_selection.py")

    # 验证基因组CSV文件
    genomic_csv = config.resolve_path(config.data.genomic_csv)
    if not genomic_csv.exists():
        print(f"警告: 基因组数据文件不存在: {genomic_csv}")
        print(f"  请确保COAD_all_dataset.csv位于 data/TCGA_COAD/ 目录下")

    # 验证预训练权重
    pretrained_path = config.resolve_path(config.model.path.pretrained)
    if not pretrained_path.exists():
        print(f"警告: 预训练权重不存在: {pretrained_path}")
        print(f"  模型将使用随机初始化")

    # 验证batch_size
    if config.training.batch_size < 1:
        raise ValueError(f"batch_size 必须 >= 1, 当前值: {config.training.batch_size}")

    # 验证梯度累积步数
    if config.training.gradient_accumulation_steps < 1:
        raise ValueError(
            f"gradient_accumulation_steps 必须 >= 1, "
            f"当前值: {config.training.gradient_accumulation_steps}"
        )

    # 验证评估时间点
    for t in config.evaluation.eval_times:
        if t <= 0:
            raise ValueError(f"评估时间点必须 > 0, 当前值: {t}")


def create_experiment_dirs(config: ConfigBundle) -> Dict[str, Path]:
    """
    创建实验输出目录结构。

    返回:
        Dict[str, Path]: 子目录路径字典
    """
    dirs = {}
    for key, subdir_name in config.experiment.subdirs.items():
        dir_path = config.get_subdir(key)
        dirs[key] = dir_path

    return dirs


# ============================================================
# 便捷函数
# ============================================================

def print_config(config: ConfigBundle, logger: Optional[logging.Logger] = None) -> None:
    """
    打印配置摘要（用于训练日志）。

    参数:
        config: ConfigBundle实例
        logger: 可选的logger，如果为None则print到stdout
    """
    lines = []
    lines.append("=" * 70)
    lines.append("YuHou多模态预后预测 - 配置摘要")
    lines.append("=" * 70)
    lines.append(f"  项目版本     : {config.project.version}")
    lines.append(f"  实验名称     : {config.experiment.name}")
    lines.append(f"  配置文件     : {config.config_path}")
    lines.append("-" * 70)
    lines.append(f"  任务类型     : {config.model.task}")
    lines.append(f"  运行模式     : {config.model.mode}")
    lines.append(f"  融合策略     : {config.model.fusion.type}")
    lines.append(f"  批次大小     : {config.training.batch_size}")
    lines.append(f"  学习率       : {config.training.optimizer.lr}")
    lines.append(f"  总训练轮数   : {config.training.scheduler.n_epochs + config.training.scheduler.n_epochs_decay}")
    lines.append(f"  AMP混合精度  : {config.training.amp.enabled}")
    lines.append(f"  早停策略     : {'启用' if config.training.early_stopping.enabled else '禁用'} "
                 f"(patience={config.training.early_stopping.patience})")
    lines.append(f"  正则化       : {config.training.loss.reg_type} "
                 f"(lambda={config.training.loss.lambda_reg})")
    lines.append(f"  GPU ID       : {config.training.gpu_id}")
    lines.append("-" * 70)
    lines.append(f"  基因组特征   : {config.model.omic.input_dim}维 -> {config.data.feature_selection.n_features_final}维(筛选后)")
    lines.append(f"  图像预处理   : {config.image_preprocessing.enabled}")
    lines.append(f"  模糊过滤     : {config.image_preprocessing.blur_filter.enabled}")
    lines.append(f"  上游算法     : {config.upstream.algorithm_name}")
    lines.append("-" * 70)
    lines.append(f"  评估时间点   : {config.evaluation.eval_times} (月)")
    lines.append(f"  交叉验证折数 : {config.cross_validation.n_folds}")
    lines.append("=" * 70)

    text = "\n".join(lines)

    if logger:
        logger.info("\n" + text)
    else:
        print(text)


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print(f"项目根目录: {get_project_root()}")

    try:
        config = load_config()
        print_config(config)
        print("\n配置加载成功!")

        # 打印实验输出目录
        print(f"\n实验输出目录: {config.get_exp_dir()}")
        for key in config.experiment.subdirs:
            print(f"  {key}: {config.get_subdir(key)}")

    except Exception as e:
        print(f"配置加载失败: {e}")
        import traceback
        traceback.print_exc()
