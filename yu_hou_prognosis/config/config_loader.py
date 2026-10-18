# -*- coding: utf-8 -*-
"""
config_loader.py
============================================================================
配置加载器：从YAML配置文件加载所有参数，支持命令行覆盖。

功能:
    1. 从YAML文件加载默认配置
    2. 命令行参数覆盖YAML中的配置项
    3. 配置验证（路径存在性、参数合法性）
    4. 返回ConfigBundle数据类实例

使用示例:
    from config.config_loader import load_config
    config = load_config("config/default_config.yaml")
    print(config.model.task_type)     # -> "classification"
    print(config.model.target_gene)   # -> "REG4"
============================================================================
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

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
    """基因组/表格数据配置"""
    input_dim: int = 0
    patient_id_col: str = "TCGA ID"
    survtime_col: str = "Survival months"
    censor_col: str = "event"
    target_genes: List[str] = field(default_factory=lambda: ["REG4", "OLFM4", "MUC2", "DEFA5"])
    covariate_cols: List[str] = field(default_factory=list)
    rnaseq_suffix: str = "_rnaseq"
    max_rnaseq_features: int = 500
    max_other_features: int = 50


@dataclass
class FeatureSelectionConfig:
    """特征选择配置（保留兼容，新任务默认不使用）"""
    method: str = "none"
    variance_threshold: float = 0.01
    n_features_mi: int = 200
    pca_variance_ratio: float = 0.95
    n_features_final: int = 100


@dataclass
class DataConfig:
    """数据路径配置"""
    root: str = "data"
    genomic_csv: str = "data/COAD_all_dataset.csv"
    split_file: str = "data/splits/coad_allst_patient5fold.pkl"
    genomic: GenomicConfig = field(default_factory=GenomicConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)


@dataclass
class DenoiseConfig:
    method: str = "gaussian"
    kernel_size: int = 3
    sigma: float = 0.0


@dataclass
class ContrastConfig:
    method: str = "clahe"
    clip_limit: float = 2.0
    tile_grid_size: List[int] = field(default_factory=lambda: [8, 8])


@dataclass
class ColorNormConfig:
    method: str = "reinhard"
    target_mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    target_std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])


@dataclass
class BlurFilterConfig:
    enabled: bool = False
    method: str = "laplacian_variance"
    threshold: float = 100.0
    strategy: str = "mark_only"
    max_blur_ratio_per_patient: float = 0.5


@dataclass
class ImagePreprocessingConfig:
    enabled: bool = False
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    contrast: ContrastConfig = field(default_factory=ContrastConfig)
    color_normalization: ColorNormConfig = field(default_factory=ColorNormConfig)
    blur_filter: BlurFilterConfig = field(default_factory=BlurFilterConfig)


@dataclass
class PathConfig:
    backbone: str = "resnet50"
    pretrained: str = "weights/resnet50-0676ba61.pth"
    dim: int = 32
    freeze_backbone: bool = True
    freeze_bn: bool = True
    input_size: int = 1024
    # 部分微调起点: fc(仅fc) / layer4 / layer3 / layer2 / all
    unfreeze_from: str = "fc"


@dataclass
class OmicConfig:
    input_dim: int = 0
    dim: int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [64, 48, 32])
    pretrained: str = ""


@dataclass
class CAUGFConfig:
    min_stream_weight: float = 0.12
    use_relation: bool = True
    relation_types: List[str] = field(default_factory=lambda: ["product", "difference", "cosine"])
    num_post_layers: int = 2
    use_layer_norm: bool = True
    temperature_init: float = 1.0


@dataclass
class CrossAttentionConfig:
    num_heads: int = 4
    num_layers: int = 2
    ff_expansion: int = 4


@dataclass
class AttentionWeightedConfig:
    num_heads: int = 4
    use_residual: bool = True


@dataclass
class LMFConfig:
    rank: int = 4


@dataclass
class POFusionConfig:
    skip: int = 1
    use_bilinear: int = 1
    gate1: int = 1
    gate2: int = 1


@dataclass
class FusionConfig:
    type: str = "caugf"
    hidden_dim: int = 64
    output_dim: int = 64
    dropout: float = 0.35
    caugf: CAUGFConfig = field(default_factory=CAUGFConfig)
    cross_attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)
    attention_weighted: AttentionWeightedConfig = field(default_factory=AttentionWeightedConfig)
    lmf: LMFConfig = field(default_factory=LMFConfig)
    pofusion: POFusionConfig = field(default_factory=POFusionConfig)


@dataclass
class ModelConfig:
    mode: str = "path"              # path / omic / pathomic
    name: str = "Histomolecular"
    task_type: str = "classification"  # classification / regression
    target_gene: str = "REG4"
    group_threshold: str = "median"    # median / mean / quantile / 数值
    label_mode: str = "binary"
    seed: int = 2026
    path: PathConfig = field(default_factory=PathConfig)
    omic: OmicConfig = field(default_factory=OmicConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)


@dataclass
class OptimizerConfig:
    type: str = "adamw"
    lr: float = 0.0001
    weight_decay: float = 0.001
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    final_lr: float = 0.1


@dataclass
class SchedulerConfig:
    type: str = "cosine"
    n_epochs: int = 8
    n_epochs_decay: int = 4
    warmup_epochs: int = 2
    min_lr: float = 1.0e-6
    lr_decay_iters: int = 10


@dataclass
class AMPConfig:
    enabled: bool = False
    dtype: str = "float16"


@dataclass
class LossConfig:
    task_loss: str = "ce"     # ce / bce / focal / mse
    lambda_task: float = 1.0
    lambda_reg: float = 0.0003
    reg_type: str = "path"


@dataclass
class ClassificationLossConfig:
    loss_type: str = "ce"
    use_class_weights: bool = True
    label_smoothing: float = 0.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    monitor: str = "auc"      # auc / accuracy / f1 / loss / pearson / spearman
    patience: int = 5
    min_delta: float = 0.001


@dataclass
class CheckpointConfig:
    save_every: int = 1
    resume: bool = True
    keep_last_n: int = 3


@dataclass
class GPUMonitorConfig:
    enabled: bool = True
    oom_protection: bool = True
    memory_warning_threshold: float = 0.85
    log_interval: int = 10


@dataclass
class TrainingConfig:
    device: str = "auto"
    gpu_id: int = 0
    gpu_memory_fraction: float = 0.0
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_workers: int = 0
    pin_memory: bool = False
    non_blocking: bool = False
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
    n_folds: int = 5
    fold_ids: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    shuffle: bool = True
    val_ratio: float = 0.2


@dataclass
class EvalMetricsConfig:
    classification: List[str] = field(default_factory=lambda: [
        "accuracy", "balanced_accuracy", "precision", "recall",
        "f1_score", "auc", "average_precision", "mcc"
    ])
    regression: List[str] = field(default_factory=lambda: [
        "mse", "mae", "rmse", "pearson", "spearman"
    ])


@dataclass
class AuxiliarySurvivalConfig:
    enabled: bool = True
    split_method: str = "median"


@dataclass
class EvaluationConfig:
    n_bootstrap: int = 100
    average: str = "binary"
    per_fold_details: bool = True
    metrics: EvalMetricsConfig = field(default_factory=EvalMetricsConfig)
    auxiliary_survival: AuxiliarySurvivalConfig = field(default_factory=AuxiliarySurvivalConfig)


@dataclass
class KMCurveConfig:
    xlabel: str = "时间 (月)"
    ylabel: str = "总体生存率 (OS)"
    title: str = "Kaplan-Meier生存曲线(按预测分子表型分组)"
    show_ci: bool = True
    show_risk_table: bool = False
    color_high_risk: str = "#e74c3c"
    color_low_risk: str = "#2ecc71"


@dataclass
class ROCCurveConfig:
    title: str = "ROC曲线"


@dataclass
class ConfusionMatrixConfig:
    title: str = "混淆矩阵"


@dataclass
class CalibrationCurveConfig:
    title: str = "校准曲线"
    n_bins: int = 10


@dataclass
class VisualizationConfig:
    enabled: bool = True
    dpi: int = 300
    formats: List[str] = field(default_factory=lambda: ["png"])
    roc_curve: ROCCurveConfig = field(default_factory=ROCCurveConfig)
    confusion_matrix: ConfusionMatrixConfig = field(default_factory=ConfusionMatrixConfig)
    km_curve: KMCurveConfig = field(default_factory=KMCurveConfig)
    calibration_curve: CalibrationCurveConfig = field(default_factory=CalibrationCurveConfig)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    language: str = "zh_CN"
    format: str = "%(asctime)s [%(levelname)s] %(message)s"
    file: str = "logs/train.log"
    console: bool = True
    log_model_arch: bool = True
    log_gpu_info: bool = True


@dataclass
class ExperimentConfig:
    name: str = "default"
    output_root: str = "experiments"
    description: str = "Histomolecular基因表达预测实验"
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
    name: str = "DealWith_Histomolecular"
    version: str = "3.0.0"
    description: str = "结直肠癌WSI病理图像预测肠分化/粘液相关基因表达"


@dataclass
class AblationConfig:
    algorithms: List[str] = field(default_factory=lambda: [
        "grid", "random", "kmeans", "yottixel", "stratified",
        "sentinel", "splice", "sdm", "largest_tissue"
    ])
    target_genes: List[str] = field(default_factory=list)


@dataclass
class ConfigBundle:
    """全局配置聚合类"""
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
    ablation: AblationConfig

    project_root: Path = field(default_factory=get_project_root)
    config_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_exp_dir(self) -> Path:
        exp_dir = self.project_root / self.experiment.output_root / self.experiment.name
        return exp_dir.resolve()

    def get_subdir(self, key: str) -> Path:
        exp_dir = self.get_exp_dir()
        subdir_name = self.experiment.subdirs.get(key, key)
        subdir = exp_dir / subdir_name
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir

    def resolve_path(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()


# ============================================================
# YAML配置加载
# ============================================================

def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_dataclass(cls, data: Dict):
    if data is None:
        return None

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key not in field_types:
            continue

        field_type = field_types[key]

        if hasattr(field_type, '__dataclass_fields__') and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(field_type, value)
        elif isinstance(value, dict) and hasattr(field_type, '__origin__'):
            origin = getattr(field_type, '__origin__', None)
            if origin is Union:
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
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None,
                        choices=["eda", "train", "test", "ablation"])
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--fusion_type", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target_gene", type=str, default=None)
    parser.add_argument("--target_genes", type=str, nargs="+", default=None)
    parser.add_argument("--task_type", type=str, default=None,
                        choices=["classification", "regression"])
    parser.add_argument("--algorithm_name", type=str, default=None)

    parsed, _unknown = parser.parse_known_args(args)

    overrides = {}
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
    if parsed.target_gene is not None:
        overrides["model.target_gene"] = parsed.target_gene
    if parsed.target_genes is not None:
        overrides["data.genomic.target_genes"] = parsed.target_genes
    if parsed.task_type is not None:
        overrides["model.task_type"] = parsed.task_type
    if parsed.algorithm_name is not None:
        overrides["upstream.algorithm_name"] = parsed.algorithm_name

    return overrides, parsed


def _apply_dot_path_override(config_dict: Dict, dot_path: str, value: Any) -> Dict:
    keys = dot_path.split(".")
    current = config_dict

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        if not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    last_key = keys[-1]
    if last_key in current and current[last_key] is not None:
        try:
            orig_type = type(current[last_key])
            if orig_type == bool and isinstance(value, str):
                value = value.lower() in ("true", "1", "yes")
            elif orig_type == list:
                value = value if isinstance(value, list) else [value]
            else:
                value = orig_type(value)
        except (ValueError, TypeError):
            pass

    current[last_key] = value
    return config_dict


def load_config(config_path: Optional[str] = None,
                cli_args: Optional[List[str]] = None) -> ConfigBundle:
    cli_overrides, parsed_args = _parse_cli_overrides(cli_args)

    if config_path is None:
        config_path = parsed_args.config or "config/default_config.yaml"

    config_abs_path = _PROJECT_ROOT / config_path
    if not config_abs_path.is_file():
        raise FileNotFoundError(
            f"配置文件不存在: {config_abs_path}\n"
            f"请确保配置文件路径正确，或使用 --config 参数指定配置文件。"
        )

    with open(config_abs_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raise ValueError(f"配置文件为空: {config_abs_path}")

    for dot_path, value in cli_overrides.items():
        raw_config = _apply_dot_path_override(raw_config, dot_path, value)

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
        ablation=_dict_to_dataclass(AblationConfig, raw_config.get("ablation", {})),
        config_path=str(config_abs_path),
    )

    _validate_config(config)

    return config


def _validate_config(config: ConfigBundle) -> None:
    # 融合策略类型
    valid_fusion_types = [
        "pofusion", "lmf", "concat", "gmu", "film",
        "caugf", "attention_weighted", "cross_attention", "tensor_concat"
    ]
    if config.model.fusion.type not in valid_fusion_types:
        raise ValueError(
            f"不支持的融合策略类型: '{config.model.fusion.type}'。"
            f"支持的类型: {valid_fusion_types}"
        )

    # 任务类型
    valid_task_types = ["classification", "regression"]
    if config.model.task_type not in valid_task_types:
        raise ValueError(
            f"不支持的任务类型: '{config.model.task_type}'。"
            f"支持的类型: {valid_task_types}"
        )

    # 模式
    valid_modes = ["path", "omic", "pathomic"]
    if config.model.mode not in valid_modes:
        raise ValueError(f"不支持的运行模式: '{config.model.mode}'。支持: {valid_modes}")

    # 上游patch路径
    if config.upstream.enabled:
        patch_root = config.resolve_path(config.upstream.patch_root)
        if not patch_root.exists():
            print(f"警告: 上游patch根目录不存在: {patch_root}")
            print(f"  如果尚未运行上游WSI切块任务，请先执行上游 run_patch_selection.py")

    # 表格文件
    genomic_csv = config.resolve_path(config.data.genomic_csv)
    if not genomic_csv.exists():
        print(f"警告: 表格数据文件不存在: {genomic_csv}")
        print(f"  请确保包含TCGA ID + 临床信息 + 目标基因rnaseq表达值的CSV存在")

    # 目标基因合法性 (若CSV存在则检查)
    if genomic_csv.exists():
        try:
            import csv as _csv
            with open(genomic_csv, "r", encoding="utf-8-sig") as f:
                header = next(_csv.reader(f))
            headers = [h.lower() for h in header]
            suffix = config.data.genomic.rnaseq_suffix.lower()
            target = config.model.target_gene
            expected = (target + config.data.genomic.rnaseq_suffix).lower()
            alt = target.lower()
            if expected not in headers and alt not in headers:
                print(f"警告: 未在CSV中找到目标基因 '{target}' 对应的表达列。")
                print(f"  期望列名形如: {target}{config.data.genomic.rnaseq_suffix} "
                      f"或精确匹配 {target}")
        except Exception:
            pass

    # 预训练权重
    pretrained_path = config.resolve_path(config.model.path.pretrained)
    if not pretrained_path.exists():
        print(f"警告: 预训练权重不存在: {pretrained_path}")
        print(f"  模型将使用随机初始化")

    if config.training.batch_size < 1:
        raise ValueError(f"batch_size 必须 >= 1, 当前值: {config.training.batch_size}")

    if config.training.gradient_accumulation_steps < 1:
        raise ValueError(
            f"gradient_accumulation_steps 必须 >= 1, "
            f"当前值: {config.training.gradient_accumulation_steps}"
        )


def create_experiment_dirs(config: ConfigBundle) -> Dict[str, Path]:
    dirs = {}
    for key, subdir_name in config.experiment.subdirs.items():
        dirs[key] = config.get_subdir(key)
    return dirs


# ============================================================
# 便捷函数
# ============================================================

def print_config(config: ConfigBundle, logger: Optional[logging.Logger] = None) -> None:
    lines = []
    lines.append("=" * 70)
    lines.append("DealWith Histomolecular - 配置摘要")
    lines.append("=" * 70)
    lines.append(f"  项目版本     : {config.project.version}")
    lines.append(f"  实验名称     : {config.experiment.name}")
    lines.append(f"  配置文件     : {config.config_path}")
    lines.append("-" * 70)
    lines.append(f"  任务类型     : {config.model.task_type}")
    lines.append(f"  目标基因     : {config.model.target_gene}")
    lines.append(f"  分组阈值     : {config.model.group_threshold}")
    lines.append(f"  运行模式     : {config.model.mode}")
    lines.append(f"  融合策略     : {config.model.fusion.type}")
    lines.append(f"  批次大小     : {config.training.batch_size}")
    lines.append(f"  学习率       : {config.training.optimizer.lr}")
    lines.append(f"  早停监控     : {config.training.early_stopping.monitor}")
    lines.append(f"  上游算法     : {config.upstream.algorithm_name}")
    lines.append(f"  协变量数     : {len(config.data.genomic.covariate_cols)}")
    lines.append("-" * 70)
    lines.append(f"  交叉验证折数 : {config.cross_validation.n_folds}")
    lines.append(f"  附属生存分析 : {'启用' if config.evaluation.auxiliary_survival.enabled else '禁用'}")
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
        print(f"\n实验输出目录: {config.get_exp_dir()}")
    except Exception as e:
        print(f"配置加载失败: {e}")
        import traceback
        traceback.print_exc()
