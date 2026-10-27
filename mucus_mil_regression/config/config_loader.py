# -*- coding: utf-8 -*-
"""配置加载器 - MucusMIL Regression (yaml驱动 + 命令行覆盖)。"""

import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
except ImportError:
    print("需要 PyYAML: pip install pyyaml")
    sys.exit(1)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_project_root() -> Path:
    return _PROJECT_ROOT


@dataclass
class UpstreamConfig:
    patch_root: str = "../wsi_patch_selection_benchmark/outputs/patches"
    algorithm_name: str = "grid"
    file_pattern: str = "{slide_base}_{idx}.png"
    num_patches_per_patient: int = 6
    patch_size: int = 224


@dataclass
class PCAConfig:
    n_components: int = 1
    standardize: bool = True
    flip_sign_positive: bool = True
    save_loadings: bool = True


@dataclass
class DataConfig:
    root: str = "data"
    table_csv: str = "data/COAD_all_dataset.csv"
    label_csv: str = "data/mucus_pc1_labels.csv"
    split_file: str = "data/splits/mucus_patient5fold.pkl"
    patient_id_col: str = "TCGA ID"
    genes: List[str] = field(default_factory=lambda: ["REG1A_rnaseq", "REG4_rnaseq", "SPINK4_rnaseq", "ITLN1_rnaseq"])
    clinical_features: List[str] = field(default_factory=lambda: [
        "age_at_diagnosis", "gender", "tumor_stage", "t", "n", "m", "histological_type"
    ])
    pca: PCAConfig = field(default_factory=PCAConfig)


@dataclass
class ModelConfig:
    model_type: str = "image_mil"     # clinical_mlp / image_mil / multimodal
    backbone: str = "resnet"          # resnet / uni
    freeze_backbone: bool = True      # True=冻结主干只做特征提取
    backbone_pretrained: str = "weights/resnet50-0676ba61.pth"
    feat_dim: int = 2048
    mil_hidden_dim: int = 256
    mil_dropout: float = 0.3
    clinical_input_dim: int = 0        # 运行时由ClinicalEncoder输出维度填充
    clinical_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    clinical_dropout: float = 0.3
    fusion_type: str = "adaptive_gate"  # late_concat / adaptive_gate
    fusion_hidden_dim: int = 256


@dataclass
class OptimizerConfig:
    type: str = "adamw"
    lr: float = 0.0001
    weight_decay: float = 0.0001
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class SchedulerConfig:
    type: str = "cosine"
    n_epochs: int = 30
    warmup_epochs: int = 3
    min_lr: float = 1.0e-6


@dataclass
class AMPConfig:
    enabled: bool = True
    dtype: str = "float16"


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    monitor: str = "spearman"
    patience: int = 8
    min_delta: float = 0.0005


@dataclass
class CheckpointConfig:
    save_every: int = 5
    resume: bool = True
    keep_last_n: int = 2


@dataclass
class TrainingConfig:
    device: str = "auto"
    gpu_id: int = 0
    gpu_memory_fraction: float = 0.0
    batch_size: int = 2
    gradient_accumulation_steps: int = 2
    num_workers: int = 0
    pin_memory: bool = False
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    amp: AMPConfig = field(default_factory=AMPConfig)
    loss: str = "mse"
    gradient_clip_norm: float = 1.0
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass
class CVConfig:
    n_folds: int = 5
    fold_ids: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    val_ratio: float = 0.2


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: ["spearman", "pearson", "mae", "rmse", "r2"])


@dataclass
class VisualizationConfig:
    enabled: bool = True
    dpi: int = 300


@dataclass
class ExperimentConfig:
    name: str = "default"
    output_root: str = "experiments"
    subdirs: Dict[str, str] = field(default_factory=lambda: {
        "ckpt": "ckpt", "logs": "logs", "results": "results",
        "figures": "figures", "preds": "preds",
    })


@dataclass
class LoggingConfig:
    level: str = "INFO"
    language: str = "zh_CN"


@dataclass
class AblationConfig:
    algorithms: List[str] = field(default_factory=lambda: [
        "grid", "random", "kmeans", "yottixel", "stratified",
        "sentinel", "splice", "sdm", "largest_tissue"
    ])
    fusions: List[str] = field(default_factory=lambda: ["late_concat", "adaptive_gate"])
    model_types: List[str] = field(default_factory=lambda: ["clinical_mlp", "image_mil", "multimodal"])


@dataclass
class ProjectConfig:
    name: str = "MucusMIL_Regression"
    version: str = "1.0.0"
    seed: int = 2026


@dataclass
class ConfigBundle:
    project: ProjectConfig
    upstream: UpstreamConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    cross_validation: CVConfig
    evaluation: EvaluationConfig
    visualization: VisualizationConfig
    experiment: ExperimentConfig
    logging: LoggingConfig
    ablation: AblationConfig

    project_root: Path = field(default_factory=get_project_root)
    config_path: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    def get_exp_dir(self) -> Path:
        return (self.project_root / self.experiment.output_root / self.experiment.name).resolve()

    def get_subdir(self, key: str) -> Path:
        d = self.get_exp_dir() / self.experiment.subdirs.get(key, key)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def resolve_path(self, p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.project_root / p).resolve()


def _dict_to_dataclass(cls, data: Dict):
    if data is None:
        return None
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        if key not in field_types:
            continue
        ft = field_types[key]
        if hasattr(ft, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(ft, value)
        elif isinstance(value, dict) and hasattr(ft, "__origin__") and ft.__origin__ is Union:
            for arg in getattr(ft, "__args__", []):
                if hasattr(arg, "__dataclass_fields__") and isinstance(value, dict):
                    kwargs[key] = _dict_to_dataclass(arg, value)
                    break
            else:
                kwargs[key] = value
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(config_path: Optional[str] = None, cli_args: Optional[List[str]] = None) -> ConfigBundle:
    import argparse
    # allow_abbrev=False 防止 --mode 被 argparse 前缀匹配成 --model_type
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--backbone_pretrained", type=str, default=None)
    parser.add_argument("--fusion_type", type=str, default=None)
    parser.add_argument("--algorithm_name", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parsed, _ = parser.parse_known_args(cli_args if cli_args is not None else sys.argv[1:])

    config_path = config_path or parsed.config or "config/default_config.yaml"
    cfg_path = _PROJECT_ROOT / config_path
    if not cfg_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    def _apply(dot, val):
        keys = dot.split(".")
        cur = raw
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = val

    if parsed.model_type:
        _apply("model.model_type", parsed.model_type)
    if parsed.backbone:
        _apply("model.backbone", parsed.backbone)
    if parsed.backbone_pretrained:
        _apply("model.backbone_pretrained", parsed.backbone_pretrained)
    if parsed.fusion_type:
        _apply("model.fusion_type", parsed.fusion_type)
    if parsed.algorithm_name:
        _apply("upstream.algorithm_name", parsed.algorithm_name)
    if parsed.exp_name:
        _apply("experiment.name", parsed.exp_name)
    if parsed.gpu_id is not None:
        _apply("training.gpu_id", parsed.gpu_id)
    if parsed.batch_size:
        _apply("training.batch_size", parsed.batch_size)
    if parsed.lr:
        _apply("training.optimizer.lr", parsed.lr)
    if parsed.n_epochs is not None:
        _apply("training.scheduler.n_epochs", parsed.n_epochs)

    config = ConfigBundle(
        project=_dict_to_dataclass(ProjectConfig, raw.get("project", {})),
        upstream=_dict_to_dataclass(UpstreamConfig, raw.get("upstream", {})),
        data=_dict_to_dataclass(DataConfig, raw.get("data", {})),
        model=_dict_to_dataclass(ModelConfig, raw.get("model", {})),
        training=_dict_to_dataclass(TrainingConfig, raw.get("training", {})),
        cross_validation=_dict_to_dataclass(CVConfig, raw.get("cross_validation", {})),
        evaluation=_dict_to_dataclass(EvaluationConfig, raw.get("evaluation", {})),
        visualization=_dict_to_dataclass(VisualizationConfig, raw.get("visualization", {})),
        experiment=_dict_to_dataclass(ExperimentConfig, raw.get("experiment", {})),
        logging=_dict_to_dataclass(LoggingConfig, raw.get("logging", {})),
        ablation=_dict_to_dataclass(AblationConfig, raw.get("ablation", {})),
        config_path=str(cfg_path),
    )
    _validate(config)
    return config


def _validate(config: ConfigBundle):
    if config.model.model_type not in ("clinical_mlp", "image_mil", "multimodal"):
        raise ValueError(f"不支持的model_type: {config.model.model_type}")
    if config.model.backbone not in ("resnet", "retccl", "uni", "phikon", "conch"):
        raise ValueError(f"不支持的backbone: {config.model.backbone}")
    if config.model.fusion_type not in ("late_concat", "adaptive_gate"):
        raise ValueError(f"不支持的fusion_type: {config.model.fusion_type}")


def create_experiment_dirs(config: ConfigBundle) -> Dict[str, Path]:
    return {k: config.get_subdir(k) for k in config.experiment.subdirs}


def auto_exp_name(config: ConfigBundle) -> str:
    """按「算法/模型类型」自动命名实验目录, 避免手动 --exp_name 混乱。

    返回形如: sentinel/image_mil_resnet, sentinel/multimodal_resnet_adaptive_gate
    """
    algo = config.upstream.algorithm_name
    mt = config.model.model_type
    if mt == "clinical_mlp":
        return f"{algo}/clinical_mlp"
    if mt == "image_mil":
        return f"{algo}/image_mil_{config.model.backbone}"
    return f"{algo}/multimodal_{config.model.backbone}_{config.model.fusion_type}"


def print_config(config: ConfigBundle, logger=None):
    lines = [
        "=" * 70,
        "MucusMIL Regression - 配置摘要",
        "=" * 70,
        f"  模型类型: {config.model.model_type}",
        f"  主干: {config.model.backbone}",
        f"  融合: {config.model.fusion_type}",
        f"  上游算法: {config.upstream.algorithm_name}",
        f"  patch尺寸: {config.upstream.patch_size}",
        f"  batch/accum: {config.training.batch_size} / {config.training.gradient_accumulation_steps}",
        f"  学习率: {config.training.optimizer.lr}",
        f"  早停监控: {config.training.early_stopping.monitor}",
        "=" * 70,
    ]
    text = "\n".join(lines)
    if logger:
        logger.info("\n" + text)
    else:
        print(text)


if __name__ == "__main__":
    c = load_config()
    print_config(c)
