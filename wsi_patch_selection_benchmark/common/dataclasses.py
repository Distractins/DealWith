# -*- coding: utf-8 -*-
"""WSI Patch Selection Benchmark 的核心数据结构。

原始脚本中所有非结构化字典均被替换为强类型数据类（dataclass），
以提供类型安全和 IDE 自动补全功能。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from common.enums import QCLevel, AlgorithmName, Status


# ============================================================
# 配置
# ============================================================

@dataclass
class ConfigBundle:
    """从 YAML 文件加载并经 CLI 覆盖的所有配置参数。

    Attributes:
        wsi_root: 包含 WSI (.svs) 文件的根目录。
        output_root: 所有输出结果的根目录。
        patch_size: 正方形图块的边长（像素）。
        patches_per_case: 每个病例的目标图块数量（K）。
        ds_mask: 组织掩码计算的下采样倍数。
        stride: 候选扫描的网格步长。
        seed: 全局随机种子，用于保证可复现性。
        only_dx1: 若为 True，则仅处理 DX1（诊断级）切片。
        innovation_weight: 创新性评分在 final_score 中的权重。
        min_center_distance_ratio: 空间多样性阈值比例。
        min_feature_distance: 特征空间多样性阈值。
        preselect_topk: 需要预筛选的组织占比最高的候选数。
        min_tissue_ratio_preselect: 预筛选的最低组织占比。
        candidate_pool_size: 每张切片的目标候选池大小。
        max_tries: 最大随机采样尝试次数（用于 random/sentinel）。
        enable_gland_irregularity: 是否计算腺体不规则度。
        enabled_samplers: 需要运行的采样器名称列表。
        n_bins_x: X 方向的箱数（分层采样）。
        n_bins_y: Y 方向的箱数（分层采样）。
        redundancy_lambda: 冗余惩罚权重（SPLICE）。
        color_clusters: 颜色聚类数（Yottixel）。
        spatial_ratio_per_cluster: 空间子聚类比例（Yottixel）。
        max_total_candidates: 马赛克后的最大候选数（Yottixel）。
        visualization_enabled: 是否生成图表。
        figures_dir: 图表输出的子目录。
        dpi: 图表 DPI。
        figure_formats: 图表的输出格式。
        log_level: 日志级别。
        log_file: 日志文件路径。
    """
    wsi_root: str = ""
    output_root: str = "outputs"
    patch_size: int = 1024
    patches_per_case: int = 6
    ds_mask: int = 32
    stride: int = 1024
    seed: int = 2025
    only_dx1: bool = True
    max_cases: int = 0  # 0 = 全部，N = 限制为前 N 个病例
    innovation_weight: float = 0.25
    min_center_distance_ratio: float = 0.65
    min_feature_distance: float = 0.10
    preselect_topk: int = 96
    min_tissue_ratio_preselect: float = 0.35
    candidate_pool_size: int = 72
    max_tries: int = 5000
    enable_gland_irregularity: bool = True
    enabled_samplers: List[str] = field(default_factory=list)

    # 算法专用参数
    n_bins_x: int = 3
    n_bins_y: int = 3
    redundancy_lambda: float = 2.0
    color_clusters: int = 8
    spatial_ratio_per_cluster: float = 0.25
    max_total_candidates: int = 400
    morphology_seed_quality_weight: float = 0.25

    # 可视化
    visualization_enabled: bool = True
    figures_dir: str = "figures"
    dpi: int = 300
    figure_formats: List[str] = field(default_factory=lambda: ["png"])

    # 日志
    log_level: str = "INFO"
    log_file: str = "logs/benchmark.log"


# ============================================================
# 图块级数据
# ============================================================

@dataclass
class Patch:
    """单个提取的图像图块及其空间元数据。

    Attributes:
        x0: Level-0 WSI 空间中的左上角 X 坐标。
        y0: Level-0 WSI 空间中的左上角 Y 坐标。
        cx: 中心 X 坐标。
        cy: 中心 Y 坐标。
        patch_np: RGB 图像数组，形状为 (patch_size, patch_size, 3)。
        slide_base: 切片基础名称（不带扩展名的文件名）。
        slide_path: 源切片的完整路径。
        patch_index: 切片内的顺序索引。
    """
    x0: int
    y0: int
    cx: int
    cy: int
    patch_np: np.ndarray
    slide_base: str
    slide_path: str
    patch_index: int = -1


@dataclass
class QualityMetrics:
    """单个图块的所有质量相关指标。

    这些指标在候选池构建时计算一次，
    并在所有采样器之间共享。
    """
    tissue_ratio: float = 0.0
    white_ratio: float = 0.0
    dark_ratio: float = 0.0
    blur_score: float = 0.0
    mean_sat: float = 0.0
    colorfulness: float = 0.0
    entropy: float = 0.0
    edge_density: float = 0.0
    stain_balance_penalty: float = 0.0
    nuclear_like_density: float = 0.0
    heterogeneity_crc: float = 0.0
    nuclear_edge_density: float = 0.0
    gland_irregularity: float = 0.0
    tumor_bias: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "tissue_ratio": self.tissue_ratio,
            "white_ratio": self.white_ratio,
            "dark_ratio": self.dark_ratio,
            "blur_score": self.blur_score,
            "mean_sat": self.mean_sat,
            "colorfulness": self.colorfulness,
            "entropy": self.entropy,
            "edge_density": self.edge_density,
            "stain_balance_penalty": self.stain_balance_penalty,
            "nuclear_like_density": self.nuclear_like_density,
            "heterogeneity_crc": self.heterogeneity_crc,
            "nuclear_edge_density": self.nuclear_edge_density,
            "gland_irregularity": self.gland_irregularity,
            "tumor_bias": self.tumor_bias,
        }


@dataclass
class ScorePack:
    """图块的组合评分与归一化评分。"""
    baseline_score: float = 0.0
    innovation_score: float = 0.0
    final_score: float = 0.0
    baseline_score_norm: float = 0.0
    innovation_score_norm: float = 0.0
    final_score_norm: float = 0.0
    tissue_norm: float = 0.0
    white_good_norm: float = 0.0
    dark_good_norm: float = 0.0
    blur_norm: float = 0.0
    sat_norm: float = 0.0
    color_norm: float = 0.0
    entropy_norm: float = 0.0
    edge_norm: float = 0.0
    tumor_bias_norm: float = 0.0
    gland_irregularity_norm: float = 0.0
    heterogeneity_norm: float = 0.0
    qc_quality_norm: float = 0.0
    tumor_morph_norm: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "baseline_score": self.baseline_score,
            "innovation_score": self.innovation_score,
            "final_score": self.final_score,
            "baseline_score_norm": self.baseline_score_norm,
            "innovation_score_norm": self.innovation_score_norm,
            "final_score_norm": self.final_score_norm,
            "tissue_norm": self.tissue_norm,
            "white_good_norm": self.white_good_norm,
            "dark_good_norm": self.dark_good_norm,
            "blur_norm": self.blur_norm,
            "sat_norm": self.sat_norm,
            "color_norm": self.color_norm,
            "entropy_norm": self.entropy_norm,
            "edge_norm": self.edge_norm,
            "tumor_bias_norm": self.tumor_bias_norm,
            "gland_irregularity_norm": self.gland_irregularity_norm,
            "heterogeneity_norm": self.heterogeneity_norm,
            "qc_quality_norm": self.qc_quality_norm,
            "tumor_morph_norm": self.tumor_morph_norm,
        }


@dataclass
class FeatureVector:
    """图块的特征向量，用于多样性度量和聚类。

    标准特征向量为 8 维：
    [tissue_ratio, white_ratio, blur_score/1000, mean_sat/255,
     colorfulness/100, entropy/8, edge_density, tumor_bias/10]
    """
    values: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.values)

    def to_numpy(self) -> np.ndarray:
        return self.values


# ============================================================
# 候选
# ============================================================

@dataclass
class CandidatePatch:
    """包含所有计算数据的候选图块。

    这是贯穿流水线的统一数据单元：
    WSI -> 组织掩码 -> 网格扫描 -> 候选池 -> 采样器。

    每个候选都携带其图像数据、质量指标、评分
    以及特征向量，供下游选择算法使用。
    """
    patch: Patch
    metrics: QualityMetrics = field(default_factory=QualityMetrics)
    scores: ScorePack = field(default_factory=ScorePack)
    feature: FeatureVector = field(default_factory=FeatureVector)
    morph_feature: Optional[FeatureVector] = None  # 扩展形态学特征
    color_hist: Optional[np.ndarray] = None  # RGB 直方图（Yottixel）
    qc_level: QCLevel = QCLevel.FALLBACK
    selection_score: float = 0.0  # 算法特定的选择评分
    redundancy_penalty: float = 0.0  # SPLICE 冗余惩罚
    cluster_id: int = -1  # 聚类分配（KMeans、Yottixel、SDM）
    spatial_cluster_id: int = -1  # 空间子聚类（Yottixel）

    @property
    def score_for_ranking(self) -> float:
        """用于排序的默认评分：若已设置 selection_score 则采用，否则使用 final_score。"""
        if self.selection_score != 0.0:
            return self.selection_score
        return self.scores.final_score


# ============================================================
# 切片级数据
# ============================================================

@dataclass
class SlideInfo:
    """单张 WSI 切片的元数据。"""
    path: str
    base_name: str
    case_id: str
    dimensions: Tuple[int, int] = (0, 0)
    is_dx1: bool = False


@dataclass
class SlideResult:
    """使用某一种算法处理单张切片的结果。"""
    case_id: str = ""
    slide_path: str = ""
    algorithm: AlgorithmName = AlgorithmName.RANDOM
    status: Status = Status.OK
    num_strict: int = 0
    num_medium_candidates: int = 0
    num_relaxed: int = 0
    num_fallback: int = 0
    num_selected: int = 0
    selection_level: str = ""
    avg_tumor_bias: float = 0.0
    avg_gland_irregularity: float = 0.0
    avg_baseline_score: float = 0.0
    avg_innovation_score: float = 0.0
    avg_final_score: float = 0.0
    avg_baseline_score_norm: float = 0.0
    avg_innovation_score_norm: float = 0.0
    avg_final_score_norm: float = 0.0
    avg_qc_quality_norm: float = 0.0
    avg_tumor_morph_norm: float = 0.0
    selected_patches: List[CandidatePatch] = field(default_factory=list)
    error: str = ""

    def to_summary_dict(self) -> Dict[str, Any]:
        """用于 CSV 输出的扁平化字典。"""
        return {
            "case_id": self.case_id,
            "slide_path": self.slide_path,
            "algorithm": self.algorithm.value,
            "status": self.status.value,
            "num_strict": self.num_strict,
            "num_medium_candidates": self.num_medium_candidates,
            "num_relaxed": self.num_relaxed,
            "num_fallback": self.num_fallback,
            "num_selected": self.num_selected,
            "selection_level": self.selection_level,
            "avg_tumor_bias": self.avg_tumor_bias,
            "avg_gland_irregularity": self.avg_gland_irregularity,
            "avg_baseline_score": self.avg_baseline_score,
            "avg_innovation_score": self.avg_innovation_score,
            "avg_final_score": self.avg_final_score,
            "avg_baseline_score_norm": self.avg_baseline_score_norm,
            "avg_innovation_score_norm": self.avg_innovation_score_norm,
            "avg_final_score_norm": self.avg_final_score_norm,
            "avg_qc_quality_norm": self.avg_qc_quality_norm,
            "avg_tumor_morph_norm": self.avg_tumor_morph_norm,
            "error": self.error,
        }


# ============================================================
# 病例级数据
# ============================================================

@dataclass
class CaseResult:
    """使用某一种算法处理单个病例的结果。"""
    case_id: str = ""
    algorithm: AlgorithmName = AlgorithmName.RANDOM
    num_slides: int = 0
    num_slides_processed: int = 0
    num_saved: int = 0
    status: Status = Status.OK
    saved_patch_names: List[str] = field(default_factory=list)
    slide_results: List[SlideResult] = field(default_factory=list)
    error: str = ""


@dataclass
class AlgorithmResult:
    """某一种算法在所有病例上的汇总结果。"""
    algorithm: AlgorithmName = AlgorithmName.RANDOM
    case_results: List[CaseResult] = field(default_factory=list)
    total_cases: int = 0
    total_ok: int = 0
    total_partial: int = 0
    total_skipped: int = 0
    total_failed: int = 0
    total_patches_saved: int = 0

    # 汇总指标
    metric_values: Dict[str, float] = field(default_factory=dict)


# ============================================================
# CSV 记录类型（供 pandas 使用的扁平化结构）
# ============================================================

@dataclass
class SlideRecord:
    """切片级 CSV 输出（slide_metrics.csv）的扁平化行记录。"""
    case_id: str = ""
    algorithm: str = ""
    slide_base: str = ""
    slide_path: str = ""
    status: str = ""
    num_selected: int = 0
    selection_level: str = ""
    avg_baseline_score: float = 0.0
    avg_innovation_score: float = 0.0
    avg_final_score: float = 0.0
    avg_qc_quality_norm: float = 0.0
    avg_tumor_morph_norm: float = 0.0
    avg_tumor_bias: float = 0.0
    error: str = ""


@dataclass
class PatchRecord:
    """图块级 CSV 输出（patch_metrics.csv）的扁平化行记录。"""
    case_id: str = ""
    algorithm: str = ""
    rank_in_case: int = 0
    saved_patch_file: str = ""
    slide_base: str = ""
    slide_path: str = ""
    # 质量指标
    tissue_ratio: float = 0.0
    white_ratio: float = 0.0
    dark_ratio: float = 0.0
    blur_score: float = 0.0
    mean_sat: float = 0.0
    colorfulness: float = 0.0
    entropy: float = 0.0
    edge_density: float = 0.0
    stain_balance_penalty: float = 0.0
    nuclear_like_density: float = 0.0
    heterogeneity_crc: float = 0.0
    nuclear_edge_density: float = 0.0
    gland_irregularity: float = 0.0
    tumor_bias: float = 0.0
    # 评分
    baseline_score: float = 0.0
    innovation_score: float = 0.0
    final_score: float = 0.0
    qc_quality_norm: float = 0.0
    tumor_morph_norm: float = 0.0
    # 坐标
    cx: int = 0
    cy: int = 0
    # 算法专用
    cluster_id: int = -1
    selection_score: float = 0.0
    redundancy_penalty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "algorithm": self.algorithm,
            "rank_in_case": self.rank_in_case,
            "saved_patch_file": self.saved_patch_file,
            "slide_base": self.slide_base,
            "slide_path": self.slide_path,
            "tissue_ratio": self.tissue_ratio,
            "white_ratio": self.white_ratio,
            "dark_ratio": self.dark_ratio,
            "blur_score": self.blur_score,
            "mean_sat": self.mean_sat,
            "colorfulness": self.colorfulness,
            "entropy": self.entropy,
            "edge_density": self.edge_density,
            "stain_balance_penalty": self.stain_balance_penalty,
            "nuclear_like_density": self.nuclear_like_density,
            "heterogeneity_crc": self.heterogeneity_crc,
            "nuclear_edge_density": self.nuclear_edge_density,
            "gland_irregularity": self.gland_irregularity,
            "tumor_bias": self.tumor_bias,
            "baseline_score": self.baseline_score,
            "innovation_score": self.innovation_score,
            "final_score": self.final_score,
            "qc_quality_norm": self.qc_quality_norm,
            "tumor_morph_norm": self.tumor_morph_norm,
            "cx": self.cx,
            "cy": self.cy,
            "cluster_id": self.cluster_id,
            "selection_score": self.selection_score,
            "redundancy_penalty": self.redundancy_penalty,
        }
