# -*- coding: utf-8 -*-
"""基准测试结果的 CSV 输出写入器。

生成具有统一字段格式的标准 CSV 文件：
- patch_metrics.csv：每个贴片的质量和分数详情
- slide_metrics.csv：每个算法每个切片的摘要
- method_summary.csv：每个算法的聚合指标
- paper_tables.csv：可直接用于发表的对比表
"""

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from common.dataclasses import (
    AlgorithmResult,
    CaseResult,
    SlideResult,
    CandidatePatch,
    PatchRecord,
)
from utils.file_io import ensure_dir

logger = logging.getLogger(__name__)


def write_patch_metrics_csv(
    algo_results: Dict[str, AlgorithmResult],
    output_dir: Path,
) -> None:
    """写入贴片级别的指标 CSV。

    每个保存的贴片占一行，包含所有质量指标和分数。

    Args:
        algo_results: 算法结果。
        output_dir: CSV 文件的输出目录。
    """
    ensure_dir(output_dir)
    records = []

    for algo_name, algo_result in algo_results.items():
        for case_result in algo_result.case_results:
            patch_rank = 0
            for slide_result in case_result.slide_results:
                for patch in slide_result.selected_patches:
                    patch_rank += 1
                    record = _candidate_to_patch_record(
                        patch, algo_name, patch_rank
                    )
                    records.append(record.to_dict())

    if records:
        df = pd.DataFrame(records)
        path = output_dir / "patch_metrics.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"Wrote {len(records)} rows to {path}")
    else:
        logger.warning("No patch records to write")


def write_slide_metrics_csv(
    algo_results: Dict[str, AlgorithmResult],
    output_dir: Path,
) -> None:
    """写入切片级别的指标 CSV。

    每个算法每个切片占一行，包含摘要统计数据。

    Args:
        algo_results: 算法结果。
        output_dir: 输出目录。
    """
    ensure_dir(output_dir)
    records = []

    for algo_name, algo_result in algo_results.items():
        for case_result in algo_result.case_results:
            for slide_result in case_result.slide_results:
                records.append(slide_result.to_summary_dict())

    if records:
        df = pd.DataFrame(records)
        path = output_dir / "slide_metrics.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"Wrote {len(records)} rows to {path}")
    else:
        logger.warning("No slide records to write")


def write_method_summary_csv(
    algo_results: Dict[str, AlgorithmResult],
    metric_summary: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path,
) -> None:
    """写入方法级别的摘要 CSV。

    每个算法占一行，包含聚合指标。

    Args:
        algo_results: 算法结果。
        metric_summary: 来自 BatchEvaluator.evaluate()。
        output_dir: 输出目录。
    """
    ensure_dir(output_dir)
    records = []

    for algo_name, algo_result in algo_results.items():
        row = {
            "algorithm": algo_name,
            "total_cases": algo_result.total_cases,
            "total_ok": algo_result.total_ok,
            "total_partial": algo_result.total_partial,
            "total_failed": algo_result.total_failed,
            "total_patches_saved": algo_result.total_patches_saved,
        }

        # 添加指标均值
        if algo_name in metric_summary:
            for metric_name, stats in metric_summary[algo_name].items():
                if metric_name.startswith("_"):
                    continue
                row[f"{metric_name}_mean"] = stats.get("mean", 0.0)
                row[f"{metric_name}_std"] = stats.get("std", 0.0)

        records.append(row)

    if records:
        df = pd.DataFrame(records)
        path = output_dir / "method_summary.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"Wrote {len(records)} rows to {path}")


def write_paper_tables_csv(
    algo_results: Dict[str, AlgorithmResult],
    metric_summary: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path,
) -> None:
    """写入可直接用于发表的对比表。

    与 method_summary 相同，但使用四舍五入的值和格式化的表头。

    Args:
        algo_results: 算法结果。
        metric_summary: 来自 BatchEvaluator.evaluate()。
        output_dir: 输出目录。
    """
    ensure_dir(output_dir)
    records = []

    # 算法的显示名称
    algo_display_names = {
        "random": "Random",
        "grid": "Grid (Quality Top-K)",
        "largest_tissue": "Largest Tissue Component",
        "stratified": "Stratified Spatial",
        "kmeans": "K-Means",
        "yottixel": "Yottixel-inspired",
        "splice": "SPLICE-inspired",
        "sdm": "SDM",
        "sentinel": "Sentinel (SAPS)",
    }

    for algo_name, algo_result in algo_results.items():
        row = {
            "Algorithm": algo_display_names.get(algo_name, algo_name),
            "Cases (Total)": algo_result.total_cases,
            "Cases (OK)": algo_result.total_ok,
            "Patches Saved": algo_result.total_patches_saved,
        }

        if algo_name in metric_summary:
            stats = metric_summary[algo_name]
            row["Spatial Coverage (mean)"] = round(
                stats.get("spatial_coverage", {}).get("mean", 0.0), 4
            )
            row["Feature Diversity (mean)"] = round(
                stats.get("feature_diversity", {}).get("mean", 0.0), 4
            )
            row["Redundancy Rate"] = round(
                stats.get("redundancy_rate", {}).get("mean", 0.0), 4
            )
            row["Covered Regions (mean)"] = round(
                stats.get("covered_region_count", {}).get("mean", 0.0), 1
            )

        records.append(row)

    if records:
        df = pd.DataFrame(records)
        path = output_dir / "paper_tables.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"Wrote paper table to {path}")


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------

def _candidate_to_patch_record(
    candidate: CandidatePatch,
    algorithm: str,
    rank: int,
) -> PatchRecord:
    """将 CandidatePatch 转换为用于 CSV 输出的 PatchRecord。

    Args:
        candidate: 选定的候选贴片。
        algorithm: 算法名称字符串。
        rank: 用例中的排名（从 1 开始）。

    Returns:
        扁平化的 PatchRecord。
    """
    p = candidate.patch
    m = candidate.metrics
    s = candidate.scores

    return PatchRecord(
        case_id="",  # 将在上游设置
        algorithm=algorithm,
        rank_in_case=rank,
        saved_patch_file=f"{p.slide_base}_{p.patch_index}.png",
        slide_base=p.slide_base,
        slide_path=p.slide_path,
        # 质量
        tissue_ratio=m.tissue_ratio,
        white_ratio=m.white_ratio,
        dark_ratio=m.dark_ratio,
        blur_score=m.blur_score,
        mean_sat=m.mean_sat,
        colorfulness=m.colorfulness,
        entropy=m.entropy,
        edge_density=m.edge_density,
        stain_balance_penalty=m.stain_balance_penalty,
        nuclear_like_density=m.nuclear_like_density,
        heterogeneity_crc=m.heterogeneity_crc,
        nuclear_edge_density=m.nuclear_edge_density,
        gland_irregularity=m.gland_irregularity,
        tumor_bias=m.tumor_bias,
        # 分数
        baseline_score=s.baseline_score,
        innovation_score=s.innovation_score,
        final_score=s.final_score,
        qc_quality_norm=s.qc_quality_norm,
        tumor_morph_norm=s.tumor_morph_norm,
        # 坐标
        cx=p.cx,
        cy=p.cy,
        # 算法特定
        cluster_id=candidate.cluster_id,
        selection_score=candidate.selection_score,
        redundancy_penalty=candidate.redundancy_penalty,
    )
