#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WSI 补丁选择基准测试 - 主入口点

用于在 WSI 上比较补丁选择算法的统一流水线。

用法:
    python run_patch_selection.py --config configs/default_config.yaml
    python run_patch_selection.py --wsi-root /path/to/wsi --output-root ./outputs

流水线:
    加载配置 -> 加载 WSI -> 生成候选池（所有算法共享） ->
    执行所有采样器 -> 指标计算 -> 可视化 -> CSV -> 报告
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

# 将项目根目录添加到路径
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config_loader import load_config
from common.dataclasses import (
    ConfigBundle,
    CaseResult,
    SlideResult,
    AlgorithmResult,
    CandidatePatch,
    Patch,
    QualityMetrics,
    ScorePack,
    FeatureVector,
    SlideInfo,
)
from common.enums import AlgorithmName, Status, QCLevel
from datasets.wsi_reader import WSIReader
from datasets.wsi_scanner import discover_wsis
from core.tissue_mask import build_hybrid_tissue_mask
from core.candidate_pool import CandidatePoolBuilder, merge_candidates_by_priority
from core.diversity import select_diverse_topk
from core.patch_io import commit_case_patches_atomic
from samplers import get_sampler, list_samplers
from samplers.base_sampler import BaseSampler
from metrics.spatial_coverage import SpatialCoverageMetric
from metrics.feature_diversity import FeatureDiversityMetric
from metrics.redundancy_rate import RedundancyRateMetric
from metrics.covered_region_count import CoveredRegionCountMetric
from evaluation.batch_stats import BatchEvaluator
from evaluation.csv_writer import (
    write_patch_metrics_csv,
    write_slide_metrics_csv,
    write_method_summary_csv,
    write_paper_tables_csv,
)
from evaluation.report_writer import generate_report
from utils.logger import setup_logging, get_logger
from utils.seed import set_global_seed
from utils.file_io import save_progress_json, load_progress_json

logger = get_logger(__name__)


# ============================================================
# 进度序列化（断点 / 恢复）
# ============================================================

# 重构 CandidatePatch 中空补丁图像数组的哨兵值
_PLACEHOLDER_IMG = None  # allocated once on first use


def _get_placeholder_img():
    """返回一个用于重构 CandidatePatch 的微型占位图像。"""
    global _PLACEHOLDER_IMG
    if _PLACEHOLDER_IMG is None:
        _PLACEHOLDER_IMG = np.zeros((1, 1, 3), dtype=np.uint8)
    return _PLACEHOLDER_IMG


def _case_result_to_dict(case_result: CaseResult) -> dict:
    """将 CaseResult 序列化为 JSON 可序列化的字典。

    仅保存标量数据。patch_np 图像被排除，
    因为它们已通过 commit_case_patches_atomic 保存为 PNG 文件。
    """
    slides = []
    for sr in case_result.slide_results:
        patches = []
        for cp in sr.selected_patches:
            patches.append({
                "cx": cp.patch.cx,
                "cy": cp.patch.cy,
                "x0": cp.patch.x0,
                "y0": cp.patch.y0,
                "slide_base": cp.patch.slide_base,
                "slide_path": cp.patch.slide_path,
                "patch_index": cp.patch.patch_index,
                # 特征向量
                "feature": cp.feature.values.tolist(),
                # 质量指标
                "tissue_ratio": cp.metrics.tissue_ratio,
                "white_ratio": cp.metrics.white_ratio,
                "dark_ratio": cp.metrics.dark_ratio,
                "blur_score": cp.metrics.blur_score,
                "mean_sat": cp.metrics.mean_sat,
                "colorfulness": cp.metrics.colorfulness,
                "entropy": cp.metrics.entropy,
                "edge_density": cp.metrics.edge_density,
                "stain_balance_penalty": cp.metrics.stain_balance_penalty,
                "nuclear_like_density": cp.metrics.nuclear_like_density,
                "heterogeneity_crc": cp.metrics.heterogeneity_crc,
                "nuclear_edge_density": cp.metrics.nuclear_edge_density,
                "gland_irregularity": cp.metrics.gland_irregularity,
                "tumor_bias": cp.metrics.tumor_bias,
                # 分数
                "baseline_score": cp.scores.baseline_score,
                "innovation_score": cp.scores.innovation_score,
                "final_score": cp.scores.final_score,
                "qc_quality_norm": cp.scores.qc_quality_norm,
                "tumor_morph_norm": cp.scores.tumor_morph_norm,
                "qc_level": cp.qc_level.value,
                "selection_score": cp.selection_score,
                "redundancy_penalty": cp.redundancy_penalty,
                "cluster_id": cp.cluster_id,
            })
        slides.append({
            "slide_path": sr.slide_path,
            "status": sr.status.value,
            "num_selected": sr.num_selected,
            "num_strict": sr.num_strict,
            "num_medium_candidates": sr.num_medium_candidates,
            "num_relaxed": sr.num_relaxed,
            "avg_baseline_score": sr.avg_baseline_score,
            "avg_final_score": sr.avg_final_score,
            "avg_tumor_bias": sr.avg_tumor_bias,
            "avg_qc_quality_norm": sr.avg_qc_quality_norm,
            "avg_tumor_morph_norm": sr.avg_tumor_morph_norm,
            "error": sr.error,
            "selected_patches": patches,
        })
    return {
        "case_id": case_result.case_id,
        "algorithm": case_result.algorithm.value,
        "num_slides": case_result.num_slides,
        "num_slides_processed": case_result.num_slides_processed,
        "num_saved": case_result.num_saved,
        "status": case_result.status.value,
        "saved_patch_names": case_result.saved_patch_names,
        "error": case_result.error,
        "slide_results": slides,
    }


def _dict_to_case_result(d: dict) -> CaseResult:
    """将字典反序列化回 CaseResult。

    CandidatePatch 对象在重建时不包含 patch_np（使用占位符）。
    这对于指标评估和 CSV 生成已足够。
    """
    slide_results = []
    for sr_dict in d.get("slide_results", []):
        selected_patches = []
        for p_dict in sr_dict.get("selected_patches", []):
            patch = Patch(
                x0=p_dict["x0"],
                y0=p_dict["y0"],
                cx=p_dict["cx"],
                cy=p_dict["cy"],
                patch_np=_get_placeholder_img(),
                slide_base=p_dict.get("slide_base", ""),
                slide_path=p_dict.get("slide_path", ""),
                patch_index=p_dict.get("patch_index", -1),
            )
            metrics = QualityMetrics(
                tissue_ratio=p_dict.get("tissue_ratio", 0.0),
                white_ratio=p_dict.get("white_ratio", 0.0),
                dark_ratio=p_dict.get("dark_ratio", 0.0),
                blur_score=p_dict.get("blur_score", 0.0),
                mean_sat=p_dict.get("mean_sat", 0.0),
                colorfulness=p_dict.get("colorfulness", 0.0),
                entropy=p_dict.get("entropy", 0.0),
                edge_density=p_dict.get("edge_density", 0.0),
                stain_balance_penalty=p_dict.get("stain_balance_penalty", 0.0),
                nuclear_like_density=p_dict.get("nuclear_like_density", 0.0),
                heterogeneity_crc=p_dict.get("heterogeneity_crc", 0.0),
                nuclear_edge_density=p_dict.get("nuclear_edge_density", 0.0),
                gland_irregularity=p_dict.get("gland_irregularity", 0.0),
                tumor_bias=p_dict.get("tumor_bias", 0.0),
            )
            scores = ScorePack(
                baseline_score=p_dict.get("baseline_score", 0.0),
                innovation_score=p_dict.get("innovation_score", 0.0),
                final_score=p_dict.get("final_score", 0.0),
                qc_quality_norm=p_dict.get("qc_quality_norm", 0.0),
                tumor_morph_norm=p_dict.get("tumor_morph_norm", 0.0),
            )
            feature_vals = np.array(p_dict.get("feature", [0.0]*8), dtype=np.float32)
            feature = FeatureVector(values=feature_vals)

            cp = CandidatePatch(
                patch=patch,
                metrics=metrics,
                scores=scores,
                feature=feature,
                qc_level=QCLevel(p_dict.get("qc_level", "fallback")),
                selection_score=p_dict.get("selection_score", 0.0),
                redundancy_penalty=p_dict.get("redundancy_penalty", 0.0),
                cluster_id=p_dict.get("cluster_id", -1),
            )
            selected_patches.append(cp)

        sr = SlideResult(
            case_id=d.get("case_id", ""),
            slide_path=sr_dict.get("slide_path", ""),
            algorithm=AlgorithmName(d.get("algorithm", "random")),
            status=Status(sr_dict.get("status", "OK")),
            num_selected=sr_dict.get("num_selected", 0),
            num_strict=sr_dict.get("num_strict", 0),
            num_medium_candidates=sr_dict.get("num_medium_candidates", 0),
            num_relaxed=sr_dict.get("num_relaxed", 0),
            avg_baseline_score=sr_dict.get("avg_baseline_score", 0.0),
            avg_final_score=sr_dict.get("avg_final_score", 0.0),
            avg_tumor_bias=sr_dict.get("avg_tumor_bias", 0.0),
            avg_qc_quality_norm=sr_dict.get("avg_qc_quality_norm", 0.0),
            avg_tumor_morph_norm=sr_dict.get("avg_tumor_morph_norm", 0.0),
            error=sr_dict.get("error", ""),
            selected_patches=selected_patches,
        )
        slide_results.append(sr)

    return CaseResult(
        case_id=d["case_id"],
        algorithm=AlgorithmName(d["algorithm"]),
        num_slides=d.get("num_slides", 0),
        num_slides_processed=d.get("num_slides_processed", 0),
        num_saved=d.get("num_saved", 0),
        status=Status(d.get("status", "OK")),
        saved_patch_names=d.get("saved_patch_names", []),
        error=d.get("error", ""),
        slide_results=slide_results,
    )


def _make_progress_key(case_id: str, algo_name: str) -> str:
    """用于进度跟踪的组合键。"""
    return f"{case_id}||{algo_name}"


def main():
    """运行完整的 WSI 补丁选择基准测试流水线。"""
    # ---- 1. 加载配置 ----
    config = load_config()
    setup_logging(
        level=config.log_level,
        log_file=config.log_file,
    )
    set_global_seed(config.seed)

    logger.info("=" * 60)
    logger.info("WSI Patch Selection Benchmark")
    logger.info(f"Patch size: {config.patch_size}, K: {config.patches_per_case}")
    logger.info(f"Enabled samplers: {config.enabled_samplers}")
    logger.info("=" * 60)

    # ---- 2. 发现 WSI ----
    logger.info("Discovering WSI files...")
    case_to_slides = discover_wsis(config.wsi_root, config.only_dx1)
    if not case_to_slides:
        logger.error("No WSI files found. Please check --wsi-root.")
        sys.exit(1)
    logger.info(
        f"Found {len(case_to_slides)} cases, "
        f"{sum(len(v) for v in case_to_slides.values())} slides"
    )

    # ---- 3. 初始化采样器 ----
    samplers: List[BaseSampler] = []
    for sampler_name in config.enabled_samplers:
        if sampler_name == "all":
            # 启用所有已注册的采样器
            for name in list_samplers():
                samplers.append(get_sampler(name, config))
            break
        else:
            try:
                samplers.append(get_sampler(sampler_name, config))
            except ValueError as e:
                logger.warning(f"Skipping unknown sampler: {e}")

    logger.info(f"Initialized {len(samplers)} samplers: "
                f"{[s.algorithm_name() for s in samplers]}")

    # ---- 4. 初始化指标和评估器 ----
    metrics_list = [
        SpatialCoverageMetric(),
        FeatureDiversityMetric(),
        RedundancyRateMetric(),
        CoveredRegionCountMetric(),
    ]
    evaluator = BatchEvaluator()

    # ---- 5. 处理所有病例（候选池共享，含断点/恢复） ----
    all_results: Dict[str, AlgorithmResult] = {}
    output_root = Path(config.output_root)
    progress_path = output_root / "progress.json"

    for sampler in samplers:
        algo_name = sampler.algorithm_name()
        algo_result = AlgorithmResult(algorithm=AlgorithmName(algo_name))
        all_results[algo_name] = algo_result

    # 加载之前的进度（中断后恢复）
    prev_progress = load_progress_json(progress_path)
    completed_keys: set = set()
    if prev_progress:
        for key, case_dict in prev_progress.items():
            algo_name = case_dict.get("algorithm", "")
            case_result = _dict_to_case_result(case_dict)
            if algo_name in all_results:
                all_results[algo_name].case_results.append(case_result)
            completed_keys.add(key)
        total_done = len(completed_keys)
        logger.info(
            f"Resume: loaded {total_done} completed (case, algorithm) pairs "
            f"from {progress_path}"
        )

    # 构建待处理病例列表（跳过全部算法均已完成的病例）
    case_list = list(case_to_slides.items())
    if config.max_cases > 0:
        case_list = case_list[:config.max_cases]
        logger.info(f"Limited to first {config.max_cases} case(s)")

    pending_cases: List[tuple] = []
    for case_id, slide_paths in case_list:
        all_done = True
        for sampler in samplers:
            key = _make_progress_key(case_id, sampler.algorithm_name())
            if key not in completed_keys:
                all_done = False
                break
        if not all_done:
            pending_cases.append((case_id, slide_paths))

    if len(pending_cases) < len(case_list):
        already_done_cases = len(case_list) - len(pending_cases)
        logger.info(
            f"Resume: {already_done_cases} cases fully completed, "
            f"{len(pending_cases)} cases remaining"
        )

    # 处理剩余病例（每个病例的所有未完成算法共享候选池）
    progress_data = prev_progress.copy() if prev_progress else {}
    for case_id, slide_paths in tqdm(pending_cases, desc="Processing cases"):
        # 找出该病例尚未完成的算法
        pending_samplers = []
        for sampler in samplers:
            key = _make_progress_key(case_id, sampler.algorithm_name())
            if key not in completed_keys:
                pending_samplers.append(sampler)

        if not pending_samplers:
            continue

        # 一次处理：候选池在所有未完成算法间共享
        algo_results = _process_case_shared_pool(
            case_id=case_id,
            slide_paths=slide_paths,
            samplers=pending_samplers,
            config=config,
            output_root=output_root,
        )

        # 汇总结果并保存进度
        for algo_name, case_result in algo_results.items():
            all_results[algo_name].case_results.append(case_result)
            key = _make_progress_key(case_id, algo_name)
            progress_data[key] = _case_result_to_dict(case_result)

        save_progress_json(progress_path, progress_data)

    # ---- 6. 聚合统计 ----
    for algo_name, algo_result in all_results.items():
        algo_result.total_cases = len(algo_result.case_results)
        algo_result.total_ok = sum(
            1 for c in algo_result.case_results if c.status == Status.OK
        )
        algo_result.total_partial = sum(
            1 for c in algo_result.case_results if c.status == Status.PARTIAL
        )
        algo_result.total_failed = sum(
            1 for c in algo_result.case_results
            if c.status in (Status.PROCESSING_FAILED, Status.OPEN_FAILED)
        )
        algo_result.total_skipped = sum(
            1 for c in algo_result.case_results if c.status == Status.SKIPPED
        )
        algo_result.total_patches_saved = sum(
            c.num_saved for c in algo_result.case_results
        )
        logger.info(
            f"[{algo_name}] {algo_result.total_cases} cases, "
            f"{algo_result.total_ok} OK, "
            f"{algo_result.total_patches_saved} patches saved"
        )

    # ---- 7. 计算指标 ----
    metric_summary = evaluator.evaluate(all_results, metrics_list)

    # ---- 8. 生成 CSV ----
    csv_dir = output_root / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    write_patch_metrics_csv(all_results, csv_dir)
    write_slide_metrics_csv(all_results, csv_dir)
    write_method_summary_csv(all_results, metric_summary, csv_dir)
    write_paper_tables_csv(all_results, metric_summary, csv_dir)

    # ---- 9. 生成报告 ----
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    generate_report(all_results, metric_summary, config.__dict__, reports_dir)

    # ---- 10. 可视化（如果可用） ----
    if config.visualization_enabled:
        _generate_visualizations(all_results, metric_summary, config)

    logger.info("=" * 60)
    logger.info("Benchmark complete!")
    logger.info(f"Results saved to: {output_root}")
    logger.info("=" * 60)


# ============================================================
# 病例处理（候选池共享版本）
# ============================================================

def _process_case_shared_pool(
    case_id: str,
    slide_paths: List[Path],
    samplers: List[BaseSampler],
    config: ConfigBundle,
    output_root: Path,
) -> Dict[str, CaseResult]:
    """处理一个病例的所有采样器，候选池在算法间共享。

    每张切片只打开一次、只构建一次组织掩码、
    只构建一次网格候选池（供 8 个算法共享）
    和一次随机候选池（供 sentinel 使用）。

    Args:
        case_id: TCGA 病例 ID。
        slide_paths: 该病例的切片路径列表。
        samplers: 需要运行的采样器列表。
        config: 配置包。
        output_root: 输出根目录。

    Returns:
        算法名 -> CaseResult 的映射字典。
    """
    from utils.slide_reader import slide_base_from_path

    # 分离 sentinel 和其他算法（它们使用不同的候选池构建策略）
    sentinel_name = "sentinel"
    grid_samplers = [s for s in samplers if s.algorithm_name() != sentinel_name]
    sentinel_sampler = next((s for s in samplers if s.algorithm_name() == sentinel_name), None)

    # 初始化每个算法的结果
    algo_results: Dict[str, CaseResult] = {}
    algo_selected: Dict[str, List[CandidatePatch]] = {}
    algo_needed: Dict[str, int] = {}

    for sampler in samplers:
        algo_name = sampler.algorithm_name()
        algo_results[algo_name] = CaseResult(
            case_id=case_id,
            algorithm=AlgorithmName(algo_name),
            num_slides=len(slide_paths),
        )
        algo_selected[algo_name] = []
        algo_needed[algo_name] = config.patches_per_case

    # 遍历切片，逐步收集补丁直到所有算法都达到 K 个
    for slide_path in slide_paths:
        # 检查是否所有算法都已满足需求
        if all(n <= 0 for n in algo_needed.values()):
            break

        slide_base = slide_base_from_path(slide_path)

        # ---- 打开切片（仅一次） ----
        try:
            slide = WSIReader.open(str(slide_path))
        except Exception as e:
            logger.error(f"Failed to open {slide_path}: {e}")
            for sampler in samplers:
                algo_name = sampler.algorithm_name()
                if algo_needed[algo_name] <= 0:
                    continue
                sr = SlideResult(
                    case_id=case_id,
                    slide_path=str(slide_path),
                    algorithm=AlgorithmName(algo_name),
                    status=Status.OPEN_FAILED,
                    error=str(e),
                )
                algo_results[algo_name].slide_results.append(sr)
                algo_results[algo_name].num_slides_processed += 1
            continue

        try:
            # ---- 构建组织掩码（仅一次） ----
            mask, ds = build_hybrid_tissue_mask(slide, downsample=config.ds_mask)

            if mask.sum() == 0:
                for sampler in samplers:
                    algo_name = sampler.algorithm_name()
                    if algo_needed[algo_name] <= 0:
                        continue
                    sr = SlideResult(
                        case_id=case_id,
                        slide_path=str(slide_path),
                        algorithm=AlgorithmName(algo_name),
                        status=Status.NO_TISSUE,
                    )
                    algo_results[algo_name].slide_results.append(sr)
                    algo_results[algo_name].num_slides_processed += 1
                continue

            # ---- 构建网格候选池（仅一次，供所有非 sentinel 算法共享） ----
            grid_pool = None
            if grid_samplers:
                builder = CandidatePoolBuilder(config)
                grid_pool = builder.build(slide, mask, ds, slide_base)
                if len(grid_pool) > 0:
                    for c in grid_pool.candidates:
                        c.patch.slide_path = str(slide_path)

            # ---- 构建随机候选池（仅一次，供 sentinel 使用） ----
            random_pool = None
            if sentinel_sampler is not None:
                builder = CandidatePoolBuilder(config)
                random_pool = builder.build_random(slide, mask, ds, slide_base)
                if len(random_pool) > 0:
                    for c in random_pool.candidates:
                        c.patch.slide_path = str(slide_path)

            # ---- 运行各采样器 ----
            for sampler in samplers:
                algo_name = sampler.algorithm_name()
                if algo_needed[algo_name] <= 0:
                    continue

                # 选择对应的候选池
                pool = random_pool if algo_name == sentinel_name else grid_pool

                if pool is None or len(pool) == 0:
                    sr = SlideResult(
                        case_id=case_id,
                        slide_path=str(slide_path),
                        algorithm=AlgorithmName(algo_name),
                        status=Status.NO_CANDIDATE,
                        error="No valid candidates",
                    )
                    algo_results[algo_name].slide_results.append(sr)
                    algo_results[algo_name].num_slides_processed += 1
                    continue

                # 运行采样器选择补丁
                selected = sampler.select_patches(pool, algo_needed[algo_name])

                sr = SlideResult(
                    case_id=case_id,
                    slide_path=str(slide_path),
                    algorithm=AlgorithmName(algo_name),
                )

                if selected:
                    sr.status = Status.OK
                    sr.num_selected = len(selected)
                    sr.selected_patches = selected
                    algo_selected[algo_name].extend(selected)
                    algo_needed[algo_name] = config.patches_per_case - len(algo_selected[algo_name])

                    # 计算平均值
                    sr.avg_baseline_score = float(
                        np.mean([c.scores.baseline_score for c in selected])
                    )
                    sr.avg_final_score = float(
                        np.mean([c.scores.final_score for c in selected])
                    )
                    sr.avg_tumor_bias = float(
                        np.mean([c.metrics.tumor_bias for c in selected])
                    )
                    sr.avg_qc_quality_norm = float(
                        np.mean([c.scores.qc_quality_norm for c in selected])
                    )
                    sr.avg_tumor_morph_norm = float(
                        np.mean([c.scores.tumor_morph_norm for c in selected])
                    )

                    # QC 计数
                    sr.num_strict = sum(
                        1 for c in pool.candidates if c.qc_level == QCLevel.STRICT
                    )
                    sr.num_medium_candidates = sum(
                        1 for c in pool.candidates if c.qc_level == QCLevel.MEDIUM
                    )
                    sr.num_relaxed = sum(
                        1 for c in pool.candidates if c.qc_level == QCLevel.RELAXED
                    )
                else:
                    sr.status = Status.NO_PATCH_SAVED

                algo_results[algo_name].slide_results.append(sr)
                algo_results[algo_name].num_slides_processed += 1

        except Exception as e:
            logger.error(f"Error processing {slide_base}: {e}", exc_info=True)
            for sampler in samplers:
                algo_name = sampler.algorithm_name()
                if algo_needed[algo_name] <= 0:
                    continue
                sr = SlideResult(
                    case_id=case_id,
                    slide_path=str(slide_path),
                    algorithm=AlgorithmName(algo_name),
                    status=Status.PROCESSING_FAILED,
                    error=str(e),
                )
                algo_results[algo_name].slide_results.append(sr)
                algo_results[algo_name].num_slides_processed += 1
        finally:
            WSIReader.close(slide)

    # ---- 最终化：排序、取 top K、保存补丁 ----
    for sampler in samplers:
        algo_name = sampler.algorithm_name()
        result = algo_results[algo_name]
        selected = algo_selected[algo_name]

        if len(selected) >= config.patches_per_case:
            selected = sorted(
                selected,
                key=lambda p: p.scores.final_score,
                reverse=True,
            )[:config.patches_per_case]
            result.status = Status.OK
        elif len(selected) > 0:
            result.status = Status.PARTIAL
        else:
            result.status = Status.NO_PATCH_SAVED

        if selected:
            output_dir = output_root / "patches" / algo_name
            saved = commit_case_patches_atomic(output_dir, selected)
            result.num_saved = len(saved)
            result.saved_patch_names = [name for name, _ in saved]
        else:
            result.num_saved = 0

    return algo_results


# ============================================================
# 可视化
# ============================================================

def _generate_visualizations(
    all_results: Dict[str, AlgorithmResult],
    metric_summary: dict,
    config: ConfigBundle,
) -> None:
    """生成所有可视化输出。

    Args:
        all_results: 算法结果。
        metric_summary: 指标摘要。
        config: 配置包。
    """
    try:
        from visualization.charts import ChartVisualizer
        from visualization.paper_figures import PaperFigureVisualizer

        figures_dir = Path(config.output_root) / config.figures_dir
        figures_dir.mkdir(parents=True, exist_ok=True)

        chart_viz = ChartVisualizer(config)

        # 每个指标的柱状图
        for metric_name in [
            "spatial_coverage", "feature_diversity",
            "redundancy_rate", "covered_region_count",
        ]:
            chart_viz.draw_bar(
                metric_summary, metric_name,
                figures_dir / f"bar_{metric_name}",
            )

        # 雷达图
        chart_viz.draw_radar(metric_summary, figures_dir / "radar_all")

        # 热力图
        chart_viz.draw_heatmap(metric_summary, figures_dir / "heatmap")

        # 论文图表
        paper_viz = PaperFigureVisualizer(config)
        paper_viz.draw_all(metric_summary, figures_dir)

        logger.info(f"Visualizations saved to {figures_dir}")

    except ImportError as e:
        logger.warning(f"Visualization not available: {e}")
    except Exception as e:
        logger.error(f"Visualization error: {e}", exc_info=True)


# ============================================================
# 入口点
# ============================================================

if __name__ == "__main__":
    main()
