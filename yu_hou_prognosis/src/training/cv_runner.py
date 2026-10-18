# -*- coding: utf-8 -*-
"""
cv_runner.py
============================================================================
交叉验证主控制器 - Histomolecular基因表达预测。

功能:
    1. 加载/构建按病人ID划分的数据 (同一病人6张切片不跨train/val/test)
    2. 匹配TCGA ID, 从CSV读取目标基因表达, 自动做中位数高/低表达标签
    3. 对每个fold训练+评估 (patient-level)
    4. 汇总ACC/AUC/F1 (分类) 或 MSE/Pearson (回归)
    5. 附属KM生存分析 (预测分子表型分组)
    6. 保存每基因实验结果CSV + 模型权重 + 图

典型用法:
    python -m src.training.cv_runner --config config/default_config.yaml
============================================================================
"""

import os
import sys
import time
import pickle
import logging
import argparse
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import psutil
import pandas as pd

from src.utils.seed import set_seed
from src.utils.logger import setup_logger, log_section
from src.training.trainer import train, test
from src.data_loading.upstream_reader import UpstreamPatchReader
from src.data_loading.gene_labels import resolve_gene_column, compute_group_threshold, build_binary_label


# ============================================================
# 系统信息采集
# ============================================================

def _get_system_info() -> Dict[str, Any]:
    info = {
        "timestamp": datetime.now().isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
    }
    mem = psutil.virtual_memory()
    info["ram_total_gb"] = round(mem.total / (1024 ** 3), 1)
    info["ram_available_gb"] = round(mem.available / (1024 ** 3), 1)
    info["gpu_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda
        gpu_details = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_details.append({"index": i, "name": props.name,
                                "total_memory_gb": round(props.total_memory / (1024 ** 3), 1)})
        info["gpu_details"] = gpu_details
    else:
        info["gpu_count"] = 0
        info["gpu_details"] = []
    return info


def _log_system_info(logger, sys_info):
    log_section(logger, "系统信息", width=70)
    logger.info(f"  平台: {sys_info['platform']}")
    logger.info(f"  Python: {sys_info['python_version'].split()[0]}, PyTorch: {sys_info['pytorch_version']}")
    logger.info(f"  CPU: {sys_info['cpu_count_physical']}物理/{sys_info['cpu_count_logical']}逻辑, "
                f"内存: {sys_info['ram_total_gb']}GB")
    if sys_info["gpu_available"]:
        for gpu in sys_info["gpu_details"]:
            logger.info(f"  GPU[{gpu['index']}]: {gpu['name']} ({gpu['total_memory_gb']}GB)")
    else:
        logger.warning("  GPU不可用, 使用CPU训练")
    logger.info("=" * 70)


# ============================================================
# 完整数据划分构建 (Histomolecular)
# ============================================================

def _auto_build_full_splits(id_splits: Dict, config, logger, cache_path: Optional[Path] = None) -> Dict[str, Any]:
    """当pkl仅含病人ID时, 匹配TCGA ID + 目标基因表达 + 协变量 + patch, 构建完整数据划分。"""
    logger.info("检测到pkl仅含病人ID, 自动构建完整数据划分...")

    csv_path = config.resolve_path(config.data.genomic_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"表格CSV不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  表格CSV: {len(df)} 个样本")

    patient_id_col = config.data.genomic.patient_id_col
    survtime_col = config.data.genomic.survtime_col
    censor_col = config.data.genomic.censor_col
    task_type = str(config.model.task_type).lower()
    target_gene = config.model.target_gene
    suffix = config.data.genomic.rnaseq_suffix

    # ---- 目标基因列 ----
    gene_col = resolve_gene_column(df, target_gene, suffix)
    if gene_col is None:
        raise ValueError(
            f"未在CSV中找到目标基因 '{target_gene}' 的表达列。"
            f"请确认列名为 '{target_gene}{suffix}' 或配置 data.genomic.target_genes。"
        )
    logger.info(f"  目标基因表达列: '{gene_col}'")

    # ---- 协变量 ----
    covariate_cols = list(config.data.genomic.covariate_cols)
    # 自动过滤非法列
    covariate_cols = [c for c in covariate_cols if c in df.columns]
    # 协变量不含目标基因自身 (防泄露)
    covariate_cols = [c for c in covariate_cols if c != gene_col]

    if config.model.mode in ("omic", "pathomic") and len(covariate_cols) == 0:
        logger.warning("mode含omic但无有效协变量, 自动回退为 path-only")
        config.model.mode = "path"

    use_covariate = (config.model.mode in ("omic", "pathomic")) and len(covariate_cols) > 0
    if use_covariate:
        logger.info(f"  协变量: {len(covariate_cols)}列 -> {covariate_cols}")
    else:
        logger.info("  无协变量, 纯图像(path-only)预测")

    # ---- 病人ID -> 表达 / 生存 / 协变量 映射 ----
    expr_map = {}
    surv_map = {}
    cov_map = {}
    for idx, row in df.iterrows():
        pid = str(row[patient_id_col]).strip()
        try:
            expr = float(row[gene_col])
        except (ValueError, TypeError):
            expr = float("nan")
        if not np.isfinite(expr):
            continue
        expr_map[pid] = expr
        surv_map[pid] = {
            "e": float(row[censor_col]) if censor_col in df.columns else 0.0,
            "t": float(row[survtime_col]) if survtime_col in df.columns else 0.0,
        }
        if use_covariate:
            cov = pd.to_numeric(df.loc[idx, covariate_cols], errors="coerce").fillna(0.0).values.astype(np.float32)
            cov_map[pid] = cov

    logger.info(f"  有效表达样本: {len(expr_map)} 个病人")

    # ---- 扫描patch目录 ----
    reader = UpstreamPatchReader(config)
    patch_dir = reader.get_patch_dir()
    patch_map = {}
    if patch_dir.exists():
        slide_patches = defaultdict(list)
        for pf in sorted(patch_dir.glob("*.png")):
            sid = reader.parse_slide_id(pf.name)
            if sid:
                slide_patches[sid].append(str(pf.resolve()))
        num_p = config.upstream.num_patches_per_patient
        for sid, patches in slide_patches.items():
            if len(patches) >= num_p:
                patch_map[sid] = sorted(patches)[:num_p]
        logger.info(f"  合格slide: {len(patch_map)} (patch≥{num_p})")
    else:
        logger.warning(f"  patch目录不存在: {patch_dir}")

    short_to_single = {}
    for slide_id in patch_map:
        parts = slide_id.split("-")
        if len(parts) >= 3:
            short_id = "-".join(parts[:3])
            if short_id not in short_to_single:
                short_to_single[short_id] = slide_id
    logger.info(f"  短ID→长slide映射: {len(short_to_single)}")

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    output_folds = {}
    total_missing = 0
    val_ratio = float(getattr(config.cross_validation, "val_ratio", 0.2))

    def _collect(patient_ids):
        """收集一批病人的原始记录 (patient-level)。"""
        records = []
        missing = 0
        for pid in patient_ids:
            pid = str(pid).strip()
            if pid not in expr_map:
                missing += 1
                continue
            patches = patch_map.get(short_to_single.get(pid, ""), [])
            if len(patches) < config.upstream.num_patches_per_patient:
                missing += 1
                continue
            cov = cov_map.get(pid, np.zeros(len(covariate_cols), dtype=np.float32)) if use_covariate else None
            records.append({
                "pid": pid, "x_path": patches, "expr": expr_map[pid],
                "e": surv_map[pid]["e"], "t": surv_map[pid]["t"], "cov": cov,
            })
        return records, missing

    def _to_dict(records):
        n = len(records)
        cov = np.array([r["cov"] for r in records], dtype=np.float32) if use_covariate else np.zeros((n, 0), dtype=np.float32)
        return {
            "x_path": [r["x_path"] for r in records],
            "x_omic": cov,
            "e": np.array([r["e"] for r in records], dtype=np.float32),
            "t": np.array([r["t"] for r in records], dtype=np.float32),
            "g": np.array([r["expr"] for r in records], dtype=np.float32),
        }

    for fold_id in sorted(id_splits.keys()):
        fold_data = id_splits[fold_id]
        fold_key = f"fold_{fold_id}" if isinstance(fold_id, int) else str(fold_id)
        output_folds[fold_key] = {}

        train_records, m1 = _collect(fold_data.get("train", {}).get("patient_ids", []))
        test_records, m2 = _collect(fold_data.get("test", {}).get("patient_ids", []))
        total_missing += (m1 + m2)

        if len(train_records) == 0:
            logger.warning(f"  {fold_key} 训练集为空，跳过")
            continue

        # 训练集内再切出val (按病人ID, 同一病人不跨train/val/test)
        strat = np.array([r["expr"] for r in train_records])
        if task_type == "classification" and 0.0 < val_ratio < 1.0:
            strat_labels = build_binary_label(strat, compute_group_threshold(strat, config.model.group_threshold))
            try:
                train_records, val_records = train_test_split(
                    train_records, test_size=val_ratio, random_state=config.model.seed + fold_id,
                    stratify=strat_labels,
                )
            except ValueError:
                train_records, val_records = train_test_split(
                    train_records, test_size=val_ratio, random_state=config.model.seed + fold_id,
                )
        elif 0.0 < val_ratio < 1.0:
            # 回归: 按表达量分箱分层
            try:
                bins = pd.qcut(pd.Series(strat), q=min(5, len(train_records)),
                               labels=False, duplicates="drop").fillna(0).astype(int).values
            except Exception:
                bins = np.zeros(len(train_records), dtype=int)
            try:
                train_records, val_records = train_test_split(
                    train_records, test_size=val_ratio, random_state=config.model.seed + fold_id,
                    stratify=bins,
                )
            except ValueError:
                train_records, val_records = train_test_split(
                    train_records, test_size=val_ratio, random_state=config.model.seed + fold_id,
                )
        else:
            val_records = []

        # 分类阈值: 基于最终训练子集表达量计算 (避免泄露)
        if task_type == "classification":
            threshold = compute_group_threshold(
                np.array([r["expr"] for r in train_records]), config.model.group_threshold)
            logger.info(f"  {fold_key} 高/低表达阈值(训练集): {threshold:.4f}")
        else:
            threshold = None

        # 协变量标准化 (基于训练子集)
        scaler = None
        if use_covariate and len(train_records) > 0:
            scaler = StandardScaler()
            scaler.fit(np.array([r["cov"] for r in train_records], dtype=np.float32))

        def _finalize(records):
            d = _to_dict(records)
            if task_type == "classification":
                d["g"] = build_binary_label(d["g"], threshold)
            if scaler is not None and use_covariate and len(records) > 0:
                d["x_omic"] = scaler.transform(d["x_omic"]).astype(np.float32)
            return d

        output_folds[fold_key]["train"] = _finalize(train_records)
        if val_records:
            output_folds[fold_key]["val"] = _finalize(val_records)
        if test_records:
            output_folds[fold_key]["test"] = _finalize(test_records)

        logger.info(f"  {fold_key}: train={len(train_records)}, "
                    f"{'val=' + str(len(val_records)) + ', ' if val_records else ''}"
                    f"test={len(test_records)} (跳过{m1 + m2})")

    logger.info(f"  总跳过(缺patch/表达): {total_missing}")

    # 更新配置中的协变量维度
    config.model.omic.input_dim = len(covariate_cols) if use_covariate else 0

    # 保存完整数据到基因级缓存 (不覆盖原始病人ID划分pkl)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(output_folds, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"  完整数据pkl已缓存: {cache_path}")

    return output_folds


def _make_cache_path(config) -> Path:
    split_file = config.resolve_path(config.data.split_file)
    key = f"{config.model.target_gene}_{config.model.task_type}_{config.upstream.algorithm_name}"
    return split_file.with_name(split_file.stem + f"_{key}.full.pkl")


def _load_data_splits(config, logger=None) -> Dict[str, Any]:
    split_file = config.resolve_path(config.data.split_file)
    if not split_file.exists():
        raise FileNotFoundError(f"数据划分文件不存在: {split_file}\n请先运行: python make_split.py")

    cache_path = _make_cache_path(config)
    if cache_path.exists():
        with open(str(cache_path), "rb") as f:
            data = pickle.load(f)
        if logger:
            logger.info(f"  命中数据缓存: {cache_path}")
        return data

    with open(str(split_file), "rb") as f:
        raw_data = pickle.load(f)

    id_splits = None
    if isinstance(raw_data, dict) and "cv_splits" in raw_data:
        id_splits = raw_data["cv_splits"]
    elif isinstance(raw_data, dict) and "fold_1" in raw_data:
        sample_fold = raw_data.get("fold_1", raw_data.get(list(raw_data.keys())[0]))
        if isinstance(sample_fold, dict):
            sample_train = sample_fold.get("train", {})
            if "patient_ids" in sample_train:
                id_splits = {int(k.split("_")[-1]) if "_" in k else i + 1: v
                             for i, (k, v) in enumerate(sorted(raw_data.items()))}
    elif isinstance(raw_data, dict) and "data" in raw_data:
        id_splits = {i + 1: {"train": t, "test": s} for i, (t, s) in enumerate(raw_data["data"])}

    if id_splits is None:
        raise ValueError(f"无法解析数据划分文件 {split_file}")

    if logger is None:
        logger = logging.getLogger("YuHou")
    return _auto_build_full_splits(id_splits, config, logger, cache_path=cache_path)


# ============================================================
# 单折结果汇总
# ============================================================

def _compile_fold_result(model, metric_logger, test_result, fold_id, config) -> Dict[str, Any]:
    metrics = test_result.get("metrics", {})
    task_type = str(config.model.task_type).lower()

    fold_result = {
        "fold_id": fold_id,
        "best_epoch": metric_logger.best_epoch,
        "best_score": metric_logger.best_score,
        "metrics": metrics,
        "km_logrank_p": test_result.get("km_logrank_p", float("nan")),
        "pred": test_result.get("pred", np.array([])).tolist(),
        "prob": test_result.get("prob", np.array([])).tolist(),
        "survtime": test_result.get("survtime", np.array([])).tolist(),
        "censor": test_result.get("censor", np.array([])).tolist(),
        "training_history": metric_logger.to_dict(),
        "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if config.experiment.save_predictions else None,
    }
    return fold_result


def _is_valid_float(val) -> bool:
    if val is None:
        return False
    try:
        return not np.isnan(float(val))
    except (ValueError, TypeError):
        return False


# ============================================================
# CV汇总
# ============================================================

def _safe_mean_std(values, ndigits=4):
    valid = [v for v in values if _is_valid_float(v)]
    if not valid:
        return float("nan"), float("nan")
    return round(float(np.mean(valid)), ndigits), round(float(np.std(valid)), ndigits)


def _compute_cv_summary(fold_results: List[Dict], config) -> Dict:
    task_type = str(config.model.task_type).lower()
    n_folds = len(fold_results)
    if n_folds == 0:
        return {"n_folds": 0}

    summary = {"n_folds": n_folds, "fold_ids": [r["fold_id"] for r in fold_results],
               "task_type": task_type, "target_gene": config.model.target_gene,
               "algorithm": config.upstream.algorithm_name}

    if task_type == "regression":
        metric_keys = ["mse", "mae", "rmse", "pearson", "spearman", "r2"]
    else:
        metric_keys = ["acc", "balanced_acc", "precision", "recall", "f1", "auc", "ap", "mcc"]

    for key in metric_keys:
        vals = [r["metrics"].get(key) for r in fold_results]
        mean, std = _safe_mean_std(vals)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
        summary[f"{key}_all"] = vals

    km_ps = [r.get("km_logrank_p") for r in fold_results]
    valid_km = [p for p in km_ps if _is_valid_float(p)]
    summary["km_logrank_p_all"] = km_ps
    summary["km_logrank_significant_folds"] = sum(1 for p in valid_km if float(p) < 0.05)
    summary["best_epoch_mean"] = _safe_mean_std([r["best_epoch"] for r in fold_results], ndigits=1)[0]

    summary["config_summary"] = {
        "task_type": task_type,
        "target_gene": config.model.target_gene,
        "group_threshold": config.model.group_threshold,
        "fusion_type": config.model.fusion.type,
        "mode": config.model.mode,
        "algorithm": config.upstream.algorithm_name,
        "batch_size": config.training.batch_size,
        "learning_rate": config.training.optimizer.lr,
        "seed": config.model.seed,
    }
    return summary


def _log_cv_summary(logger, summary):
    task_type = summary.get("task_type")
    log_section(logger, "交叉验证最终结果", width=70)
    logger.info(f"  折数: {summary['n_folds']}, 基因: {summary.get('target_gene')}")
    logger.info("-" * 70)

    if task_type == "regression":
        for k in ["mse", "mae", "rmse", "pearson", "spearman"]:
            m = summary.get(f"{k}_mean")
            s = summary.get(f"{k}_std")
            if _is_valid_float(m):
                logger.info(f"  {k}: {m:.4f} +/- {s:.4f}")
    else:
        for k in ["acc", "balanced_acc", "f1", "auc"]:
            m = summary.get(f"{k}_mean")
            s = summary.get(f"{k}_std")
            if _is_valid_float(m):
                logger.info(f"  {k}: {m:.4f} +/- {s:.4f}")

    km_all = summary.get("km_logrank_p_all", [])
    if km_all:
        logger.info(f"  附属KM log-rank p<0.05折数: {summary.get('km_logrank_significant_folds', 0)}/{len(km_all)}")

    cfg = summary.get("config_summary", {})
    logger.info("-" * 70)
    logger.info(f"  任务类型={cfg.get('task_type')}, 融合={cfg.get('fusion_type')}, "
                f"算法={cfg.get('algorithm')}, 模式={cfg.get('mode')}")
    logger.info("=" * 70)


# ============================================================
# 结果CSV保存 (每基因)
# ============================================================

def _save_results_csv(fold_results: List[Dict], cv_summary: Dict, config):
    task_type = str(config.model.task_type).lower()
    gene = config.model.target_gene
    results_dir = config.get_subdir("results")

    if task_type == "regression":
        cols = ["fold", "mse", "mae", "rmse", "pearson", "spearman", "r2", "km_logrank_p"]
    else:
        cols = ["fold", "acc", "balanced_acc", "precision", "recall", "f1", "auc", "ap", "mcc", "km_logrank_p"]

    rows = []
    for r in fold_results:
        row = {"fold": r["fold_id"]}
        for c in cols[1:]:
            if c == "km_logrank_p":
                row[c] = r.get("km_logrank_p", float("nan"))
            else:
                row[c] = r["metrics"].get(c, float("nan"))
        rows.append(row)

    # 追加mean±std行
    mean_row = {"fold": "mean"}
    std_row = {"fold": "std"}
    for c in cols[1:]:
        if c == "km_logrank_p":
            continue
        mean_row[c] = cv_summary.get(f"{c}_mean", float("nan"))
        std_row[c] = cv_summary.get(f"{c}_std", float("nan"))
    rows.append(mean_row)
    rows.append(std_row)

    df_out = pd.DataFrame(rows)
    csv_path = results_dir / f"{gene}_results.csv"
    df_out.to_csv(csv_path, index=False, float_format="%.4f")
    return csv_path


# ============================================================
# 附属KM图
# ============================================================

def _plot_auxiliary_km(fold_result, fold_id, config):
    """用预测分子表型分组画KM曲线。"""
    if not config.evaluation.auxiliary_survival.enabled or not config.visualization.enabled:
        return
    survtime = np.asarray(fold_result.get("survtime", []))
    censor = np.asarray(fold_result.get("censor", []))
    prob = np.asarray(fold_result.get("prob", []))
    pred = np.asarray(fold_result.get("pred", []))

    if len(survtime) < 2:
        return

    task_type = str(config.model.task_type).lower()
    # 分组: 分类用预测标签(0/1), 回归用预测值中位数
    if task_type == "classification":
        group = pred.astype(int)
    else:
        group = (prob > np.median(prob)).astype(int)

    try:
        from src.visualization.km_curves import plot_km_by_group
        fig_dir = config.get_subdir("figures")
        save_path = fig_dir / f"fold{fold_id}_km_by_predicted_group.png"
        plot_km_by_group(group, censor.astype(bool), survtime, save_path=str(save_path))
    except Exception as e:
        logging.getLogger("YuHou").warning(f"  附属KM图绘制失败: {e}")


# ============================================================
# 主流程: run_cross_validation()
# ============================================================

def run_cross_validation(config) -> Dict[str, Any]:
    log_dir = config.get_subdir("logs")
    logger = setup_logger(log_file=str(log_dir / "cv_train.log"),
                          level=config.logging.level, language=config.logging.language)

    logger.info("=" * 70)
    logger.info(f"  DealWith Histomolecular {config.project.version} - 交叉验证启动")
    logger.info(f"  目标基因: {config.model.target_gene}, 任务: {config.model.task_type}, "
                f"算法: {config.upstream.algorithm_name}")
    logger.info("=" * 70)

    sys_info = _get_system_info()
    _log_system_info(logger, sys_info)

    if config.training.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.training.device)
    if device.type == "cuda":
        gpu_id = config.training.gpu_id
        if gpu_id >= torch.cuda.device_count():
            gpu_id = 0
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")

    set_seed(config.model.seed, deterministic=False, logger=logger)

    logger.info("")
    log_section(logger, "加载数据划分", width=70)
    data_splits = _load_data_splits(config, logger)
    logger.info(f"  可用折数: {len(data_splits)}")

    fold_ids = config.cross_validation.fold_ids
    available_fold_names = sorted(data_splits.keys())
    run_folds = [f"fold_{fid}" for fid in fold_ids if f"fold_{fid}" in data_splits]
    if not run_folds:
        run_folds = available_fold_names[:config.cross_validation.n_folds]
    logger.info(f"  实际运行折: {run_folds}")

    cv_start = time.time()
    fold_results = []

    for fold_idx, fold_name in enumerate(run_folds):
        fold_id = int(fold_name.split("_")[-1]) if "_" in fold_name else fold_idx + 1
        logger.info("")
        logger.info("#" * 70)
        logger.info(f"  >>> {fold_name} (第{fold_idx + 1}/{len(run_folds)}折) <<<")
        logger.info("#" * 70)

        fold_data = data_splits[fold_name]
        if "train" not in fold_data:
            logger.error(f"  {fold_name}缺少训练集，跳过")
            continue

        try:
            model, optimizer, metric_logger = train(config=config, data=fold_data,
                                                     device=device, fold_id=fold_id)
            test_split = "test" if "test" in fold_data else "val"
            if test_split in fold_data:
                test_result = test(config=config, model=model, data=fold_data,
                                   split=test_split, device=device)
                fold_result = _compile_fold_result(model, metric_logger, test_result,
                                                   fold_id, config)
                fold_results.append(fold_result)
                _plot_auxiliary_km(fold_result, fold_id, config)

            if device.type == "cuda":
                torch.cuda.empty_cache()
            del model
        except Exception as e:
            logger.error(f"  [{fold_name}] 训练/测试失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

    if not fold_results:
        logger.error("所有折均失败!")
        return {"error": "所有折训练失败", "fold_results": [], "cv_summary": {}}

    cv_summary = _compute_cv_summary(fold_results, config)
    cv_summary["total_time_minutes"] = round((time.time() - cv_start) / 60, 1)
    _log_cv_summary(logger, cv_summary)

    # 保存CSV (每基因)
    try:
        csv_path = _save_results_csv(fold_results, cv_summary, config)
        logger.info(f"  实验结果CSV已保存: {csv_path}")
    except Exception as e:
        logger.warning(f"  结果CSV保存失败: {e}")

    # 保存完整CV结果pkl
    results_dir = config.get_subdir("results")
    cv_package = {"cv_summary": cv_summary, "fold_results": fold_results,
                  "system_info": sys_info, "config": config.to_dict(),
                  "timestamp": datetime.now().isoformat()}
    with open(str(results_dir / "cv_results.pkl"), "wb") as f:
        pickle.dump(cv_package, f)
    logger.info(f"  CV结果pkl已保存: {results_dir / 'cv_results.pkl'}")

    return cv_package


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Histomolecular交叉验证训练")
    parser.add_argument("--config", "-c", type=str, default="config/default_config.yaml")
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--target_gene", type=str, default=None)
    parser.add_argument("--task_type", type=str, default=None, choices=["classification", "regression"])
    parser.add_argument("--algorithm_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--fusion_type", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    from config.config_loader import load_config, print_config
    config = load_config(config_path=args.config)
    if args.gpu_id is not None:
        config.training.gpu_id = args.gpu_id
    if args.target_gene is not None:
        config.model.target_gene = args.target_gene
    if args.task_type is not None:
        config.model.task_type = args.task_type
    if args.algorithm_name is not None:
        config.upstream.algorithm_name = args.algorithm_name
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.fusion_type is not None:
        config.model.fusion.type = args.fusion_type
    if args.exp_name is not None:
        config.experiment.name = args.exp_name
    if args.seed is not None:
        config.model.seed = args.seed

    from config.config_loader import create_experiment_dirs
    create_experiment_dirs(config)
    print_config(config)

    run_cross_validation(config)


if __name__ == "__main__":
    main()
