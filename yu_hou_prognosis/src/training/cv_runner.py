# -*- coding: utf-8 -*-
"""
cv_runner.py
============================================================================
交叉验证主控制器模块。

功能:
    1. 从pickle文件加载数据划分
    2. 对每个fold依次执行train()和test()
    3. 收集各fold结果，计算交叉验证均值±标准差
    4. 保存最终CV结果为pickle文件
    5. 启动时记录系统信息（GPU、CPU、RAM）
    6. 中文日志输出CV摘要

典型用法:
    python -m src.training.cv_runner --config config/default_config.yaml --gpu_id 0

    或通过代码调用:
        from src.training.cv_runner import run_cross_validation
        cv_results = run_cross_validation(config)
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

import numpy as np
import torch
import psutil  # CPU/RAM监控

import pandas as pd
import numpy as np
from collections import defaultdict

from src.utils.seed import set_seed
from src.utils.logger import setup_logger, log_section
from src.training.trainer import train, test, MetricLogger
from src.data_loading.upstream_reader import UpstreamPatchReader


# ============================================================
# 系统信息采集
# ============================================================

def _get_system_info() -> Dict[str, Any]:
    """
    采集运行环境的系统信息。

    返回:
        dict: 包含操作系统、CPU、GPU、RAM等信息的字典
    """
    info = {
        "timestamp": datetime.now().isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
    }

    # ---- CPU信息 ----
    info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    info["cpu_count_logical"] = psutil.cpu_count(logical=True)
    info["cpu_percent"] = psutil.cpu_percent(interval=0.5)

    # ---- RAM信息 ----
    mem = psutil.virtual_memory()
    info["ram_total_gb"] = round(mem.total / (1024 ** 3), 1)
    info["ram_available_gb"] = round(mem.available / (1024 ** 3), 1)
    info["ram_used_percent"] = mem.percent

    # ---- GPU信息 ----
    info["gpu_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()

        gpu_details = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_details.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 1),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            })
        info["gpu_details"] = gpu_details
    else:
        info["gpu_count"] = 0
        info["cuda_version"] = None
        info["gpu_details"] = []

    return info


def _log_system_info(logger: logging.Logger, sys_info: Dict[str, Any]) -> None:
    """
    以中文格式输出系统信息到日志。

    参数:
        logger: 日志器实例
        sys_info: _get_system_info()的返回值
    """
    log_section(logger, "系统信息", width=70)

    logger.info(f"  运行时间:       {sys_info['timestamp']}")
    logger.info(f"  主机名:         {sys_info['hostname']}")
    logger.info(f"  操作系统:       {sys_info['platform']}")
    logger.info(f"  Python版本:     {sys_info['python_version'].split()[0]}")
    logger.info(f"  PyTorch版本:    {sys_info['pytorch_version']}")
    logger.info("-" * 70)
    logger.info(f"  CPU物理核心:    {sys_info['cpu_count_physical']}")
    logger.info(f"  CPU逻辑核心:    {sys_info['cpu_count_logical']}")
    logger.info(f"  CPU使用率:      {sys_info['cpu_percent']:.1f}%")
    logger.info(f"  总内存:         {sys_info['ram_total_gb']:.1f} GB")
    logger.info(f"  可用内存:       {sys_info['ram_available_gb']:.1f} GB")
    logger.info(f"  内存使用率:     {sys_info['ram_used_percent']:.1f}%")
    logger.info("-" * 70)

    if sys_info["gpu_available"]:
        logger.info(f"  GPU数量:        {sys_info['gpu_count']}")
        logger.info(f"  CUDA版本:       {sys_info['cuda_version']}")
        logger.info(f"  cuDNN版本:      {sys_info['cudnn_version']}")
        for gpu in sys_info["gpu_details"]:
            logger.info(f"  GPU[{gpu['index']}]: {gpu['name']}")
            logger.info(f"    显存总量:     {gpu['total_memory_gb']:.1f} GB")
            logger.info(f"    计算能力:     {gpu['compute_capability']}")
            logger.info(f"    流处理器数:   {gpu['multi_processor_count']}")
    else:
        logger.warning(f"  GPU:            不可用 (将使用CPU训练)")
    logger.info("=" * 70)


# ============================================================
# 数据划分加载 + 自动构建
# ============================================================

def _auto_build_full_splits(id_splits: Dict, config, logger) -> Dict[str, Any]:
    """
    当pkl仅包含病人ID划分时，自动扫描上游patch和基因组CSV，
    构建完整的 x_path/x_omic/e/t/g 数据划分。
    """
    logger.info("检测到pkl仅含病人ID，自动构建完整数据划分...")

    # ---- 读取基因组CSV ----
    csv_path = config.resolve_path(config.data.genomic_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"基因组CSV不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  基因组CSV: {len(df)} 个样本")

    # CSV列名
    patient_id_col = "TCGA ID"
    event_col = "event"
    time_col = "Survival months"

    # 过滤数值特征列
    exclude_patterns = [
        "tcga", "id", "indexes", "event", "censored", "survival",
        "vital_status", "survival_source", "tumor_grade", "tumor_stage",
        "histological_type", "suspicious", "gender", "age_at_diagnosis",
        "codeletion", "idh mutation",
    ]
    feature_cols = []
    for c in df.columns:
        cl = c.lower()
        if any(p in cl for p in exclude_patterns):
            continue
        if df[c].dtype in (np.float64, np.float32, np.int64, np.int32):
            feature_cols.append(c)
    logger.info(f"  基因组特征维度: {len(feature_cols)}")

    # 更新配置
    if len(feature_cols) != config.data.genomic.input_dim:
        logger.info(f"  维度适配: {config.data.genomic.input_dim} -> {len(feature_cols)}")
        config.data.genomic.input_dim = len(feature_cols)
        config.model.omic.input_dim = len(feature_cols)

    # 构建病人ID → 基因组特征映射
    genomic_map = {}
    survival_map = {}
    for _, row in df.iterrows():
        pid = str(row[patient_id_col]).strip()
        genomic_map[pid] = row[feature_cols].values.astype(np.float32)
        survival_map[pid] = {
            "e": float(row[event_col]),
            "t": float(row[time_col]),
            "g": 0,
        }
    logger.info(f"  基因组映射: {len(genomic_map)} 个病人")

    # ---- 扫描patch目录 ----
    reader = UpstreamPatchReader(config)
    patch_dir = reader.get_patch_dir()
    patch_map = {}  # slide_id → [patch_paths]
    if patch_dir.exists():
        png_files = sorted(patch_dir.glob("*.png"))
        logger.info(f"  扫描patch: {len(png_files)} 个PNG文件")
        slide_patches = defaultdict(list)
        for pf in png_files:
            sid = reader.parse_slide_id(pf.name)
            if sid:
                slide_patches[sid].append(str(pf.resolve()))
        num_p = config.upstream.num_patches_per_patient
        for sid, patches in slide_patches.items():
            if len(patches) >= num_p:
                patch_map[sid] = sorted(patches)[:num_p]
        logger.info(f"  合格slide: {len(patch_map)} 个 (patch≥{num_p})")
    else:
        logger.warning(f"  patch目录不存在: {patch_dir}")

    # 短ID → 长slide ID 映射
    short_to_single = {}
    for slide_id in patch_map:
        parts = slide_id.split("-")
        if len(parts) >= 3:
            short_id = "-".join(parts[:3])
            if short_id not in short_to_single:
                short_to_single[short_id] = slide_id
    logger.info(f"  短ID→长slide映射: {len(short_to_single)} 个")

    # ---- 构建每折完整数据 ----
    output_folds = {}
    total_missing = 0
    for fold_id in sorted(id_splits.keys()):
        fold_data = id_splits[fold_id]
        fold_key = f"fold_{fold_id}" if isinstance(fold_id, int) else str(fold_id)
        output_folds[fold_key] = {}

        for split_name in ["train", "test"]:
            if split_name not in fold_data:
                continue
            patient_ids = fold_data[split_name].get("patient_ids", [])

            x_path_list, x_omic_list, e_list, t_list, g_list = [], [], [], [], []
            fold_missing = 0

            for pid in patient_ids:
                pid = str(pid).strip()
                if pid not in genomic_map:
                    fold_missing += 1; continue
                patches = patch_map.get(short_to_single.get(pid, ""), [])
                if len(patches) < config.upstream.num_patches_per_patient:
                    fold_missing += 1; continue

                x_omic_list.append(genomic_map[pid])
                s = survival_map[pid]
                e_list.append(s["e"]); t_list.append(s["t"]); g_list.append(s["g"])
                x_path_list.append(patches)

            n = len(x_omic_list)
            output_folds[fold_key][split_name] = {
                "x_path": x_path_list,
                "x_omic": np.array(x_omic_list, dtype=np.float32) if x_omic_list else np.array([]),
                "e": np.array(e_list, dtype=np.float32) if e_list else np.array([]),
                "t": np.array(t_list, dtype=np.float32) if t_list else np.array([]),
                "g": g_list,
            }
            logger.info(f"  {fold_key}/{split_name}: {n}病人 (跳过{fold_missing})")
            total_missing += fold_missing

    logger.info(f"  总跳过(缺数据): {total_missing}")

    # 保存完整pkl，下次直接加载
    output_path = config.resolve_path(config.data.split_file)
    if output_path.exists():
        bak = output_path.with_suffix(".pkl.bak")
        output_path.rename(bak)
    with open(output_path, "wb") as f:
        pickle.dump(output_folds, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"  完整数据pkl已保存: {output_path}")

    return output_folds


def _load_data_splits(config, logger=None) -> Dict[str, Any]:
    """
    从pickle文件加载交叉验证数据划分。

    参数:
        config: ConfigBundle配置对象

    返回:
        data_splits: dict of dict
            {
                "fold_1": {"train": {...}, "test": {...}},
                "fold_2": {"train": {...}, "test": {...}},
                ...
            }
            每个划分包含:
                - x_path: List[List[str]]  病人patch路径列表
                - x_omic: np.ndarray [N, dim] 基因组特征
                - e: np.ndarray [N] 删失标记
                - t: np.ndarray [N] 生存时间
                - g: List 标签

    异常:
        FileNotFoundError: pickle文件不存在
        ValueError: 数据格式无效
    """
    split_file = config.resolve_path(config.data.split_file)

    if not split_file.exists():
        raise FileNotFoundError(
            f"数据划分文件不存在: {split_file}\n"
            f"请确保已运行数据划分脚本生成 {config.data.split_file}"
        )

    with open(str(split_file), "rb") as f:
        raw_data = pickle.load(f)

    # 抽取id_splits（统一各种格式为 {fold_id: fold_data}）
    id_splits = None  # 仅含patient_ids的原始划分

    if isinstance(raw_data, dict) and "cv_splits" in raw_data:
        id_splits = raw_data["cv_splits"]
    elif isinstance(raw_data, dict) and "fold_1" in raw_data:
        # 检查是否已含实际数据
        sample_fold = raw_data.get("fold_1", raw_data.get(list(raw_data.keys())[0]))
        if isinstance(sample_fold, dict):
            sample_train = sample_fold.get("train", {})
            if "x_path" in sample_train and len(sample_train.get("x_path", [])) > 0:
                return raw_data  # 已有完整数据，直接返回
            elif "patient_ids" in sample_train:
                id_splits = {int(k.split("_")[-1]) if "_" in k else i+1: v
                            for i, (k, v) in enumerate(sorted(raw_data.items()))}
    elif isinstance(raw_data, dict) and "data" in raw_data:
        id_splits = {i+1: {"train": t, "test": s} for i, (t, s) in enumerate(raw_data["data"])}
    elif isinstance(raw_data, list):
        id_splits = {}
        for i, item in enumerate(raw_data):
            if isinstance(item, dict) and "train" in item:
                id_splits[i+1] = item
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                id_splits[i+1] = {"train": item[0], "test": item[1]}

    if id_splits is not None:
        # 需要自动构建完整数据
        if logger is None:
            logger = logging.getLogger("YuHou")
        return _auto_build_full_splits(id_splits, config, logger)

    raise ValueError(
        f"无法解析数据划分文件 {split_file}。\n"
        f"支持格式: dict with fold keys / dict with 'data' key / list of folds / cv_splits"
    )


# ============================================================
# 单折结果汇总
# ============================================================

def _compile_fold_result(model, metric_logger, test_results, fold_id: int,
                         config) -> Dict[str, Any]:
    """
    将单折训练和测试结果汇总为统一字典。

    参数:
        model: 训练完成的模型
        metric_logger: MetricLogger实例
        test_results: test()函数的返回值元组
        fold_id: 折号
        config: ConfigBundle配置对象

    返回:
        dict: 该折的完整结果
    """
    (loss, cindex, pvalue, surv_acc, grad_acc, pred,
     td_auc, binary_metrics, group_summary, hr, cls_metrics) = test_results

    fold_result = {
        "fold_id": fold_id,
        "best_epoch": metric_logger.best_epoch,
        "best_train_cindex": metric_logger.best_cindex,
        "test_loss": loss,
        "test_cindex": cindex,
        "test_logrank_pvalue": pvalue if not np.isnan(pvalue) else float("nan"),
        "test_tdauc_mean": td_auc.get("mean_auc", float("nan")) if isinstance(td_auc, dict) else float("nan"),
        "test_tdauc_times": td_auc.get("times", []) if isinstance(td_auc, dict) else [],
        "test_tdauc_values": td_auc.get("auc_values", []) if isinstance(td_auc, dict) else [],
        "test_hazard_ratio": float(hr) if isinstance(hr, (int, float)) and not np.isnan(float(hr)) else float("nan"),
        "test_binary_metrics": binary_metrics,
        "test_group_summary": group_summary,
        "test_surv_accuracy": surv_acc,
        "test_cls_metrics": cls_metrics,
        "test_predictions": pred.tolist() if isinstance(pred, np.ndarray) else [],
        "training_history": metric_logger.to_dict(),
        "model_state_dict": copy_state_dict_for_cpu(model.state_dict())
        if config.experiment.save_predictions else None,
    }

    return fold_result


def copy_state_dict_for_cpu(state_dict: Dict) -> Dict:
    """将state_dict中的所有张量转移到CPU (节省GPU显存)"""
    return {k: v.cpu().clone() for k, v in state_dict.items()}


def _is_valid_float(val) -> bool:
    """检查值是否为有效的浮点数"""
    if val is None:
        return False
    try:
        f = float(val)
        return not np.isnan(f)
    except (ValueError, TypeError):
        return False


# ============================================================
# 计算CV汇总统计
# ============================================================

def _compute_cv_summary(fold_results: List[Dict[str, Any]],
                        config) -> Dict[str, Any]:
    """
    对所有fold结果计算交叉验证均值±标准差。

    参数:
        fold_results: 各fold结果字典列表
        config: ConfigBundle配置对象

    返回:
        dict: CV汇总结果
    """
    n_folds = len(fold_results)
    if n_folds == 0:
        return {"error": "无有效fold结果", "n_folds": 0}

    def _safe_mean_std(values, ndigits=4):
        """安全计算均值和标准差"""
        valid = [v for v in values if _is_valid_float(v)]
        if not valid:
            return float("nan"), float("nan")
        return round(float(np.mean(valid)), ndigits), round(float(np.std(valid)), ndigits)

    # 收集各折指标
    cindices = [r["test_cindex"] for r in fold_results]
    pvalues = [r["test_logrank_pvalue"] for r in fold_results]
    tdaucs = [r["test_tdauc_mean"] for r in fold_results]
    hrs = [r["test_hazard_ratio"] for r in fold_results]
    losses = [r["test_loss"] for r in fold_results]
    best_train_cindices = [r["best_train_cindex"] for r in fold_results]
    best_epochs = [r["best_epoch"] for r in fold_results]

    # 二分类指标
    binary_accs = [r["test_binary_metrics"].get("accuracy", float("nan"))
                   for r in fold_results if r.get("test_binary_metrics")]
    binary_f1s = [r["test_binary_metrics"].get("f1_score", float("nan"))
                  for r in fold_results if r.get("test_binary_metrics")]

    # 时间依赖AUC逐时间点统计 (跨fold)
    per_time_aucs = {}
    all_times = set()
    for r in fold_results:
        times = r.get("test_tdauc_times", [])
        values = r.get("test_tdauc_values", [])
        for t, v in zip(times, values):
            all_times.add(round(float(t), 1))
    for t in sorted(all_times):
        fold_vals = []
        for r in fold_results:
            times = r.get("test_tdauc_times", [])
            values = r.get("test_tdauc_values", [])
            for rt, rv in zip(times, values):
                if abs(round(float(rt), 1) - t) < 0.01:
                    fold_vals.append(float(rv))
                    break
        if fold_vals:
            mean, std = _safe_mean_std(fold_vals)
            per_time_aucs[f"t{t:.0f}"] = {"mean": mean, "std": std, "n_folds": len(fold_vals)}

    summary = {
        "n_folds": n_folds,
        "fold_ids": [r["fold_id"] for r in fold_results],

        # 核心指标
        "cindex_mean": _safe_mean_std(cindices)[0],
        "cindex_std": _safe_mean_std(cindices)[1],
        "cindex_all": cindices,

        "tdauc_mean": _safe_mean_std(tdaucs)[0],
        "tdauc_std": _safe_mean_std(tdaucs)[1],
        "tdauc_all": tdaucs,
        "tdauc_per_time": per_time_aucs,

        "loss_mean": _safe_mean_std(losses)[0],
        "loss_std": _safe_mean_std(losses)[1],
        "loss_all": losses,

        "logrank_pvalue_mean": _safe_mean_std(pvalues)[0],
        "logrank_pvalue_all": pvalues,

        "hazard_ratio_mean": _safe_mean_std(hrs)[0],
        "hazard_ratio_std": _safe_mean_std(hrs)[1],
        "hazard_ratio_all": hrs,

        "best_train_cindex_mean": _safe_mean_std(best_train_cindices)[0],
        "best_epoch_mean": _safe_mean_std(best_epochs, ndigits=1)[0],

        # 二分类指标
        "binary_accuracy_mean": _safe_mean_std(binary_accs)[0],
        "binary_accuracy_std": _safe_mean_std(binary_accs)[1],
        "binary_f1_mean": _safe_mean_std(binary_f1s)[0],
        "binary_f1_std": _safe_mean_std(binary_f1s)[1],

        # 配置引用
        "config_summary": {
            "task": config.model.task,
            "fusion_type": config.model.fusion.type,
            "batch_size": config.training.batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "learning_rate": config.training.optimizer.lr,
            "total_epochs": config.training.scheduler.n_epochs + config.training.scheduler.n_epochs_decay,
            "eval_times": config.evaluation.eval_times,
            "amp_enabled": config.training.amp.enabled,
            "reg_type": config.training.loss.reg_type,
            "seed": config.model.seed,
        },

        "timestamp": datetime.now().isoformat(),
    }

    return summary


# ============================================================
# CV结果日志输出
# ============================================================

def _log_cv_summary(logger: logging.Logger, cv_summary: Dict[str, Any]) -> None:
    """
    以中文格式打印CV最终摘要到日志。

    参数:
        logger: 日志器实例
        cv_summary: _compute_cv_summary()的返回值
    """
    n_folds = cv_summary.get("n_folds", 0)
    if n_folds == 0:
        logger.warning("无有效CV结果可输出")
        return

    log_section(logger, "交叉验证最终结果", width=70)
    logger.info(f"  折数: {n_folds}")
    logger.info(f"  折ID: {cv_summary['fold_ids']}")
    logger.info("-" * 70)

    # C-index
    ci_mean = cv_summary["cindex_mean"]
    ci_std = cv_summary["cindex_std"]
    logger.info(f"  C-index:     {ci_mean:.4f} +/- {ci_std:.4f}")
    logger.info(f"  各折C-index: {[f'{c:.4f}' for c in cv_summary.get('cindex_all', [])]}")

    # tdAUC
    tdauc_mean = cv_summary["tdauc_mean"]
    tdauc_std = cv_summary["tdauc_std"]
    if not np.isnan(tdauc_mean):
        logger.info(f"  tdAUC:       {tdauc_mean:.4f} +/- {tdauc_std:.4f}")

    # 逐时间点tdAUC
    per_time = cv_summary.get("tdauc_per_time", {})
    if per_time:
        time_parts = []
        for t_key, t_info in sorted(per_time.items()):
            t_mean = t_info["mean"]
            t_std = t_info["std"]
            if not np.isnan(t_mean):
                time_parts.append(f"{t_key}: {t_mean:.4f}+/-{t_std:.4f}")
        if time_parts:
            logger.info(f"  逐时间tdAUC:  {', '.join(time_parts)}")

    # Hazard Ratio
    hr_mean = cv_summary["hazard_ratio_mean"]
    hr_std = cv_summary["hazard_ratio_std"]
    if not np.isnan(hr_mean):
        logger.info(f"  Hazard Ratio:{hr_mean:.4f} +/- {hr_std:.4f}")

    # 二分类指标
    bin_acc_mean = cv_summary.get("binary_accuracy_mean", float("nan"))
    bin_acc_std = cv_summary.get("binary_accuracy_std", float("nan"))
    if not np.isnan(bin_acc_mean):
        logger.info(f"  二分类准确率: {bin_acc_mean:.4f} +/- {bin_acc_std:.4f}")

    bin_f1_mean = cv_summary.get("binary_f1_mean", float("nan"))
    bin_f1_std = cv_summary.get("binary_f1_std", float("nan"))
    if not np.isnan(bin_f1_mean):
        logger.info(f"  二分类F1:     {bin_f1_mean:.4f} +/- {bin_f1_std:.4f}")

    # 训练信息
    logger.info("-" * 70)
    logger.info(f"  平均最佳轮次: {cv_summary.get('best_epoch_mean', 'N/A')}")
    logger.info(f"  平均训练最佳C-index: {cv_summary.get('best_train_cindex_mean', 'N/A'):.4f}"
                if _is_valid_float(cv_summary.get('best_train_cindex_mean'))
                else f"  平均训练最佳C-index: N/A")

    # Log-rank p值分布
    pvalues = cv_summary.get("logrank_pvalue_all", [])
    if pvalues:
        significant = sum(1 for p in pvalues if _is_valid_float(p) and float(p) < 0.05)
        logger.info(f"  Log-rank p<0.05的折数: {significant}/{len(pvalues)}")

    # 配置确认
    cfg = cv_summary.get("config_summary", {})
    logger.info("-" * 70)
    logger.info(f"  任务类型:     {cfg.get('task', 'N/A')}")
    logger.info(f"  融合策略:     {cfg.get('fusion_type', 'N/A')}")
    logger.info(f"  有效批次大小: {cfg.get('batch_size', 'N/A')} x "
                f"{cfg.get('gradient_accumulation_steps', 'N/A')}")
    logger.info(f"  学习率:       {cfg.get('learning_rate', 'N/A')}")
    logger.info(f"  总训练轮数:   {cfg.get('total_epochs', 'N/A')}")
    logger.info(f"  AMP混合精度:  {'启用' if cfg.get('amp_enabled') else '禁用'}")
    logger.info("=" * 70)


# ============================================================
# 主流程: run_cross_validation()
# ============================================================

def run_cross_validation(config) -> Dict[str, Any]:
    """
    执行完整的K折交叉验证流程。

    流程:
        1. 加载数据划分
        2. 对每个fold:
            a. 构建数据和设备
            b. 调用train()训练模型
            c. 调用test()评估模型
            d. 收集fold结果
        3. 计算CV均值±标准差
        4. 保存CV结果到pickle
        5. 输出CV摘要日志

    参数:
        config: ConfigBundle配置对象

    返回:
        cv_package: 包含所有CV结果的字典
            {
                "cv_summary": dict,      # CV汇总统计
                "fold_results": list,    # 各折详细结果
                "system_info": dict,     # 系统信息
                "config": dict,          # 配置字典
            }

    异常:
        FileNotFoundError: 数据划分文件不存在
        ValueError: 数据格式无效
        RuntimeError: 训练/测试失败
    """
    # ---- 初始化日志 ----
    log_dir = config.get_subdir("logs")
    log_file = str(log_dir / "cv_train.log")

    logger = setup_logger(
        log_file=log_file,
        level=config.logging.level,
        language=config.logging.language,
    )

    logger.info("=" * 70)
    logger.info(f"  YuHou {config.project.version} - 交叉验证启动")
    logger.info(f"  配置文件: {config.config_path}")
    logger.info(f"  实验名称: {config.experiment.name}")
    logger.info("=" * 70)

    # ---- 系统信息 ----
    sys_info = _get_system_info()
    _log_system_info(logger, sys_info)

    # ---- 设置设备 ----
    if config.training.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.training.device)

    if device.type == "cuda":
        gpu_id = config.training.gpu_id
        if gpu_id >= torch.cuda.device_count():
            logger.warning(f"GPU ID {gpu_id} 不可用，回退到GPU 0")
            gpu_id = 0
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        logger.info(f"  使用设备: {device} ({torch.cuda.get_device_name(gpu_id)})")
    else:
        logger.info(f"  使用设备: CPU")

    # ---- 设置随机种子 ----
    set_seed(config.model.seed, deterministic=False, logger=logger)

    # ---- 加载数据划分 ----
    logger.info("")
    log_section(logger, "加载数据划分", width=70)

    try:
        data_splits = _load_data_splits(config, logger)
    except Exception as e:
        logger.error(f"  数据加载失败: {e}")
        raise

    n_folds_available = len(data_splits)
    logger.info(f"  数据划分文件: {config.data.split_file}")
    logger.info(f"  可用折数: {n_folds_available}")

    # 确定要运行的fold
    fold_ids = config.cross_validation.fold_ids
    # 过滤不存在的fold
    available_fold_names = sorted(data_splits.keys())
    if any(isinstance(fid, int) for fid in fold_ids):
        # 整数格式: [1, 2, 3, 4, 5]
        run_folds = [f"fold_{fid}" for fid in fold_ids
                     if f"fold_{fid}" in data_splits]
    else:
        # 字符串格式: ["fold_1", "fold_2", ...]
        run_folds = [fid for fid in fold_ids if fid in data_splits]

    if not run_folds:
        logger.warning(f"  指定的fold {fold_ids} 不在数据中，使用前{config.cross_validation.n_folds}折")
        run_folds = available_fold_names[:config.cross_validation.n_folds]

    logger.info(f"  实际运行折: {run_folds}")
    logger.info("=" * 70)

    # ---- 逐折训练与测试 ----
    cv_start_time = time.time()
    fold_results = []
    fold_models = []  # (可选) 保存模型用于ensemble

    for fold_idx, fold_name in enumerate(run_folds):
        fold_id = int(fold_name.split("_")[-1]) if "_" in fold_name else fold_idx + 1
        logger.info("")
        logger.info("#" * 70)
        logger.info(f"  >>> 开始处理 {fold_name} (第{fold_idx + 1}/{len(run_folds)}折) <<<")
        logger.info("#" * 70)

        fold_data = data_splits[fold_name]

        # 验证数据格式
        if "train" not in fold_data and "test" not in fold_data:
            # 如果数据直接就是train/test列表
            if isinstance(fold_data, (list, tuple)) and len(fold_data) >= 2:
                fold_data = {"train": fold_data[0], "test": fold_data[1]}
            else:
                logger.error(f"  {fold_name}数据格式无效，跳过")
                continue

        if "train" not in fold_data:
            logger.error(f"  {fold_name}缺少训练集，跳过")
            continue

        try:
            # ---- 训练 ----
            fold_start_time = time.time()

            model, optimizer, metric_logger = train(
                config=config,
                data=fold_data,
                device=device,
                fold_id=fold_id,
            )

            train_time = time.time() - fold_start_time
            logger.info(f"  [{fold_name}] 训练耗时: {train_time / 60:.1f} 分钟")

            # ---- 测试 ----
            test_split = "test" if "test" in fold_data else "val"
            if test_split in fold_data:
                test_results = test(
                    config=config,
                    model=model,
                    data=fold_data,
                    split=test_split,
                    device=device,
                )

                # 汇总本折结果
                fold_result = _compile_fold_result(
                    model=model,
                    metric_logger=metric_logger,
                    test_results=test_results,
                    fold_id=fold_id,
                    config=config,
                )
                fold_results.append(fold_result)

                (_, cindex, pvalue, _, _, _, td_auc, _, _, hr, _) = test_results
                logger.info(f"  [{fold_name}] 测试C-index: {cindex:.4f} | "
                            f"训练最佳C-index: {metric_logger.best_cindex:.4f}")
            else:
                logger.warning(f"  [{fold_name}] 无测试集，仅记录训练指标")
                # 创建最小fold_result
                fold_result = {
                    "fold_id": fold_id,
                    "best_epoch": metric_logger.best_epoch,
                    "best_train_cindex": metric_logger.best_cindex,
                    "training_history": metric_logger.to_dict(),
                }
                fold_results.append(fold_result)

            # 清理GPU缓存
            if device.type == "cuda":
                torch.cuda.empty_cache()

            fold_models.append(model.cpu())  # 移回CPU
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"  [{fold_name}] 训练/测试失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

    cv_total_time = time.time() - cv_start_time

    # ---- 计算CV汇总 ----
    if not fold_results:
        logger.error("所有折均失败! 无法生成CV汇总")
        return {"error": "所有折训练失败", "fold_results": [], "cv_summary": {}}

    logger.info("")
    cv_summary = _compute_cv_summary(fold_results, config)

    # 添加时间信息
    cv_summary["total_time_minutes"] = round(cv_total_time / 60, 1)
    cv_summary["n_folds_completed"] = len(fold_results)
    cv_summary["n_folds_total"] = len(run_folds)
    cv_summary["device"] = str(device)

    # ---- 输出CV摘要 ----
    _log_cv_summary(logger, cv_summary)
    logger.info(f"  CV总耗时: {cv_total_time / 60:.1f} 分钟 "
                f"({cv_total_time / 3600:.2f} 小时)")

    # ---- 保存CV结果 ----
    results_dir = config.get_subdir("results")
    cv_package = {
        "cv_summary": cv_summary,
        "fold_results": fold_results,
        "system_info": sys_info,
        "config": config.to_dict() if hasattr(config, "to_dict") else {},
        "timestamp": datetime.now().isoformat(),
    }

    cv_results_path = results_dir / "cv_results.pkl"
    try:
        with open(str(cv_results_path), "wb") as f:
            pickle.dump(cv_package, f)
        logger.info(f"  CV结果已保存: {cv_results_path}")
    except Exception as e:
        logger.warning(f"  CV结果保存失败: {e}")

    # 同时保存一份可读的文本摘要
    try:
        txt_summary_path = results_dir / "cv_summary.txt"
        _save_text_summary(txt_summary_path, cv_summary, sys_info)
        logger.info(f"  文本摘要已保存: {txt_summary_path}")
    except Exception as e:
        logger.warning(f"  文本摘要保存失败: {e}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("  交叉验证完成!")
    logger.info("=" * 70)

    return cv_package


def _save_text_summary(path: Path, cv_summary: Dict, sys_info: Dict) -> None:
    """
    将CV摘要和系统信息保存为可读的文本文件。

    参数:
        path: 输出文件路径
        cv_summary: CV汇总字典
        sys_info: 系统信息字典
    """
    lines = []
    lines.append("=" * 70)
    lines.append("YuHou Multimodal Prognosis - CV Summary")
    lines.append("=" * 70)
    lines.append(f"Timestamp: {cv_summary.get('timestamp', 'N/A')}")
    lines.append(f"Platform: {sys_info.get('platform', 'N/A')}")

    if sys_info.get("gpu_details"):
        for gpu in sys_info["gpu_details"]:
            lines.append(f"GPU: {gpu['name']} ({gpu['total_memory_gb']:.1f} GB)")

    lines.append("-" * 70)
    lines.append(f"Folds: {cv_summary.get('n_folds', 0)}")

    ci_mean = cv_summary.get("cindex_mean", float("nan"))
    ci_std = cv_summary.get("cindex_std", float("nan"))
    if not np.isnan(ci_mean):
        lines.append(f"C-index: {ci_mean:.4f} +/- {ci_std:.4f}")
        folds_ci = cv_summary.get("cindex_all", [])
        lines.append(f"  Per fold: {[f'{c:.4f}' for c in folds_ci]}")

    tdauc_mean = cv_summary.get("tdauc_mean", float("nan"))
    tdauc_std = cv_summary.get("tdauc_std", float("nan"))
    if not np.isnan(tdauc_mean):
        lines.append(f"tdAUC: {tdauc_mean:.4f} +/- {tdauc_std:.4f}")

    hr_mean = cv_summary.get("hazard_ratio_mean", float("nan"))
    hr_std = cv_summary.get("hazard_ratio_std", float("nan"))
    if not np.isnan(hr_mean):
        lines.append(f"Hazard Ratio: {hr_mean:.4f} +/- {hr_std:.4f}")

    cfg = cv_summary.get("config_summary", {})
    lines.append("-" * 70)
    lines.append(f"Task: {cfg.get('task', 'N/A')}")
    lines.append(f"Fusion: {cfg.get('fusion_type', 'N/A')}")
    lines.append(f"LR: {cfg.get('learning_rate', 'N/A')}")
    lines.append(f"Total Epochs: {cfg.get('total_epochs', 'N/A')}")
    lines.append(f"AMP: {cfg.get('amp_enabled', 'N/A')}")

    lines.append("=" * 70)

    with open(str(path), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 命令行入口
# ============================================================

def main():
    """
    命令行入口函数。

    用法:
        python -m src.training.cv_runner --config config/default_config.yaml --gpu_id 0
    """
    parser = argparse.ArgumentParser(
        description="YuHou多模态预后预测 - 交叉验证训练",
    )
    parser.add_argument("--config", "-c", type=str, default="config/default_config.yaml",
                        help="YAML配置文件路径 (默认: config/default_config.yaml)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU卡号 (默认: 0)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="批次大小 (覆盖配置文件)")
    parser.add_argument("--fusion_type", type=str, default=None,
                        help="融合策略类型 (覆盖配置文件)")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="实验名称 (覆盖配置文件)")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率 (覆盖配置文件)")
    parser.add_argument("--fold_ids", type=int, nargs="+", default=None,
                        help="要运行的折ID列表，如: --fold_ids 1 2 3")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子 (覆盖配置文件)")

    args = parser.parse_args()

    # 加载配置
    from config.config_loader import load_config, print_config

    try:
        config = load_config(config_path=args.config)
    except Exception as e:
        print(f"配置加载失败: {e}")
        sys.exit(1)

    # 命令行覆盖
    if args.gpu_id is not None:
        config.training.gpu_id = args.gpu_id
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.fusion_type is not None:
        config.model.fusion.type = args.fusion_type
    if args.exp_name is not None:
        config.experiment.name = args.exp_name
    if args.lr is not None:
        config.training.optimizer.lr = args.lr
    if args.fold_ids is not None:
        config.cross_validation.fold_ids = args.fold_ids
    if args.seed is not None:
        config.model.seed = args.seed

    # 创建实验目录
    from config.config_loader import create_experiment_dirs
    create_experiment_dirs(config)

    # 打印配置
    print_config(config)

    # 启动交叉验证
    try:
        cv_package = run_cross_validation(config)
        return cv_package
    except Exception as e:
        print(f"\n交叉验证失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    """
    自测: 验证CV流程。
    需要数据划分文件存在才能完整运行，
    否则仅验证系统信息和数据加载接口。
    """
    print("=" * 70)
    print("交叉验证控制器 (cv_runner.py) 自测")
    print("=" * 70)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    # ---- 测试系统信息采集 ----
    print("\n[1/3] 采集系统信息...")
    try:
        sys_info = _get_system_info()
        print(f"  平台: {sys_info['platform']}")
        print(f"  CPU核心数: {sys_info['cpu_count_physical']}核(物理) / "
              f"{sys_info['cpu_count_logical']}核(逻辑)")
        print(f"  内存: {sys_info['ram_total_gb']:.1f} GB "
              f"(可用 {sys_info['ram_available_gb']:.1f} GB)")
        print(f"  GPU: {'可用' if sys_info['gpu_available'] else '不可用'}")
        if sys_info["gpu_available"] and sys_info["gpu_details"]:
            for gpu in sys_info["gpu_details"]:
                print(f"    GPU[{gpu['index']}]: {gpu['name']} "
                      f"({gpu['total_memory_gb']:.1f} GB)")
        print("  [系统信息采集] 通过")
    except Exception as e:
        print(f"  [系统信息采集] 失败: {e}")

    # ---- 测试CV汇总计算 ----
    print("\n[2/3] 测试CV汇总统计...")
    try:
        # 模拟3折结果
        mock_fold_results = [
            {
                "fold_id": 1,
                "test_cindex": 0.72,
                "test_logrank_pvalue": 0.003,
                "test_tdauc_mean": 0.70,
                "test_hazard_ratio": 2.5,
                "test_loss": 0.45,
                "best_train_cindex": 0.75,
                "best_epoch": 5,
                "test_tdauc_times": [12.0, 24.0],
                "test_tdauc_values": [0.73, 0.68],
                "test_binary_metrics": {"accuracy": 0.76, "f1_score": 0.74},
            },
            {
                "fold_id": 2,
                "test_cindex": 0.68,
                "test_logrank_pvalue": 0.01,
                "test_tdauc_mean": 0.67,
                "test_hazard_ratio": 2.1,
                "test_loss": 0.50,
                "best_train_cindex": 0.72,
                "best_epoch": 6,
                "test_tdauc_times": [12.0, 24.0],
                "test_tdauc_values": [0.69, 0.65],
                "test_binary_metrics": {"accuracy": 0.72, "f1_score": 0.70},
            },
            {
                "fold_id": 3,
                "test_cindex": 0.71,
                "test_logrank_pvalue": 0.005,
                "test_tdauc_mean": 0.69,
                "test_hazard_ratio": 2.3,
                "test_loss": 0.48,
                "best_train_cindex": 0.74,
                "best_epoch": 5,
                "test_tdauc_times": [12.0, 24.0],
                "test_tdauc_values": [0.72, 0.67],
                "test_binary_metrics": {"accuracy": 0.74, "f1_score": 0.72},
            },
        ]

        # 模拟config用于计算
        from dataclasses import dataclass, field
        from typing import List as _List

        @dataclass
        class MockConfig:
            class model:
                task = "surv"
                fusion = type("Fusion", (), {"type": "caugf"})()
            class training:
                batch_size = 4
                gradient_accumulation_steps = 2
                amp = type("AMP", (), {"enabled": True})()
                class optimizer:
                    lr = 0.0001
                class scheduler:
                    n_epochs = 8
                    n_epochs_decay = 4
                class loss:
                    reg_type = "none"
                seed = 2026
            class evaluation:
                eval_times = [12, 24, 36]
            class experiment:
                save_predictions = False

        cv_summary = _compute_cv_summary(mock_fold_results, MockConfig())
        print(f"  CV C-index: {cv_summary['cindex_mean']:.4f} +/- {cv_summary['cindex_std']:.4f}")
        print(f"  CV tdAUC:   {cv_summary['tdauc_mean']:.4f} +/- {cv_summary['tdauc_std']:.4f}")
        print(f"  [CV汇总计算] 通过")
    except Exception as e:
        print(f"  [CV汇总计算] 失败: {e}")

    # ---- 测试数据划分加载接口 ----
    print("\n[3/3] 测试数据加载接口...")
    try:
        from config.config_loader import load_config

        config = load_config()
        split_file = config.resolve_path(config.data.split_file)

        if split_file.exists():
            data_splits = _load_data_splits(config, logger)
            print(f"  数据划分文件: {split_file}")
            print(f"  可用折数: {len(data_splits)}")
            for fold_name in sorted(data_splits.keys()):
                fold_data = data_splits[fold_name]
                if isinstance(fold_data, dict):
                    keys = list(fold_data.keys())
                    train_n = len(fold_data.get("train", {}).get("x_path", [])) if "train" in fold_data else 0
                    test_n = len(fold_data.get("test", {}).get("x_path", [])) if "test" in fold_data else 0
                    print(f"    {fold_name}: train={train_n}, test={test_n}")
            print("  [数据加载] 通过")
        else:
            print(f"  数据划分文件不存在: {split_file}")
            print("  (跳过数据加载测试，请先运行split_builder生成划分)")
    except ImportError as e:
        print(f"  配置模块导入失败: {e}")
        print("  (跳过数据加载测试)")
    except Exception as e:
        print(f"  [数据加载] 失败: {e}")

    print("\n" + "=" * 70)
    print("cv_runner.py 自测完成!")
    print("=" * 70)
