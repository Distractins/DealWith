# -*- coding: utf-8 -*-
"""
cv_runner.py - 交叉验证主控 (MIL回归)。

流程:
    1. 读取EDA生成的PC1标签csv + 临床表格
    2. 建立 短TCGA ID -> patch路径 映射 (仅image/multimodal需要)
    3. 病人级5折拆分 (按y分箱分层, 同一病人patch不跨折)
    4. 每折: 训练集切val, 拟合临床编码器, 训练+测试
    5. 汇总 Spearman/Pearson/MAE/RMSE/R², 保存每病人预测CSV + 折级CSV
"""

import os
import time
import pickle
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.data_loading.upstream_reader import UpstreamPatchReader
from src.data_loading.clinical_features import ClinicalEncoder
from src.data_loading.split_builder import build_regression_splits, save_splits
from src.training.trainer import train, test, build_model


# ============================================================
# 数据构建
# ============================================================

def _load_labels(config) -> pd.DataFrame:
    label_csv = config.resolve_path(config.data.label_csv)
    if not label_csv.exists():
        raise FileNotFoundError(
            f"标签csv不存在: {label_csv}\n请先运行: python main.py --mode eda (生成PCA PC1标签)")
    return pd.read_csv(label_csv)


def _load_table(config) -> pd.DataFrame:
    return pd.read_csv(config.resolve_path(config.data.table_csv))


def _build_patient_df(config, labels: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """合并 标签 + 临床特征 -> 病人级DataFrame (patient_id, y, 临床列...)。"""
    pid = config.data.patient_id_col
    labels[pid] = labels[pid].astype(str).str.strip()
    table[pid] = table[pid].astype(str).str.strip()

    df = labels.merge(table[[pid] + list(config.data.clinical_features)], on=pid, how="left")
    df["y"] = pd.to_numeric(df["pc1_score"], errors="coerce")
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    return df


def _build_fold_data(config, patient_df, patch_map, fold_data, clinical_encoder, logger) -> Dict:
    """为一个fold构建 train/val/test 数据字典。"""
    pid_col = config.data.patient_id_col
    need_patches = config.model.model_type in ("image_mil", "multimodal")

    def _collect(patient_ids):
        records = []
        for p in patient_ids:
            row = patient_df[patient_df[pid_col] == p]
            if len(row) == 0:
                continue
            if need_patches and p not in patch_map:
                continue
            records.append({"pid": p, "y": float(row["y"].iloc[0])})
        return records

    train_rec = _collect(fold_data["train"]["patient_ids"])
    test_rec = _collect(fold_data["test"]["patient_ids"])

    # 训练集切val
    from sklearn.model_selection import train_test_split
    val_ratio = config.cross_validation.val_ratio
    if 0 < val_ratio < 1 and len(train_rec) >= 5:
        y_all = np.array([r["y"] for r in train_rec])
        try:
            bins = pd.qcut(pd.Series(y_all), q=min(5, len(y_all)), labels=False, duplicates="drop").fillna(0).astype(int).values
            train_rec, val_rec = train_test_split(train_rec, test_size=val_ratio,
                                                  random_state=config.project.seed, stratify=bins)
        except Exception:
            train_rec, val_rec = train_test_split(train_rec, test_size=val_ratio,
                                                  random_state=config.project.seed)
    else:
        val_rec = []

    # 拟合临床编码器 (仅训练子集)
    train_ids = [r["pid"] for r in train_rec]
    sub = patient_df[patient_df[pid_col].isin(train_ids)].copy()
    clinical_encoder.fit(sub)

    def _build(records):
        if not records:
            return {"x_path": [], "x_clinical": np.zeros((0, clinical_encoder.output_dim), dtype=np.float32),
                    "y": np.array([], dtype=np.float32), "patient_ids": []}
        ids = [r["pid"] for r in records]
        sub_df = patient_df[patient_df[pid_col].isin(ids)].copy()
        # 按 ids 顺序对齐 (isin 不保持顺序, 需显式reindex)
        sub_df = sub_df.set_index(pid_col).loc[ids].reset_index()
        clinical = clinical_encoder.transform(sub_df)
        x_path = [patch_map.get(p, []) for p in ids] if need_patches else [[] for _ in ids]
        y = np.array([r["y"] for r in records], dtype=np.float32)
        return {"x_path": x_path, "x_clinical": clinical, "y": y, "patient_ids": ids}

    return {"train": _build(train_rec), "val": _build(val_rec), "test": _build(test_rec)}


def _load_or_build_splits(config, patient_ids, y) -> Dict:
    split_file = config.resolve_path(config.data.split_file)
    if split_file.exists():
        with open(split_file, "rb") as f:
            raw = pickle.load(f)
        if isinstance(raw, dict) and "cv_splits" in raw:
            return raw["cv_splits"]
    splits = build_regression_splits(patient_ids, y, config.cross_validation.n_folds,
                                     config.project.seed)
    save_splits(splits, split_file)
    return splits


def _build_all_data(config, logger) -> Dict:
    labels = _load_labels(config)
    table = _load_table(config)
    patient_df = _build_patient_df(config, labels, table)
    logger.info(f"  有效病人(有PC1标签): {len(patient_df)}")

    need_patches = config.model.model_type in ("image_mil", "multimodal")
    patch_map = {}
    if need_patches:
        reader = UpstreamPatchReader(config)
        patch_map = reader.build_short_to_patches()
        # 过滤出有patch的病人
        before = len(patient_df)
        patient_df = patient_df[patient_df[config.data.patient_id_col].isin(patch_map.keys())].reset_index(drop=True)
        logger.info(f"  有patch病人: {len(patient_df)}/{before}")

    pid_col = config.data.patient_id_col
    patient_ids = patient_df[pid_col].tolist()
    y = patient_df["y"].values.astype(np.float64)

    splits = _load_or_build_splits(config, patient_ids, y)
    clinical_encoder = ClinicalEncoder(config.data.clinical_features)

    folds = {}
    for fold_id, fold_data in sorted(splits.items()):
        folds[f"fold_{fold_id}"] = _build_fold_data(config, patient_df, patch_map,
                                                     fold_data, clinical_encoder, logger)

    config.model.clinical_input_dim = clinical_encoder.output_dim
    return folds


# ============================================================
# 结果汇总 + 保存
# ============================================================

def _safe_mean_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def _save_results(config, fold_results, logger):
    results_dir = config.get_subdir("results")
    metrics_list = config.evaluation.metrics

    rows = []
    patient_rows = []
    for fr in fold_results:
        r = {"fold": fr["fold_id"]}
        for m in metrics_list:
            r[m] = fr["metrics"].get(m, float("nan"))
        rows.append(r)
        for pid, yt, yp in zip(fr["patient_ids"], fr["targets"], fr["preds"]):
            patient_rows.append({"fold": fr["fold_id"], "patient_id": pid,
                                 "y_true": float(yt), "y_pred": float(yp)})

    df_fold = pd.DataFrame(rows)
    # 均值±标准差行
    mean_row = {"fold": "mean"}
    std_row = {"fold": "std"}
    for m in metrics_list:
        mean, std = _safe_mean_std([r[m] for r in rows])
        mean_row[m] = mean
        std_row[m] = std
    df_fold = pd.concat([df_fold, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    name = config.experiment.name
    df_fold.to_csv(results_dir / f"{name}_fold_metrics.csv", index=False, float_format="%.4f")
    pd.DataFrame(patient_rows).to_csv(results_dir / f"{name}_patient_predictions.csv",
                                      index=False, float_format="%.4f")
    logger.info(f"  折级指标CSV: {results_dir / f'{name}_fold_metrics.csv'}")
    logger.info(f"  病人预测CSV: {results_dir / f'{name}_patient_predictions.csv'}")
    return df_fold


# ============================================================
# 主流程
# ============================================================

def run_cross_validation(config) -> Dict:
    logger = setup_logger(log_file=str(config.get_subdir("logs") / "cv_train.log"),
                          level=config.logging.level, language=config.logging.language)
    logger.info("=" * 70)
    logger.info(f"  MucusMIL Regression - 交叉验证 (model={config.model.model_type}, "
                f"backbone={config.model.backbone}, algorithm={config.upstream.algorithm_name})")
    logger.info("=" * 70)

    if config.training.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.training.device)
    if device.type == "cuda":
        torch.cuda.set_device(min(config.training.gpu_id, torch.cuda.device_count() - 1))
        device = torch.device(f"cuda:{config.training.gpu_id}")

    set_seed(config.project.seed)

    logger.info("加载数据划分...")
    data_splits = _build_all_data(config, logger)

    fold_ids = config.cross_validation.fold_ids
    run_folds = [f"fold_{f}" for f in fold_ids if f"fold_{f}" in data_splits]
    if not run_folds:
        run_folds = list(data_splits.keys())[: config.cross_validation.n_folds]
    logger.info(f"  实际运行折: {run_folds}")

    fold_results = []
    for fold_name in run_folds:
        fold_id = int(fold_name.split("_")[-1])
        logger.info("")
        logger.info(f"  >>> {fold_name} 训练开始 <<<")
        fold_data = data_splits[fold_name]

        model, metric_logger = train(config, fold_data, device, fold_id)
        test_result = test(config, model, fold_data, "test", device)

        fold_results.append({
            "fold_id": fold_id,
            "metrics": test_result["metrics"],
            "preds": test_result["preds"],
            "targets": test_result["targets"],
            "attn": test_result["attn"],
            "patient_ids": test_result["patient_ids"],
        })
        _log_metrics(logger, fold_id, test_result["metrics"])

        if device.type == "cuda":
            torch.cuda.empty_cache()
        del model

    logger.info("")
    logger.info("=" * 70)
    logger.info("  交叉验证最终结果 (五折汇总)")
    logger.info("=" * 70)
    for m in config.evaluation.metrics:
        mean, std = _safe_mean_std([fr["metrics"].get(m) for fr in fold_results])
        if not np.isnan(mean):
            logger.info(f"  {m}: {mean:.4f} +/- {std:.4f}")

    df_fold = _save_results(config, fold_results, logger)

    # 可视化: 散点图 + 注意力热力图
    if config.visualization.enabled and fold_results:
        _plot_scatter(config, fold_results)
        _plot_attention(config, data_splits, fold_results, logger)

    return {"fold_results": fold_results, "fold_metrics_df": df_fold}


def _log_metrics(logger, fold_id, metrics):
    parts = [f"  [Fold {fold_id} TEST]"]
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
    logger.info(" | ".join(parts))


def _plot_scatter(config, fold_results):
    from src.visualization.scatter import plot_pred_true_scatter
    all_true = np.concatenate([fr["targets"] for fr in fold_results])
    all_pred = np.concatenate([fr["preds"] for fr in fold_results])
    fig_dir = config.get_subdir("figures")
    plot_pred_true_scatter(all_true, all_pred,
                           save_path=str(fig_dir / f"{config.experiment.name}_scatter.png"))


def _plot_attention(config, data_splits, fold_results, logger):
    """对测试集第一个有patch的病人画注意力热力图 (仅image/multimodal)。"""
    if config.model.model_type == "clinical_mlp":
        return
    from src.visualization.attention_heatmap import plot_attention_heatmap
    fr = fold_results[0]
    if fr["attn"] is None or len(fr["attn"]) == 0:
        return
    pids = fr["patient_ids"]
    if not pids:
        return
    # 取平均注意力最大的病人
    best_idx = int(np.argmax(fr["attn"].sum(axis=1)))
    pid = pids[best_idx]
    patch_paths = data_splits[f"fold_{fr['fold_id']}"]["test"]["x_path"][best_idx]
    fig_dir = config.get_subdir("figures")
    try:
        plot_attention_heatmap(patch_paths, fr["attn"][best_idx],
                               save_path=str(fig_dir / f"{config.experiment.name}_attention_{pid}.png"))
    except Exception as e:
        logger.warning(f"  注意力热力图绘制失败: {e}")
