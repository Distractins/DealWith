# -*- coding: utf-8 -*-
"""
regression_metrics.py - 回归评估指标 (无任何生存指标)。

指标: Spearman相关系数 / Pearson相关系数 / MAE / RMSE / R²。

注: 后续可利用模型预测得到的粘液分化得分做生存相关性附属分析,
    但不作为模型训练目标, 也不在此处实现。
"""

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _to_numpy(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def regression_metrics(pred, target) -> dict:
    """计算回归指标, 返回 {spearman, pearson, mae, rmse, r2, n}。"""
    pred = _to_numpy(pred)
    target = _to_numpy(target)
    valid = np.isfinite(pred) & np.isfinite(target)
    pred, target = pred[valid], target[valid]

    result = {"n": int(len(pred))}
    if len(pred) < 2:
        for k in ["spearman", "pearson", "mae", "rmse", "r2"]:
            result[k] = float("nan")
        return result

    try:
        result["spearman"] = float(spearmanr(target, pred)[0])
    except Exception:
        result["spearman"] = float("nan")
    try:
        result["pearson"] = float(pearsonr(target, pred)[0]) if np.std(pred) > 0 and np.std(target) > 0 else float("nan")
    except Exception:
        result["pearson"] = float("nan")

    result["mae"] = float(mean_absolute_error(target, pred))
    result["rmse"] = float(np.sqrt(mean_squared_error(target, pred)))
    try:
        result["r2"] = float(r2_score(target, pred))
    except Exception:
        result["r2"] = float("nan")
    return result
