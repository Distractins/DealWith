# -*- coding: utf-8 -*-
"""
regression_metrics.py
============================================================================
连续表达量回归任务评估指标模块。

指标:
    - MSE  (均方误差)
    - MAE  (平均绝对误差)
    - RMSE (均方根误差)
    - Pearson 相关系数
    - Spearman 秩相关系数
    - R^2   (决定系数)

使用示例:
    from src.evaluation.regression_metrics import regression_metrics
    m = regression_metrics(pred, target)
    print(m["pearson"], m["rmse"])
============================================================================
"""

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def _to_numpy_1d(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def regression_metrics(pred, target):
    """
    计算连续表达量回归指标。

    参数:
        pred: [N, 1] 或 [N] 预测值
        target: [N] 真实表达量

    返回:
        dict: {"n", "mse", "mae", "rmse", "pearson", "spearman", "r2"}
    """
    pred = _to_numpy_1d(pred)
    target = _to_numpy_1d(target)

    valid = np.isfinite(pred) & np.isfinite(target)
    pred = pred[valid]
    target = target[valid]

    result = {"n": int(len(pred))}
    if len(pred) < 2:
        for k in ["mse", "mae", "rmse", "pearson", "spearman", "r2"]:
            result[k] = float("nan")
        return result

    try:
        result["mse"] = float(mean_squared_error(target, pred))
        result["mae"] = float(mean_absolute_error(target, pred))
        result["rmse"] = float(np.sqrt(result["mse"]))
        result["r2"] = float(r2_score(target, pred))
    except Exception:
        result["mse"] = result["mae"] = result["rmse"] = result["r2"] = float("nan")

    try:
        if np.std(pred) > 0 and np.std(target) > 0:
            result["pearson"] = float(pearsonr(target, pred)[0])
        else:
            result["pearson"] = float("nan")
    except Exception:
        result["pearson"] = float("nan")

    try:
        result["spearman"] = float(spearmanr(target, pred)[0])
    except Exception:
        result["spearman"] = float("nan")

    return result


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    target = np.linspace(0, 10, 50) + np.random.randn(50) * 0.3
    pred = target + np.random.randn(50) * 1.0

    m = regression_metrics(pred, target)
    print("回归指标:")
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("测试通过!")
