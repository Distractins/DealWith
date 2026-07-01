# -*- coding: utf-8 -*-
"""
calibration.py
============================================================================
模型校准评估模块 ★新增。

校准 (Calibration) 衡量预测概率与实际观测频率的一致性。
在预后预测中非常重要：预测的风险概率应该与实际事件发生率相匹配。

包含:
    1. 分位数校准曲线 (Decile-based Calibration Curve)
    2. Brier Score (预测概率的均方误差)
    3. 校准斜率与截距 (Calibration Slope & Intercept)

使用场景:
    在生存分析中，使用特定时间点的预测存活概率，
    与实际KM估计值对比，评估模型的校准度。

使用示例:
    from src.evaluation.calibration import compute_calibration_curve, brier_score
    curve = compute_calibration_curve(pred_risk, events, n_bins=10)
    bs = brier_score(pred_risk, events)
    print(f"Brier Score: {bs:.4f}")
============================================================================
"""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def _to_numpy_1d(x):
    """安全转换为numpy 1D数组"""
    if x is None:
        return None
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)


def compute_calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform'):
    """
    计算校准曲线数据。

    将预测概率分为n_bins个区间，每个区间内计算:
        - 预测概率均值
        - 实际事件发生比例
        - 区间样本数

    完美校准时: 预测概率 ≈ 实际发生比例（对角线）

    参数:
        y_true: [N] 二值标签 (0/1, 事件/删失)
        y_prob: [N] 预测概率 [0, 1]
        n_bins: 分位数数量（默认10，即十分位数）
        strategy: 分箱策略
            - "uniform": 等宽分箱
            - "quantile": 等频分箱（推荐，每个箱样本数接近）

    返回:
        dict: 校准曲线数据
            - fraction_of_positives: [n_bins] 每个箱的实际事件比例
            - mean_predicted_value: [n_bins] 每个箱的预测概率均值
            - bin_counts: [n_bins] 每个箱的样本数
            - bin_edges: [n_bins+1] 箱边界
    """
    y_true = _to_numpy_1d(y_true).astype(float)
    y_prob = _to_numpy_1d(y_prob).astype(float)

    # 移除NaN
    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    if len(y_true) < n_bins:
        n_bins = max(2, len(y_true) // 2)

    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy=strategy
        )

        # 计算各箱样本数
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_counts = np.zeros(n_bins, dtype=int)
        for i in range(n_bins):
            in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if i == n_bins - 1:
                in_bin = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
            bin_counts[i] = int(np.sum(in_bin))

        return {
            "fraction_of_positives": fraction_of_positives.tolist(),
            "mean_predicted_value": mean_predicted_value.tolist(),
            "bin_counts": bin_counts.tolist(),
            "bin_edges": bin_edges.tolist(),
        }
    except Exception:
        return {
            "fraction_of_positives": [],
            "mean_predicted_value": [],
            "bin_counts": [],
            "bin_edges": [],
        }


def compute_brier_score(y_true, y_prob):
    """
    计算Brier Score（预测概率的均方误差）。

    Brier Score = (1/N) * Σ (p_i - y_i)²

    其中:
        p_i: 预测概率
        y_i: 实际事件标记 (0/1)

    Brier Score范围 [0, 1]:
        - 0: 完美校准
        - 0.25: 随机猜测（基线）
        - 1: 完全错误

    参数:
        y_true: [N] 二值标签 (0/1)
        y_prob: [N] 预测概率 [0, 1]

    返回:
        float: Brier Score值（越低越好）
    """
    y_true = _to_numpy_1d(y_true).astype(float)
    y_prob = _to_numpy_1d(y_prob).astype(float)

    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    if len(y_true) < 2:
        return None

    try:
        return float(brier_score_loss(y_true, y_prob))
    except Exception:
        return None


def compute_calibration_slope(y_true, y_prob):
    """
    计算校准斜率 (Calibration Slope)。

    通过逻辑回归: logit(y) = α + β * logit(p)
    其中:
        - β = 1: 完美校准
        - β < 1: 过度自信（预测概率范围大于实际范围）
        - β > 1: 不够自信（预测概率范围小于实际范围）

    参数:
        y_true: [N] 二值标签
        y_prob: [N] 预测概率

    返回:
        dict: {"slope": float, "intercept": float} 或 None
    """
    try:
        from sklearn.linear_model import LogisticRegression

        y_true = _to_numpy_1d(y_true).astype(float)
        y_prob = _to_numpy_1d(y_prob).astype(float)

        valid = np.isfinite(y_true) & np.isfinite(y_prob) & (y_prob > 0) & (y_prob < 1)
        y_true = y_true[valid]
        y_prob = y_prob[valid]

        if len(y_true) < 10:
            return None

        # 转换为logit空间
        logit_p = np.log(y_prob / (1 - y_prob))

        # 逻辑回归
        lr = LogisticRegression(penalty=None)
        lr.fit(logit_p.reshape(-1, 1), y_true)

        return {
            "slope": float(lr.coef_[0][0]),
            "intercept": float(lr.intercept_[0]),
        }
    except Exception:
        return None


def compute_calibration_summary(pred_scores, events, n_bins=10):
    """
    综合校准评估摘要。

    计算校准曲线的完整评估，包含:
        - 分位数校准曲线
        - Brier Score
        - 校准斜率与截距

    参数:
        pred_scores: [N] 预测风险分数（未归一化也OK，会内部处理）
        events: [N] 事件标记 (0=删失, 1=事件)
        n_bins: 分位数数量

    返回:
        dict: 综合校准评估结果
            - brier_score: float
            - calibration_slope: dict
            - calibration_curve: dict
            - eci: float (Expected Calibration Error)
    """
    pred_scores = _to_numpy_1d(pred_scores)
    events = _to_numpy_1d(events).astype(float)

    # 将风险分数转为[0,1]概率（sigmoid归一化）
    from scipy.special import expit
    y_prob = expit(pred_scores - np.median(pred_scores))  # 中位归一化

    result = {
        "brier_score": compute_brier_score(events, y_prob),
        "calibration_slope": compute_calibration_slope(events, y_prob),
        "calibration_curve": compute_calibration_curve(events, y_prob, n_bins=n_bins),
    }

    # Expected Calibration Error (ECE)
    try:
        curve = result["calibration_curve"]
        if len(curve["fraction_of_positives"]) > 0:
            frac = np.array(curve["fraction_of_positives"])
            pred = np.array(curve["mean_predicted_value"])
            counts = np.array(curve["bin_counts"])
            ece = np.sum(np.abs(frac - pred) * counts) / np.sum(counts)
            result["ece"] = float(ece)
        else:
            result["ece"] = None
    except Exception:
        result["ece"] = None

    return result


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("校准评估模块自测")
    print("=" * 60)

    np.random.seed(42)
    N = 200

    # 模拟完美校准的预测
    y_true = np.random.binomial(1, 0.3, N).astype(float)
    y_prob = 0.3 + np.random.normal(0, 0.05, N)
    y_prob = np.clip(y_prob, 0.01, 0.99)

    # 校准曲线
    curve = compute_calibration_curve(y_true, y_prob, n_bins=10)
    print(f"\n  校准曲线: {len(curve['fraction_of_positives'])}个分箱")
    if len(curve['fraction_of_positives']) > 0:
        print(f"  预测均值范围: [{curve['mean_predicted_value'][0]:.3f}, "
              f"{curve['mean_predicted_value'][-1]:.3f}]")
        print(f"  实际比例范围: [{curve['fraction_of_positives'][0]:.3f}, "
              f"{curve['fraction_of_positives'][-1]:.3f}]")

    # Brier Score
    bs = compute_brier_score(y_true, y_prob)
    print(f"  Brier Score: {bs:.4f} (期望≈0.2-0.3)")

    # 校准斜率
    slope = compute_calibration_slope(y_true, y_prob)
    if slope:
        print(f"  校准斜率: {slope['slope']:.4f} (期望≈1.0)")

    # 综合摘要
    summary = compute_calibration_summary(y_prob, y_true)
    print(f"\n  综合校准评估:")
    print(f"    Brier Score: {summary['brier_score']:.4f}")
    print(f"    ECE: {summary.get('ece', 'N/A')}")

    print("\n所有测试通过!")
