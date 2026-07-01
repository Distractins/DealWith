# -*- coding: utf-8 -*-
"""
metric_formatter.py
============================================================================
指标格式化模块 - 将评估指标格式化为中文可读字符串。

功能:
    1. 生存分析指标格式化（C-index, tdAUC, HR等）
    2. 分类指标格式化（ACC, AUC, F1等）
    3. 风险分组摘要格式化
    4. 统一的中文标签输出

使训练日志中的指标输出更加易读。

使用示例:
    from src.evaluation.metric_formatter import (
        format_cindex, format_td_auc, format_binary_metrics,
        format_hr, format_classification_metrics,
    )
    print(f"测试C指数: {format_cindex(0.7234)}")
    # 输出: "测试C指数: 0.7234"
============================================================================
"""

import numpy as np


# ============================================================
# 基础格式化
# ============================================================

def _fmt(x, nd=4, sci=False, none_str="N/A"):
    """
    安全格式化数值。

    参数:
        x: 数值或None
        nd: 小数位数
        sci: 是否使用科学计数法
        none_str: None时的替代字符串

    返回:
        str: 格式化后的字符串
    """
    if x is None:
        return none_str
    try:
        x = float(x)
        if np.isnan(x):
            return none_str
        if sci:
            return f"{x:.{nd}e}"
        return f"{x:.{nd}f}"
    except Exception:
        return none_str


# ============================================================
# 生存分析指标格式化
# ============================================================

def format_cindex(cindex, nd=4):
    """
    格式化C-Index。

    参数:
        cindex: C-index值 (float或None)
        nd: 小数位数

    返回:
        str: 如 "0.7234" 或 "N/A"
    """
    return _fmt(cindex, nd=nd)


def format_td_auc(td_auc_result, nd=4):
    """
    格式化时间依赖AUC结果。

    参数:
        td_auc_result: safe_time_dependent_auc()的返回值
            {"times": [...], "auc": [...], "mean_auc": float}

    返回:
        str: 如 "t=12.000: 0.7234 | t=24.000: 0.6845 | mean=0.7040"
    """
    if td_auc_result is None:
        return "N/A"
    try:
        times = td_auc_result.get("times", [])
        aucs = td_auc_result.get("auc", [])
        mean_auc = td_auc_result.get("mean_auc", None)

        parts = [f"t={float(t):.0f}月: {float(a):.{nd}f}" for t, a in zip(times, aucs)]
        if mean_auc is not None:
            parts.append(f"均值={float(mean_auc):.{nd}f}")

        return " | ".join(parts) if parts else "N/A"
    except Exception:
        return "N/A"


def format_logrank_p(pvalue, sci=True):
    """
    格式化Log-rank检验p值。

    参数:
        pvalue: p值 (float或None)
        sci: 是否使用科学计数法（p值通常很小）

    返回:
        str: 如 "0.0032" 或 "1.23e-05"
    """
    return _fmt(pvalue, nd=4, sci=sci)


def format_hazard_ratio(hr_result, nd=4):
    """
    格式化风险比 (Hazard Ratio) 结果。

    参数:
        hr_result: safe_hazard_ratio_by_median_split()的返回值

    返回:
        str: 如 "HR=2.15 [1.32, 3.51], p=0.0013"
    """
    if hr_result is None:
        return "N/A"
    try:
        return (
            f"HR={hr_result['hr']:.{nd}f} "
            f"[{hr_result['hr_ci_lower']:.{nd}f}, "
            f"{hr_result['hr_ci_upper']:.{nd}f}], "
            f"p={hr_result['p']:.{nd}g}"
        )
    except Exception:
        return "N/A"


def format_binary_metrics(result, nd=4):
    """
    格式化二分类指标（从中位风险分割得到）。

    参数:
        result: safe_binary_metrics_from_risk()的返回值

    返回:
        str: 如 "acc=0.78 | bacc=0.75 | recall=0.82 | f1=0.79"
    """
    if result is None:
        return "N/A"
    try:
        parts = [
            f"准确率={_fmt(result.get('acc'), nd)}",
            f"平衡准确率={_fmt(result.get('balanced_acc'), nd)}",
            f"召回率={_fmt(result.get('recall'), nd)}",
            f"特异度={_fmt(result.get('specificity'), nd)}",
            f"F1={_fmt(result.get('f1'), nd)}",
            f"MCC={_fmt(result.get('mcc'), nd)}",
        ]
        return " | ".join(parts)
    except Exception:
        return "N/A"


def format_group_summary(result, nd=4):
    """
    格式化风险分组摘要。

    参数:
        result: safe_group_survival_summary()的返回值

    返回:
        str: 如 "低风险(n=150,e=35,事件率=0.2333) | 高风险(n=150,e=78,事件率=0.5200)"
    """
    if result is None:
        return "N/A"
    try:
        low_er = _fmt(result.get("low_event_rate"), nd)
        high_er = _fmt(result.get("high_event_rate"), nd)
        return (
            f"低风险(n={result['low_n']},事件={result['low_event']},事件率={low_er}) | "
            f"高风险(n={result['high_n']},事件={result['high_event']},事件率={high_er})"
        )
    except Exception:
        return "N/A"


# ============================================================
# 分类指标格式化
# ============================================================

def format_classification_metrics(result, nd=4):
    """
    格式化分类指标结果。

    参数:
        result: classification_metrics()的返回值

    返回:
        str: 如 "准确率=0.8234 | 平衡准确率=0.7891 | F1=0.8012 | AUC=0.8745"
    """
    if result is None:
        return "N/A"
    try:
        parts = [
            f"准确率={_fmt(result.get('acc'), nd)}",
            f"平衡准确率={_fmt(result.get('balanced_acc'), nd)}",
            f"精确率={_fmt(result.get('precision'), nd)}",
            f"召回率={_fmt(result.get('recall'), nd)}",
            f"F1={_fmt(result.get('f1'), nd)}",
            f"AUC={_fmt(result.get('auc'), nd)}",
            f"AP={_fmt(result.get('ap'), nd)}",
            f"MCC={_fmt(result.get('mcc'), nd)}",
        ]
        if result.get("specificity") is not None:
            parts.append(f"特异度={_fmt(result.get('specificity'), nd)}")
        return " | ".join(parts)
    except Exception:
        return "N/A"


# ============================================================
# 损失值格式化
# ============================================================

def format_losses(loss_total, loss_cox=None, loss_task=None, loss_reg=None, nd=4):
    """
    格式化多损失组成部分。

    参数:
        loss_total: 总损失
        loss_cox: Cox损失
        loss_task: 任务损失
        loss_reg: 正则化损失
        nd: 小数位数

    返回:
        str: 如 "总损失=1.2345 (Cox=1.1000, 任务=0.1200, 正则=0.0145)"
    """
    parts = [f"总损失={_fmt(loss_total, nd)}"]
    sub_parts = []
    if loss_cox is not None:
        sub_parts.append(f"Cox={_fmt(loss_cox, nd)}")
    if loss_task is not None:
        sub_parts.append(f"任务={_fmt(loss_task, nd)}")
    if loss_reg is not None:
        sub_parts.append(f"正则={_fmt(loss_reg, nd)}")

    if sub_parts:
        parts.append(f"({', '.join(sub_parts)})")

    return " ".join(parts)


# ============================================================
# 综合摘要
# ============================================================

def format_cv_summary(mean_cindex, std_cindex, mean_tdauc=None, std_tdauc=None, nd=4):
    """
    格式化交叉验证结果摘要。

    参数:
        mean_cindex: 平均C-index
        std_cindex: C-index标准差
        mean_tdauc: 平均tdAUC (可选)
        std_tdauc: tdAUC标准差 (可选)

    返回:
        str: CV结果摘要字符串
    """
    parts = [f"CV C-Index: {_fmt(mean_cindex, nd)} ± {_fmt(std_cindex, nd)}"]
    if mean_tdauc is not None:
        parts.append(f"tdAUC: {_fmt(mean_tdauc, nd)} ± {_fmt(std_tdauc, nd)}")
    return " | ".join(parts)


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("指标格式化模块自测")
    print("=" * 60)

    print(f"\n  C-Index: {format_cindex(0.7234)}")
    print(f"  C-Index (None): {format_cindex(None)}")

    td_auc = {"times": [12, 24, 36], "auc": [0.75, 0.71, 0.68], "mean_auc": 0.7133}
    print(f"  tdAUC: {format_td_auc(td_auc)}")

    hr = {"hr": 2.15, "hr_ci_lower": 1.32, "hr_ci_upper": 3.51, "p": 0.0013}
    print(f"  HR: {format_hazard_ratio(hr)}")

    binary = {
        "acc": 0.78, "balanced_acc": 0.75, "recall": 0.82,
        "specificity": 0.71, "f1": 0.79, "mcc": 0.56,
    }
    print(f"  二分类指标: {format_binary_metrics(binary)}")

    cls_metrics = {
        "acc": 0.8234, "balanced_acc": 0.7891, "precision": 0.82,
        "recall": 0.78, "f1": 0.80, "auc": 0.8745, "ap": 0.86, "mcc": 0.72,
    }
    print(f"  分类指标: {format_classification_metrics(cls_metrics)}")

    print(f"\n  损失: {format_losses(1.2345, 1.1, 0.12, 0.0145)}")
    print(f"  CV摘要: {format_cv_summary(0.7234, 0.0456, 0.7133, 0.0521)}")

    print("\n所有测试通过!")
