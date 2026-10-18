# -*- coding: utf-8 -*-
"""
km_curves.py
============================================================================
Kaplan-Meier生存曲线绘制模块。

功能:
    1. 绘制按预测风险中位数分组的KM生存曲线
    2. 计算并标注log-rank检验p值
    3. 显示风险表 (risk table)
    4. 支持中文标签和自定义颜色

使用示例:
    from src.visualization.km_curves import plot_km_curve
    plot_km_curve(risk_scores, events, times, save_path="km_curve.png")
============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from typing import Optional, Tuple


# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


def plot_km_curve(
    risk_scores: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Kaplan-Meier生存曲线",
    xlabel: str = "时间 (月)",
    ylabel: str = "总体生存率 (OS)",
    color_high: str = "#e74c3c",
    color_low: str = "#2ecc71",
    show_ci: bool = True,
    show_risk_table: bool = False,
    dpi: int = 300,
) -> Tuple[plt.Figure, Optional[float]]:
    """
    绘制按中位风险分层的KM生存曲线。

    参数:
        risk_scores: [N] 预测风险分数（越大风险越高）
        events: [N] 事件标记 (1=死亡, 0=删失)
        times: [N] 生存时间（天）
        save_path: 图片保存路径
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        color_high: 高风险组颜色
        color_low: 低风险组颜色
        show_ci: 是否显示置信区间
        show_risk_table: 是否显示风险表
        dpi: 图片分辨率

    返回:
        (figure, p_value): matplotlib Figure对象和log-rank p值
    """
    # 输入验证
    risk_scores = np.asarray(risk_scores).reshape(-1)
    events = np.asarray(events).reshape(-1).astype(bool)
    times = np.asarray(times).reshape(-1).astype(float)

    valid = np.isfinite(risk_scores) & np.isfinite(times)
    risk_scores = risk_scores[valid]
    events = events[valid]
    times = times[valid]

    if len(risk_scores) < 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "样本量不足，无法绘制KM曲线", ha='center', va='center')
        return fig, None

    # 中位数分割
    median = np.median(risk_scores)
    high_mask = risk_scores > median
    low_mask = ~high_mask

    # Log-rank检验
    try:
        lr_result = logrank_test(
            times[low_mask], times[high_mask],
            event_observed_A=events[low_mask],
            event_observed_B=events[high_mask],
        )
        p_value = float(lr_result.p_value)
    except Exception:
        p_value = None

    # 拟合KM曲线
    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    fig, ax = plt.subplots(figsize=(10, 7))

    # 低风险组（绿色）
    if np.sum(low_mask) > 0:
        kmf_low.fit(
            times[low_mask],
            events[low_mask],
            label=f'低风险组 (n={np.sum(low_mask)})',
        )
        kmf_low.plot_survival_function(
            ax=ax, color=color_low, linewidth=2.5,
            show_censors=True, ci_show=show_ci,
        )

    # 高风险组（红色）
    if np.sum(high_mask) > 0:
        kmf_high.fit(
            times[high_mask],
            events[high_mask],
            label=f'高风险组 (n={np.sum(high_mask)})',
        )
        kmf_high.plot_survival_function(
            ax=ax, color=color_high, linewidth=2.5,
            show_censors=True, ci_show=show_ci,
        )

    # 标注
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Log-rank p值标注
    if p_value is not None:
        p_text = f"Log-rank p = {p_value:.4f}"
        if p_value < 0.05:
            p_text += " *"  # 显著
        if p_value < 0.01:
            p_text += "*"
        ax.text(0.05, 0.05, p_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    return fig, p_value


def plot_km_by_group(
    group: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Kaplan-Meier生存曲线(按预测分子表型分组)",
    xlabel: str = "时间 (月)",
    ylabel: str = "总体生存率 (OS)",
    group_names: Optional[dict] = None,
    colors: Optional[dict] = None,
    show_ci: bool = True,
    dpi: int = 300,
):
    """
    按预测分子表型分组绘制KM生存曲线 (附属生存分析)。

    参数:
        group: [N] 分组标签 (0/1 或任意整数)
        events: [N] 事件标记 (1=事件, 0=删失)
        times: [N] 生存时间
        save_path: 保存路径
        title/xlabel/ylabel: 标题与坐标轴
        group_names: {group_value: 名称}
        colors: {group_value: 颜色}
        show_ci: 是否显示置信区间
        dpi: 分辨率

    返回:
        (fig, p_value)
    """
    group = np.asarray(group).reshape(-1)
    events = np.asarray(events).reshape(-1).astype(bool)
    times = np.asarray(times).reshape(-1).astype(float)

    valid = np.isfinite(times) & np.isfinite(group.astype(float))
    group = group[valid]
    events = events[valid]
    times = times[valid]

    if group_names is None:
        group_names = {0: "低表达组", 1: "高表达组"}
    if colors is None:
        colors = {0: "#2ecc71", 1: "#e74c3c"}

    fig, ax = plt.subplots(figsize=(10, 7))
    uniq = sorted(np.unique(group))

    p_value = None
    if len(uniq) >= 2:
        g0 = group == uniq[0]
        g1 = group == uniq[1]
        try:
            lr = logrank_test(
                times[g0], times[g1],
                event_observed_A=events[g0],
                event_observed_B=events[g1],
            )
            p_value = float(lr.p_value)
        except Exception:
            p_value = None

    for gv in uniq:
        mask = group == gv
        if mask.sum() == 0:
            continue
        kmf = KaplanMeierFitter()
        label = group_names.get(gv, f"组{gv}")
        kmf.fit(times[mask], events[mask], label=f"{label} (n={mask.sum()})")
        kmf.plot_survival_function(
            ax=ax, color=colors.get(gv, None), linewidth=2.5,
            show_censors=True, ci_show=show_ci,
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if p_value is not None:
        p_text = f"Log-rank p = {p_value:.4f}"
        if p_value < 0.05:
            p_text += " *"
        if p_value < 0.01:
            p_text += "*"
        ax.text(0.05, 0.05, p_text, transform=ax.transAxes,
                fontsize=11, verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig, p_value


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("KM曲线可视化模块自测")
    np.random.seed(42)
    N = 200

    risk = np.random.randn(N) * 2
    events = np.random.binomial(1, 0.3, N).astype(bool)
    times = np.random.exponential(scale=1000, size=N)

    fig, p = plot_km_curve(risk, events, times)
    if p:
        print(f"  Log-rank p = {p:.4f}")

    group = (risk > np.median(risk)).astype(int)
    fig2, p2 = plot_km_by_group(group, events, times)
    if p2:
        print(f"  分组KM Log-rank p = {p2:.4f}")
    print("测试通过!")
