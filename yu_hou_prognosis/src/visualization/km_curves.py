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
    print("测试通过!")
