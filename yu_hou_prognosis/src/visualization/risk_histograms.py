# -*- coding: utf-8 -*-
"""
risk_histograms.py
============================================================================
预测风险分布直方图绘制模块。

功能:
    1. 绘制预测风险分数分布直方图
    2. 按事件/删失分组着色显示
    3. 标注中位风险分割线

使用示例:
    from src.visualization.risk_histograms import plot_risk_histogram
    plot_risk_histogram(risk_scores, events, save_path="risk_hist.png")
============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


def plot_risk_histogram(
    risk_scores: np.ndarray,
    events: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "预测风险分布",
    n_bins: int = 30,
    dpi: int = 300,
) -> plt.Figure:
    """
    绘制预测风险分数的分布直方图。

    事件组（死亡）和删失组的分布分别用不同颜色显示，
    帮助评估模型的风险区分能力。

    期望输出:
        - 事件组的风险分数应整体偏高（分布右偏）
        - 删失组的风险分数应整体偏低（分布左偏）
        - 两组分布有明显分离说明模型区分能力强

    参数:
        risk_scores: [N] 预测风险分数
        events: [N] 事件标记 (1=死亡, 0=删失)
        save_path: 图片保存路径
        title: 图表标题
        n_bins: 直方图分箱数
        dpi: 图片分辨率

    返回:
        fig: matplotlib Figure对象
    """
    risk_scores = np.asarray(risk_scores).reshape(-1)
    events = np.asarray(events).reshape(-1).astype(int)

    # 分组
    event_mask = events == 1
    censor_mask = events == 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- 图1: 分组直方图 ----
    ax = axes[0]
    ax.hist(risk_scores[censor_mask], bins=n_bins, alpha=0.6,
            color='#3498db', edgecolor='white', label=f'删失 (n={censor_mask.sum()})',
            density=True)
    ax.hist(risk_scores[event_mask], bins=n_bins, alpha=0.6,
            color='#e74c3c', edgecolor='white', label=f'事件 (n={event_mask.sum()})',
            density=True)

    # 中位分割线
    median = np.median(risk_scores)
    ax.axvline(x=median, color='black', linestyle='--', linewidth=2,
               label=f'中位风险 = {median:.3f}')

    ax.set_xlabel('预测风险分数', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ---- 图2: KDE密度曲线 ----
    ax2 = axes[1]
    try:
        from scipy import stats
        for mask, color, label in [
            (event_mask, '#e74c3c', '事件'),
            (censor_mask, '#3498db', '删失'),
        ]:
            if mask.sum() > 1:
                kde = stats.gaussian_kde(risk_scores[mask])
                x_range = np.linspace(risk_scores.min(), risk_scores.max(), 200)
                ax2.plot(x_range, kde(x_range), color=color, linewidth=2, label=label)
                ax2.fill_between(x_range, 0, kde(x_range), color=color, alpha=0.1)
    except ImportError:
        pass

    ax2.axvline(x=median, color='black', linestyle='--', linewidth=2)
    ax2.set_xlabel('预测风险分数', fontsize=12)
    ax2.set_ylabel('密度', fontsize=12)
    ax2.set_title('风险分数密度曲线', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    return fig


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("风险直方图可视化模块自测")
    np.random.seed(42)
    N = 200

    risk = np.random.randn(N) * 2
    events = np.random.binomial(1, 0.3, N)
    fig = plot_risk_histogram(risk, events)
    print("  风险直方图绘制成功")
    print("测试通过!")
