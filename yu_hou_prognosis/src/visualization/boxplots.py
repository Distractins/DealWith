# -*- coding: utf-8 -*-
"""
boxplots.py
============================================================================
箱线图绘制模块。

功能:
    1. 按风险组/类别绘制生存时间箱线图
    2. 统计检验标注

使用示例:
    from src.visualization.boxplots import plot_risk_boxplot
    plot_risk_boxplot(risk_groups, times, save_path="boxplot.png")
============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict


try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


def plot_risk_boxplot(
    risk_groups: Dict[str, np.ndarray],
    times: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "风险分组生存时间箱线图",
    ylabel: str = "生存时间 (天)",
    dpi: int = 300,
) -> plt.Figure:
    """
    绘制不同风险组的生存时间箱线图。

    参数:
        risk_groups: {"低风险": np.array, "高风险": np.array} 风险组->生存时间
        times: [N] 完整生存时间（备用）
        save_path: 图片保存路径
        title: 图表标题
        ylabel: Y轴标签
        dpi: 图片分辨率

    返回:
        fig: matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    groups = list(risk_groups.keys())
    data = [risk_groups[g] for g in groups]

    # 箱线图
    bp = ax.boxplot(data, labels=groups, patch_artist=True,
                    widths=0.4, showfliers=True)

    # 着色
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    for i, (patch, color) in enumerate(zip(bp['boxes'], colors[:len(groups)])):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 添加抖动散点
    for i, d in enumerate(data):
        if len(d) > 0:
            jitter = np.random.normal(i + 1, 0.04, size=len(d))
            ax.scatter(jitter, d, alpha=0.3, s=15, color='black', zorder=3)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 组间比较
    if len(data) >= 2:
        try:
            from scipy.stats import mannwhitneyu
            stat, p = mannwhitneyu(data[0], data[1], alternative='two-sided')
            p_text = f"Mann-Whitney p = {p:.4f}"
            if p < 0.05:
                p_text += " *"
            ax.text(0.5, 0.95, p_text, transform=ax.transAxes, ha='center',
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.2",
                   facecolor="white", alpha=0.8))
        except ImportError:
            pass

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    return fig


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("箱线图可视化模块自测")
    np.random.seed(42)

    groups = {
        "低风险": np.random.exponential(scale=800, size=100),
        "高风险": np.random.exponential(scale=400, size=100),
    }
    times = np.concatenate(list(groups.values()))

    fig = plot_risk_boxplot(groups, times)
    print("  箱线图绘制成功")
    print("测试通过!")
