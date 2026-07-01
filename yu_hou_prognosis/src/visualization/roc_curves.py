# -*- coding: utf-8 -*-
"""
roc_curves.py
============================================================================
时间依赖ROC曲线绘制模块。

功能:
    1. 绘制特定时间点的ROC曲线 (tdROC)
    2. 支持多个时间点对比
    3. 计算AUC并标注在图例中
    4. 中文标签支持

使用示例:
    from src.visualization.roc_curves import plot_time_dependent_roc
    plot_time_dependent_roc(td_auc_result, save_path="roc.png")
============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List

try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "ROC曲线",
    dpi: int = 300,
) -> plt.Figure:
    """
    绘制标准ROC曲线（二分类）。

    参数:
        y_true: [N] 二值标签 (0/1)
        y_score: [N] 预测概率分数
        save_path: 图片保存路径
        title: 图表标题
        dpi: 图片分辨率

    返回:
        fig: matplotlib Figure对象
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_val:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='随机分类器')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假阳性率 (1 - 特异度)', fontsize=12)
    ax.set_ylabel('真阳性率 (灵敏度)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_time_dependent_roc(
    td_auc_results: List[Dict],
    save_path: Optional[str] = None,
    title: str = "时间依赖ROC曲线",
    dpi: int = 300,
) -> plt.Figure:
    """
    绘制多个时间点的时间依赖ROC曲线对比。

    参数:
        td_auc_results: 每个时间点的tdAUC结果列表
            [{"time": 12, "fpr": [...], "tpr": [...], "auc": 0.75}, ...]
        save_path: 图片保存路径
        title: 图表标题
        dpi: 图片分辨率

    返回:
        fig: matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(td_auc_results)))

    for i, result in enumerate(td_auc_results):
        time_point = result.get("time", i)
        auc_val = result.get("auc", 0)
        fpr = result.get("fpr", [])
        tpr = result.get("tpr", [])

        if len(fpr) > 0 and len(tpr) > 0:
            ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                    label=f't={time_point}月 (AUC={auc_val:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='参考线')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假阳性率', fontsize=12)
    ax.set_ylabel('真阳性率', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    return fig


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("ROC曲线可视化模块自测")
    np.random.seed(42)

    y_true = np.random.binomial(1, 0.3, 200)
    y_score = np.random.rand(200)
    fig = plot_roc_curve(y_true, y_score)
    print("  ROC曲线绘制成功")

    # 模拟tdROC
    mock_td = []
    for t in [12, 24, 36]:
        mock_td.append({
            "time": t, "auc": 0.7 + t/100,
            "fpr": np.linspace(0, 1, 50),
            "tpr": np.sort(np.random.rand(50))[::-1],
        })
    fig2 = plot_time_dependent_roc(mock_td)
    print("  时间依赖ROC曲线绘制成功")
    print("测试通过!")
