# -*- coding: utf-8 -*-
"""
confusion_matrix.py
============================================================================
混淆矩阵可视化模块。

使用示例:
    from src.visualization.confusion_matrix import plot_confusion_matrix
    plot_confusion_matrix(cm=[[10,2],[3,15]], save_path="cm.png")
============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List

try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "混淆矩阵",
    class_names: Optional[List[str]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    绘制混淆矩阵热力图。

    参数:
        cm: [C, C] 混淆矩阵
        save_path: 保存路径
        title: 标题
        class_names: 类别名 (默认 低表达/高表达)
        dpi: 分辨率
    """
    cm = np.asarray(cm)
    if class_names is None:
        class_names = ["低表达(0)", "高表达(1)"] if cm.shape[0] == 2 else \
            [f"类别{i}" for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel("预测标签", fontsize=12)
    ax.set_ylabel("真实标签", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)

    # 归一化注释 (召回率)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = cm.astype(float) / np.maximum(row_sums, 1e-8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i + 0.38, f"({norm[i, j]:.2f})",
                    ha="center", va="center", fontsize=9, alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


if __name__ == "__main__":
    cm = np.array([[42, 8], [6, 40]])
    fig = plot_confusion_matrix(cm)
    print("混淆矩阵绘制成功")
