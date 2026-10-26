# -*- coding: utf-8 -*-
"""scatter.py - 预测得分 vs 真实得分散点图 (含相关性标注)。"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from scipy.stats import spearmanr, pearsonr


def plot_pred_true_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "预测 vs 真实粘液分化得分",
    dpi: int = 200,
):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    sp = spearmanr(y_true, y_pred)[0]
    pr = pearsonr(y_true, y_pred)[0] if np.std(y_true) > 0 and np.std(y_pred) > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=30, alpha=0.6, edgecolor="k", linewidth=0.3)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y=x")
    ax.set_xlabel("真实得分")
    ax.set_ylabel("预测得分")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.text(0.05, 0.95, f"Spearman={sp:.3f}\nPearson={pr:.3f}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig
