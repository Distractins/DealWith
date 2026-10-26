# -*- coding: utf-8 -*-
"""ablation_plots.py - 消融对比柱状图 (模型类型/融合方式/采样算法)。"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional


def plot_ablation_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    save_path: Optional[str] = None,
    title: str = "消融对比",
    dpi: int = 200,
):
    """df: 每行一个配置, 含 x_col(配置名) 与 y_col(指标值)。"""
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 1.2), 5))
    x = df[x_col].astype(str).values
    y = df[y_col].astype(float).values
    bars = ax.bar(x, y, color="#4c72b0", edgecolor="k", linewidth=0.5)
    ax.set_ylabel(y_col)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticklabels(x, rotation=30, ha="right", fontsize=9)
    for b, v in zip(bars, y):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig
