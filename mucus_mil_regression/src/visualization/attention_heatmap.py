# -*- coding: utf-8 -*-
"""
attention_heatmap.py - patch注意力可视化。

将一个bag内各patch按注意力权重排序, 生成带权重标注的缩略图 (注意力热力图)。
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import List, Optional

try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


def plot_attention_heatmap(
    patch_paths: List[str],
    attn_weights: np.ndarray,
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    """
    参数:
        patch_paths: 一个bag的patch路径 (长度N)
        attn_weights: [N] 注意力权重 (softmax后)
        save_path: 保存路径
    """
    attn = np.asarray(attn_weights, dtype=np.float64).reshape(-1)
    order = np.argsort(attn)[::-1]
    N = len(order)

    cols = min(N, 6)
    rows = int(np.ceil(N / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.atleast_1d(axes).reshape(-1)

    for i, idx in enumerate(order):
        ax = axes[i]
        try:
            img = Image.open(patch_paths[idx]).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax.set_title(f"w={attn[idx]:.3f}", fontsize=8)
        ax.axis("off")

    for j in range(N, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Patch注意力权重 (从高到低)", fontsize=13)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig
