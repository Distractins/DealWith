# -*- coding: utf-8 -*-
"""
calibration_plots.py
============================================================================
校准曲线绘制模块 ★新增。

功能:
    1. 绘制校准曲线（预测概率 vs 实际事件发生率）
    2. 标注Brier Score和ECE
    3. 完美校准对角线参考

使用示例:
    from src.visualization.calibration_plots import plot_calibration_curve
    plot_calibration_curve(curve_data, brier_score, ece, save_path="calibration.png")
============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict


try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


def plot_calibration_curve(
    calibration_curve_data: Dict,
    brier_score: Optional[float] = None,
    ece: Optional[float] = None,
    save_path: Optional[str] = None,
    title: str = "校准曲线",
    dpi: int = 300,
) -> plt.Figure:
    """
    绘制校准曲线 (Calibration Curve)。

    完美校准时，预测概率 = 实际事件发生率（重合于对角线）。

    参数:
        calibration_curve_data: compute_calibration_curve()的返回值
            {
                "fraction_of_positives": [0.1, 0.2, ...],  # 实际事件比例
                "mean_predicted_value": [0.12, 0.18, ...],  # 预测概率均值
                "bin_counts": [20, 18, ...],                # 每箱样本数
            }
        brier_score: Brier Score值 (可选)
        ece: Expected Calibration Error (可选)
        save_path: 图片保存路径
        title: 图表标题
        dpi: 图片分辨率

    返回:
        fig: matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    frac_pos = np.array(calibration_curve_data.get("fraction_of_positives", []))
    mean_pred = np.array(calibration_curve_data.get("mean_predicted_value", []))
    bin_counts = np.array(calibration_curve_data.get("bin_counts", []))

    if len(frac_pos) > 0 and len(mean_pred) > 0:
        # 校准曲线
        ax.plot(mean_pred, frac_pos, 's-', color='#2e86c1',
                linewidth=2, markersize=8, label='校准曲线')

        # 各箱样本数标记
        for i, (x, y) in enumerate(zip(mean_pred, frac_pos)):
            if i < len(bin_counts):
                ax.annotate(f'n={int(bin_counts[i])}',
                           (x, y), textcoords="offset points",
                           xytext=(-10, 10), fontsize=8, alpha=0.7)

        # 置信区间（各箱误差条）
        if len(bin_counts) > 0 and len(frac_pos) > 0:
            std_err = np.sqrt(frac_pos * (1 - frac_pos) / np.maximum(bin_counts, 1))
            ax.fill_between(mean_pred, frac_pos - 1.96 * std_err,
                          frac_pos + 1.96 * std_err, alpha=0.2,
                          color='#2e86c1', label='95%置信区间')

    # 完美校准对角线
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='完美校准')

    # 标注Brier Score和ECE
    stats_text = ""
    if brier_score is not None:
        stats_text += f"Brier Score = {brier_score:.4f}"
    if ece is not None:
        stats_text += f"\nECE = {ece:.4f}"
    if stats_text:
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('预测概率', fontsize=12)
    ax.set_ylabel('实际事件发生率', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    return fig


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("校准曲线可视化模块自测")
    np.random.seed(42)

    curve_data = {
        "fraction_of_positives": [0.05, 0.15, 0.28, 0.35, 0.52, 0.68, 0.75, 0.85, 0.92, 0.98],
        "mean_predicted_value": [0.05, 0.13, 0.22, 0.38, 0.48, 0.65, 0.72, 0.82, 0.91, 0.98],
        "bin_counts": [25, 22, 20, 18, 20, 21, 23, 19, 17, 15],
    }

    fig = plot_calibration_curve(curve_data, brier_score=0.085, ece=0.042)
    print("  校准曲线绘制成功")
    print("测试通过!")
