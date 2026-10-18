# -*- coding: utf-8 -*-
"""
gene_eda.py
============================================================================
Histomolecular任务 - 目标基因表达探索性数据分析 (EDA)。

分析内容:
    1. 目标基因表达分布 (直方图 + 中位数阈值标注)
    2. 高/低表达组的临床特征差异统计
       - 数值型临床特征: t检验/Mann-Whitney U
       - 分类型临床特征: 卡方检验
    3. 高/低表达组生存KM曲线 (附属验证)

使用示例:
    from src.eda.gene_eda import GeneExpressionEDA
    eda = GeneExpressionEDA("data/COAD_all_dataset.csv")
    eda.run("REG4")
============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

from src.data_loading.gene_labels import resolve_gene_column, compute_group_threshold, build_binary_label


# 常见临床/协变量列 (数值 + 分类)
_NUMERIC_CLINICAL = ["age_at_diagnosis"]
_CATEGORICAL_CLINICAL = ["gender", "tumor_stage", "t", "n", "m", "histological_type"]


class GeneExpressionEDA:
    """目标基因表达EDA分析器。"""

    def __init__(self, csv_path: str, output_dir: str = "experiments/eda",
                 patient_id_col: str = "TCGA ID",
                 survtime_col: str = "Survival months",
                 censor_col: str = "event"):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.patient_id_col = patient_id_col
        self.survtime_col = survtime_col
        self.censor_col = censor_col
        self.df = None

    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.csv_path)
        print(f"[GeneEDA] 加载数据: {len(self.df)} 个样本")
        return self.df

    def run(self, target_gene: str, suffix: str = "_rnaseq",
            threshold_method: str = "median") -> Dict:
        """对单个目标基因执行完整EDA。"""
        if self.df is None:
            self.load()

        gene_col = resolve_gene_column(self.df, target_gene, suffix)
        if gene_col is None:
            print(f"[GeneEDA] 未找到目标基因 '{target_gene}' 的表达列，跳过")
            return {}

        vals = pd.to_numeric(self.df[gene_col], errors="coerce").values
        valid = np.isfinite(vals)
        threshold = compute_group_threshold(vals[valid], threshold_method)
        labels = build_binary_label(np.where(valid, vals, np.nan_to_num(vals, nan=threshold)), threshold)
        labels = pd.Series(labels, index=self.df.index)

        summary = {
            "gene": target_gene,
            "gene_col": gene_col,
            "n_valid": int(valid.sum()),
            "threshold": float(threshold),
            "mean": float(np.mean(vals[valid])),
            "std": float(np.std(vals[valid])),
            "min": float(np.min(vals[valid])),
            "max": float(np.max(vals[valid])),
            "n_high": int((labels == 1).sum()),
            "n_low": int((labels == 0).sum()),
        }

        print("\n" + "=" * 60)
        print(f"  目标基因 {target_gene} 表达EDA")
        print("=" * 60)
        for k, v in summary.items():
            print(f"  {k}: {v}")

        self._plot_expression_distribution(vals[valid], threshold, target_gene)
        clinical_stats = self._clinical_differential(labels)
        self._plot_km_by_group(labels, target_gene)

        summary["clinical_differential"] = clinical_stats
        return summary

    def _plot_expression_distribution(self, vals, threshold, target_gene):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(vals, bins=40, color="steelblue", edgecolor="white")
        axes[0].axvline(threshold, color="red", linestyle="--",
                        label=f"中位数阈值={threshold:.2f}")
        axes[0].set_xlabel("表达量")
        axes[0].set_ylabel("样本数")
        axes[0].set_title(f"{target_gene} 表达分布")
        axes[0].legend()

        labels = (vals > threshold).astype(int)
        axes[1].pie([int((labels == 0).sum()), int((labels == 1).sum())],
                    labels=["低表达", "高表达"], colors=["#2ecc71", "#e74c3c"],
                    autopct="%1.1f%%", explode=(0.03, 0.03))
        axes[1].set_title(f"{target_gene} 高低表达分组 (N={len(vals)})")

        plt.tight_layout()
        path = self.output_dir / f"gene_{target_gene}_expression_distribution.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  表达分布图已保存: {path}")

    def _clinical_differential(self, labels: pd.Series) -> Dict:
        """高/低表达组的临床特征差异统计。"""
        from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency

        df = self.df.copy()
        df["_group"] = labels.values
        stats = {}

        for col in _NUMERIC_CLINICAL:
            if col not in df.columns:
                continue
            numeric = pd.to_numeric(df[col], errors="coerce")
            high = numeric[df["_group"] == 1].dropna().values
            low = numeric[df["_group"] == 0].dropna().values
            if len(high) < 2 or len(low) < 2:
                continue
            try:
                t_p = ttest_ind(high, low, equal_var=False).pvalue
                mw_p = mannwhitneyu(high, low, alternative="two-sided").pvalue
                stats[col] = {
                    "high_mean": float(np.mean(high)), "low_mean": float(np.mean(low)),
                    "t_pvalue": float(t_p), "mannwhitney_pvalue": float(mw_p),
                }
            except Exception:
                pass

        for col in _CATEGORICAL_CLINICAL:
            if col not in df.columns:
                continue
            try:
                tab = pd.crosstab(df["_group"], df[col].astype(str))
                chi2, p, dof, _ = chi2_contingency(tab)
                stats[col] = {"chi2_pvalue": float(p), "contingency": tab.to_dict()}
            except Exception:
                pass

        print("\n  临床特征差异统计 (高 vs 低表达):")
        for col, s in stats.items():
            if "t_pvalue" in s:
                print(f"    {col}: 高表达均值={s['high_mean']:.2f}, "
                      f"低表达均值={s['low_mean']:.2f}, "
                      f"t检验p={s['t_pvalue']:.4f}, MannWhitney p={s['mannwhitney_pvalue']:.4f}")
            else:
                print(f"    {col}: 卡方p={s.get('chi2_pvalue', float('nan')):.4f}")

        return stats

    def _plot_km_by_group(self, labels, target_gene):
        """附属生存分析: 高/低表达组的KM曲线。"""
        if self.survtime_col not in self.df.columns or self.censor_col not in self.df.columns:
            return

        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test

        times = pd.to_numeric(self.df[self.survtime_col], errors="coerce").values
        events = pd.to_numeric(self.df[self.censor_col], errors="coerce").values
        valid = np.isfinite(times) & np.isfinite(events)
        times = times[valid]
        events = events[valid].astype(bool)
        grp = labels.values[valid]

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = {0: "#2ecc71", 1: "#e74c3c"}
        names = {0: "低表达组", 1: "高表达组"}
        p_value = None
        for gv in [0, 1]:
            mask = grp == gv
            if mask.sum() == 0:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(times[mask], events[mask], label=f"{names[gv]} (n={mask.sum()})")
            kmf.plot_survival_function(ax=ax, color=colors[gv], linewidth=2.5)

        if (grp == 0).sum() > 0 and (grp == 1).sum() > 0:
            try:
                lr = logrank_test(times[grp == 0], times[grp == 1],
                                  event_observed_A=events[grp == 0],
                                  event_observed_B=events[grp == 1])
                p_value = float(lr.p_value)
            except Exception:
                p_value = None

        ax.set_xlabel("时间 (月)", fontsize=12)
        ax.set_ylabel("总体生存率", fontsize=12)
        ax.set_title(f"{target_gene} 高/低表达组KM生存曲线", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        if p_value is not None:
            ax.text(0.05, 0.05, f"Log-rank p = {p_value:.4f}",
                    transform=ax.transAxes, fontsize=11,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.tight_layout()
        path = self.output_dir / f"gene_{target_gene}_km_by_expression.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  高低表达KM曲线已保存: {path}")


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/COAD_all_dataset.csv"
    gene = sys.argv[2] if len(sys.argv) > 2 else "REG4"

    if Path(csv_path).exists():
        eda = GeneExpressionEDA(csv_path)
        eda.run(gene)
    else:
        print(f"数据文件不存在: {csv_path}")
