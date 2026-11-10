# -*- coding: utf-8 -*-
"""② 粘液基因集 EDA: 分布 / 相关性 / 组间 Mann-Whitney / 队列内独立 PCA。"""
import os
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from loader import COLOR_NONMUC, COLOR_MUC


def run_gene_eda(df, genes, core_genes, cohort, gene_unit, out_fig, out_tab, plt):
    sub = df.dropna(subset=genes).reset_index(drop=True)
    n = len(sub)
    X = sub[genes].values.astype(float)
    y_muc = sub["is_mucinous"].astype(int).values

    _plot_histograms(sub, genes, cohort, gene_unit, out_fig, plt)
    _plot_boxplots(sub, genes, cohort, gene_unit, out_fig, plt)

    corr = sub[genes].corr(method="spearman")
    _plot_corr_heatmap(corr, cohort, out_fig, plt)
    corr.to_csv(os.path.join(out_tab, "02_gene_corr_matrix.csv"), encoding="utf-8-sig")

    mw = {g: st.mannwhitneyu(sub.loc[y_muc == 1, g], sub.loc[y_muc == 0, g],
                             alternative="two-sided").pvalue for g in genes}
    pd.DataFrame([{"基因": g, "MannWhitney_p": p} for g, p in mw.items()]).to_csv(
        os.path.join(out_tab, "02_gene_mannwhitney.csv"), index=False, encoding="utf-8-sig")

    pca = _run_pca(X, genes)
    _plot_pc1_boxplot(pca["pc1"], y_muc, cohort, out_fig, plt)

    has_both = (y_muc.sum() > 0) and ((1 - y_muc).sum() > 0)
    auc = roc_auc_score(y_muc, pca["pc1"]) if has_both else np.nan
    mw_pc1 = st.mannwhitneyu(pca["pc1"][y_muc == 1], pca["pc1"][y_muc == 0],
                             alternative="two-sided").pvalue if has_both else np.nan

    pd.DataFrame([{"指标": "PC1解释方差比", "值": pca["evr"]},
                  {"指标": "PC1 vs 粘液 AUC", "值": auc},
                  {"指标": "PC1 vs 粘液 MannWhitney_p", "值": mw_pc1}] +
                 [{"指标": f"{g}载荷", "值": l} for g, l in pca["loadings"].items()]).to_csv(
        os.path.join(out_tab, "02_pca_summary.csv"), index=False, encoding="utf-8-sig")

    # 核心杯状细胞基因子集: 若与全基因集不同, 评估基因集选择对信号的影响
    core = [g for g in core_genes if g in genes]
    core_auc = core_evr = np.nan
    if core and set(core) != set(genes):
        pca_core = _run_pca(sub[core].values.astype(float), core)
        core_evr = pca_core["evr"]
        core_auc = roc_auc_score(y_muc, pca_core["pc1"]) if has_both else np.nan

    return {
        "n_complete": n,
        "mw": mw,
        "corr": corr,
        "pca": {"evr": pca["evr"], "loadings": pca["loadings"],
                "pc1": pca["pc1"], "auc": auc, "mw_p": mw_pc1},
        "core_pca": {"genes": core, "auc": core_auc, "evr": core_evr},
    }


def _plot_histograms(sub, genes, cohort, gene_unit, out_fig, plt):
    ncol, nrow = 3, int(np.ceil(len(genes) / 3))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.2, nrow * 2.6), squeeze=False)
    for i, g in enumerate(genes):
        ax = axes[i // ncol][i % ncol]
        ax.hist(sub[g], bins=30, color=COLOR_NONMUC, alpha=0.85)
        ax.set_title(g.replace("_rnaseq", ""))
        ax.set_xlabel(gene_unit)
        ax.set_ylabel("样本数")
    for j in range(len(genes), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{cohort} 粘液基因表达分布", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_fig, "02_gene_histograms.png"), bbox_inches="tight")
    plt.close(fig)


def _plot_boxplots(sub, genes, cohort, gene_unit, out_fig, plt):
    ncol, nrow = 3, int(np.ceil(len(genes) / 3))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.2, nrow * 2.6), squeeze=False)
    data_muc = sub[sub["is_mucinous"]]
    data_non = sub[~sub["is_mucinous"]]
    for i, g in enumerate(genes):
        ax = axes[i // ncol][i % ncol]
        bp = ax.boxplot(
            [data_non[g].dropna(), data_muc[g].dropna()],
            labels=[f"非粘液\nn={len(data_non)}", f"粘液\nn={len(data_muc)}"],
            patch_artist=True, widths=0.5)
        for patch, c in zip(bp["boxes"], [COLOR_NONMUC, COLOR_MUC]):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
        ax.set_title(g.replace("_rnaseq", ""))
        ax.set_ylabel(gene_unit)
    for j in range(len(genes), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{cohort} 粘液 vs 非粘液 基因表达", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_fig, "02_gene_boxplots.png"), bbox_inches="tight")
    plt.close(fig)


def _plot_corr_heatmap(corr, cohort, out_fig, plt):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                center=0, square=True, cbar_kws={"label": "Spearman ρ"},
                ax=ax, annot_kws={"size": 8})
    ax.set_title(f"{cohort} 粘液基因两两相关 (Spearman)")
    ax.set_xticklabels([c.replace("_rnaseq", "") for c in corr.columns], rotation=45)
    ax.set_yticklabels([c.replace("_rnaseq", "") for c in corr.columns], rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_fig, "02_gene_corr_heatmap.png"), bbox_inches="tight")
    plt.close(fig)


def _run_pca(X, genes):
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(Z)[:, 0]
    loadings = dict(zip(genes, pca.components_[0]))
    if np.corrcoef(pc1, Z.mean(axis=1))[0, 1] < 0:
        pc1 = -pc1
        loadings = {g: -v for g, v in loadings.items()}
    return {"evr": float(pca.explained_variance_ratio_[0]), "loadings": loadings, "pc1": pc1}


def _plot_pc1_boxplot(pc1, y_muc, cohort, out_fig, plt):
    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot([pc1[y_muc == 0], pc1[y_muc == 1]],
                    labels=[f"非粘液\nn={(y_muc == 0).sum()}", f"粘液\nn={(y_muc == 1).sum()}"],
                    patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], [COLOR_NONMUC, COLOR_MUC]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
    ax.set_ylabel("粘液分化 PC1 (标准化)")
    ax.set_title(f"{cohort} PC1 在粘液/非粘液分组")
    fig.tight_layout()
    fig.savefig(os.path.join(out_fig, "02_pca_pc1_boxplot.png"), bbox_inches="tight")
    plt.close(fig)
