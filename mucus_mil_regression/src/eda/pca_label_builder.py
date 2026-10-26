# -*- coding: utf-8 -*-
"""
pca_label_builder.py - 标签预处理 (EDA阶段一次性生成)。

读取4个目标基因表达, 标准化后做PCA, 取第一主成分PC1作为
"粘液分化复合分子得分"(回归目标 y)。保存 label_csv 供数据集划分使用。

注: 本模块仅生成标签, 不参与训练; 生物学内容仅作为标签来源。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def build_pc1_labels(config) -> Path:
    """从表格读取4个基因, PCA -> PC1, 保存标签csv。返回标签csv路径。"""
    table_csv = config.resolve_path(config.data.table_csv)
    label_csv = config.resolve_path(config.data.label_csv)
    patient_id_col = config.data.patient_id_col
    genes = config.data.genes

    df = pd.read_csv(table_csv)
    print(f"[PCALabelBuilder] 表格样本: {len(df)}")

    # 过滤缺失基因表达
    gene_cols = [g for g in genes if g in df.columns]
    missing = [g for g in genes if g not in df.columns]
    if missing:
        print(f"  [警告] 缺少基因列: {missing}")
    if len(gene_cols) == 0:
        raise ValueError("无可用基因列")

    X = df[gene_cols].apply(pd.to_numeric, errors="coerce")
    valid = X.notna().all(axis=1)
    X = X[valid].values.astype(np.float64)
    ids = df.loc[valid, patient_id_col].astype(str).values
    print(f"  有效样本(4基因无缺失): {len(X)}")

    # 标准化 + PCA
    if config.data.pca.standardize:
        X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=config.data.pca.n_components)
    pc1 = pca.fit_transform(X)[:, 0].astype(np.float64)

    # 翻转符号使PC1与4基因均值正相关 (高分=更分化)
    if config.data.pca.flip_sign_positive:
        gene_mean = X.mean(axis=1)
        if np.corrcoef(pc1, gene_mean)[0, 1] < 0:
            pc1 = -pc1

    # 保存标签csv
    label_csv.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({patient_id_col: ids, "pc1_score": pc1})
    out.to_csv(label_csv, index=False)
    print(f"  标签已保存: {label_csv} ({len(out)}行)")

    # 打印记录
    print(f"  PCA解释方差比: {pca.explained_variance_ratio_}")
    loadings = pca.components_[0]
    print(f"  PC1基因载荷: {dict(zip(gene_cols, np.round(loadings, 4)))}")
    print(f"  PC1得分: 均值={pc1.mean():.3f}, 标准差={pc1.std():.3f}, "
          f"范围=[{pc1.min():.3f}, {pc1.max():.3f}]")

    # 可选保存载荷
    if config.data.pca.save_loadings:
        load_path = label_csv.with_suffix(".loadings.csv")
        pd.DataFrame({"gene": gene_cols, "loading": loadings}).to_csv(load_path, index=False)
        print(f"  载荷已保存: {load_path}")

    return label_csv
