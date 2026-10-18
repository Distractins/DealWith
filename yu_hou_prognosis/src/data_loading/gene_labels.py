# -*- coding: utf-8 -*-
"""
gene_labels.py
============================================================================
目标基因表达标签构建模块 (Histomolecular任务核心)。

职责:
    1. 从表格CSV中定位目标基因的表达列 (自动解析 "REG4" -> "REG4_rnaseq")
    2. 依据表达量做高/低表达二分类标签 (中位数/均值/分位数/自定义阈值)
    3. 支持连续表达量回归标签 (不做离散化)

分组阈值设计 (避免标签泄露):
    中位数阈值应基于训练集表达量计算，再应用到训练集与测试集。
    本模块提供 `compute_group_threshold(values, method)` 纯函数，
    由 cv_runner 在每个 fold 的训练集上调用。

使用示例:
    from src.data_loading.gene_labels import resolve_gene_column, compute_group_threshold
    col = resolve_gene_column(df, "REG4", suffix="_rnaseq")
    th = compute_group_threshold(df[col].values, "median")
    y = (df[col].values > th).astype(int)
============================================================================
"""

import numpy as np
import pandas as pd
from typing import Optional, List


# ============================================================
# 目标基因列解析
# ============================================================

def resolve_gene_column(df: pd.DataFrame, target_gene: str,
                        suffix: str = "_rnaseq") -> Optional[str]:
    """
    在DataFrame中定位目标基因的表达列名。

    匹配优先级:
        1. 精确匹配 target_gene (大小写不敏感)
        2. 匹配 target_gene + suffix (大小写不敏感), 如 "REG4_rnaseq"

    参数:
        df: 表格DataFrame
        target_gene: 目标基因名, 如 "REG4" 或 "REG4_rnaseq"
        suffix: RNA-seq表达列后缀 (默认 "_rnaseq")

    返回:
        str: 实际列名; 找不到返回None
    """
    if target_gene is None:
        return None

    target = str(target_gene)
    cols = list(df.columns)

    # 1) 精确匹配 (大小写不敏感)
    for c in cols:
        if c.lower() == target.lower():
            return c

    # 2) target_gene + suffix
    expected = (target + suffix).lower()
    for c in cols:
        if c.lower() == expected:
            return c

    # 3) 兜底: 以 target 开头且以 suffix 结尾 (如 "MUC2" -> "MUC2_rnaseq")
    if not target.lower().endswith(suffix.lower()):
        candidates = [c for c in cols if c.lower() == expected]
        if candidates:
            return candidates[0]

    return None


# ============================================================
# 分组阈值计算
# ============================================================

def compute_group_threshold(values: np.ndarray, method: str = "median"):
    """
    根据表达量计算高/低表达分组阈值。

    参数:
        values: [N] 连续表达量 (已剔除NaN)
        method:
            - "median"  : 中位数 (默认)
            - "mean"    : 均值
            - "quantile": 中位数 (等同median, 兼容)
            - 数值字符串/数字: 自定义阈值

    返回:
        float: 阈值 (大于阈值 -> 高表达=1)
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0

    m = str(method).strip().lower()
    if m in ("median", "quantile"):
        return float(np.median(values))
    if m == "mean":
        return float(np.mean(values))

    # 尝试解析为自定义数值阈值
    try:
        return float(method)
    except (TypeError, ValueError):
        return float(np.median(values))


def build_binary_label(values: np.ndarray, threshold: float) -> np.ndarray:
    """
    依据阈值构建二分类标签: 高表达=1, 低表达=0。
    """
    values = np.asarray(values, dtype=np.float64)
    return (values > threshold).astype(np.int64)


def binarize_expression(df: pd.DataFrame, gene_col: str,
                        threshold_method: str = "median") -> pd.Series:
    """
    在DataFrame上直接生成目标基因二分类标签列。

    注意: 这里使用全数据集阈值，仅用于EDA/拆分阶段的粗略分层;
    正式训练折内应基于训练集重算阈值以避免泄露。
    """
    vals = df[gene_col].astype(float).values
    th = compute_group_threshold(vals, threshold_method)
    return pd.Series(build_binary_label(vals, th), index=df.index)


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    import pandas as pd

    df = pd.DataFrame({
        "TCGA ID": [f"TCGA-{i:02d}" for i in range(10)],
        "REG4_rnaseq": np.linspace(0, 10, 10),
    })

    col = resolve_gene_column(df, "REG4", "_rnaseq")
    print(f"解析列名: {col}")

    th = compute_group_threshold(df[col].values, "median")
    print(f"中位数阈值: {th:.3f}")

    lab = build_binary_label(df[col].values, th)
    print(f"二分类标签: {lab.tolist()}")
    print(f"高表达样本数: {int(lab.sum())}/{len(lab)}")

    print("测试通过!")
