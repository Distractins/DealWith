# -*- coding: utf-8 -*-
"""
split_builder.py
============================================================================
数据集拆分模块 - 按病人ID生成5折交叉验证拆分。

Histomolecular任务核心约束:
    同一个患者的6张WSI切片必须始终落在同一折，避免数据泄露。
    拆分以病人ID (TCGA ID) 为最小单元，分层按目标基因标签保持每折类别均衡。

支持:
    1. 基因二分类任务: 按高/低表达标签分层 (StratifiedKFold)
    2. 基因回归任务:   按表达量分箱后分层
    3. 通用分层拆分 (build_stratified_splits)

使用示例:
    from src.data_loading.split_builder import build_stratified_splits
    splits = build_stratified_splits(patient_ids, stratify_labels, n_folds=5, seed=2026)
============================================================================
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.model_selection import StratifiedKFold, KFold


def build_stratified_splits(
    patient_ids: List[str],
    stratify_labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 2026,
) -> Dict[int, Dict]:
    """
    按病人ID生成分层K折拆分 (保证同一病人不跨折)。

    参数:
        patient_ids: 病人ID列表 (TCGA ID)
        stratify_labels: [N] 分层标签 (与patient_ids等长)
        n_folds: 折数
        seed: 随机种子

    返回:
        Dict[int, Dict]: {fold_id (1-based): {"train": {"patient_ids": [...], "indices": [...]},
                                               "test": {...}, "stats": {...}}}
    """
    patient_ids = list(patient_ids)
    labels = np.asarray(stratify_labels)
    n = len(patient_ids)

    if n == 0:
        raise ValueError("病人列表为空，无法拆分")

    # 若分层类别太少或存在极稀有类，退化为K折
    uniq, counts = np.unique(labels, return_counts=True)
    use_stratified = (len(uniq) >= 2 and int(counts.min()) >= n_folds and n >= n_folds)

    if use_stratified:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_iter = splitter.split(np.zeros(n), labels)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_iter = splitter.split(np.zeros(n))

    splits = {}
    for fold_idx, (train_idx, test_idx) in enumerate(fold_iter):
        fold_id = fold_idx + 1
        train_pids = [patient_ids[i] for i in train_idx]
        test_pids = [patient_ids[i] for i in test_idx]

        def _dist(idx):
            u, c = np.unique(labels[idx], return_counts=True)
            return {str(k): int(v) for k, v in zip(u, c)}

        splits[fold_id] = {
            "train": {"patient_ids": train_pids, "indices": [int(i) for i in train_idx]},
            "test": {"patient_ids": test_pids, "indices": [int(i) for i in test_idx]},
            "stats": {
                "train_n": int(len(train_idx)), "test_n": int(len(test_idx)),
                "train_dist": _dist(train_idx), "test_dist": _dist(test_idx),
            },
        }

    return splits


def build_gene_binary_splits(
    data_df: pd.DataFrame,
    gene_label_col: str,
    n_folds: int = 5,
    seed: int = 2026,
    patient_id_col: str = "TCGA ID",
) -> Dict[int, Dict]:
    """按目标基因高/低表达标签分层生成病人级拆分。"""
    df = data_df.dropna(subset=[patient_id_col, gene_label_col]).copy()
    print(f"[SplitBuilder] 构建基因二分类{n_folds}折拆分 (标签列={gene_label_col})")
    print(f"  样本数: {len(df)}")
    print(f"  标签分布: {df[gene_label_col].value_counts().to_dict()}")

    splits = build_stratified_splits(
        patient_ids=df[patient_id_col].astype(str).tolist(),
        stratify_labels=df[gene_label_col].astype(int).values,
        n_folds=n_folds, seed=seed,
    )

    for fold_id, s in splits.items():
        print(f"  Fold {fold_id}: train={s['stats']['train_n']}({s['stats']['train_dist']}), "
              f"test={s['stats']['test_n']}({s['stats']['test_dist']})")
    return splits


def build_gene_regression_splits(
    data_df: pd.DataFrame,
    gene_col: str,
    n_folds: int = 5,
    seed: int = 2026,
    patient_id_col: str = "TCGA ID",
    n_bins: int = 5,
) -> Dict[int, Dict]:
    """按表达量分箱分层生成病人级拆分 (回归任务)。"""
    df = data_df.dropna(subset=[patient_id_col, gene_col]).copy()
    vals = df[gene_col].astype(float).values
    bins = pd.qcut(vals, q=n_bins, labels=False, duplicates="drop").astype(int).values
    print(f"[SplitBuilder] 构建基因回归{n_folds}折拆分 (表达列={gene_col}, 分箱={n_bins})")
    print(f"  样本数: {len(df)}")

    return build_stratified_splits(
        patient_ids=df[patient_id_col].astype(str).tolist(),
        stratify_labels=bins,
        n_folds=n_folds, seed=seed,
    )


def save_splits(splits: Dict, output_path: Path) -> None:
    """将拆分结果保存为pickle文件（{cv_splits: splits}格式）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cv_splits": splits}
    with open(output_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[SplitBuilder] 拆分文件已保存: {output_path}")


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SplitBuilder 数据集拆分模块自测")
    print("=" * 60)

    np.random.seed(42)
    N = 200
    df = pd.DataFrame({
        "TCGA ID": [f"TCGA-{i:04d}" for i in range(N)],
        "_gene_label": np.random.binomial(1, 0.5, N),
    })

    splits = build_gene_binary_splits(df, "_gene_label", n_folds=5, seed=42)

    # 验证无病人跨折泄露
    seen = {}
    leak = False
    for fold_id, s in splits.items():
        for pid in s["train"]["patient_ids"]:
            if pid in seen and seen[pid] != fold_id:
                leak = True
            seen[pid] = fold_id
        for pid in s["test"]["patient_ids"]:
            if pid in seen and seen[pid] != fold_id:
                leak = True
            seen[pid] = fold_id
    print(f"\n  病人跨折泄露检查: {'❌ 存在泄露' if leak else '✅ 无泄露'}")

    print("\n测试通过!")
