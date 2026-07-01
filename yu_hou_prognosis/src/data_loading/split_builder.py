# -*- coding: utf-8 -*-
"""
split_builder.py
============================================================================
数据集拆分模块 - 生成5折交叉验证拆分。

从源项目 split_dataset.py 迁移而来，支持:
    1. 生存分析任务的5折分层拆分（按事件+生存时间分层）
    2. N分期分类任务的5折分层拆分（按类别标签分层）

分层策略确保每折的数据分布（事件率、类别比例）与总体一致。

使用示例:
    from src.data_loading.split_builder import build_survival_splits
    splits = build_survival_splits(data_df, n_folds=5, seed=2026)
============================================================================
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import StratifiedKFold


def build_survival_splits(
    data_df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 2026,
    event_col: str = "OS",
    time_col: str = "OS.time",
    patient_id_col: str = "PatientID",
    n_time_bins: int = 3,
) -> Dict[int, Dict[str, Dict]]:
    """
    为生存分析任务生成5折交叉验证拆分。

    分层策略:
        1. 将生存时间分为n_time_bins个区间
        2. 生成组合标签: event × time_bin
        3. 使用StratifiedKFold按组合标签分层拆分

    这确保了每折都有相似的事件率分布和时间分布。

    参数:
        data_df: 包含全部数据的DataFrame
        n_folds: 交叉验证折数 (默认5)
        seed: 随机种子
        event_col: 事件标记列名
        time_col: 生存时间列名
        patient_id_col: 病人ID列名
        n_time_bins: 生存时间分箱数 (默认3)

    返回:
        Dict[int, Dict]: {
            fold_id (1-based): {
                "train": {"patient_ids": [...], ...},
                "test": {"patient_ids": [...], ...},
            }
        }
    """
    print(f"[SplitBuilder] 构建生存分析{n_folds}折交叉验证拆分")
    print(f"  样本数: {len(data_df)}")
    print(f"  事件列: {event_col}, 时间列: {time_col}")

    # 检查必要列
    for col in [event_col, time_col, patient_id_col]:
        if col not in data_df.columns:
            raise ValueError(f"列 '{col}' 不在数据集中。可用列: {list(data_df.columns)[:20]}...")

    # 生存时间分箱
    events = data_df[event_col].values.astype(int)
    times = data_df[time_col].values.astype(float)

    # 对发生事件的患者进行时间分箱
    event_times = times[events == 1]
    if len(event_times) > n_time_bins:
        time_bins = np.percentile(event_times, np.linspace(0, 100, n_time_bins + 1))
    else:
        time_bins = np.linspace(times.min(), times.max(), n_time_bins + 1)

    # 生成组合分层标签
    time_labels = np.digitize(times, time_bins[1:-1])
    stratify_labels = events * n_time_bins + time_labels  # 组合标签

    print(f"  时间分箱边界: {[f'{b:.0f}' for b in time_bins]}")
    print(f"  分层类别数: {len(np.unique(stratify_labels))}")

    # StratifiedKFold拆分
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    splits = {}
    patient_ids = data_df[patient_id_col].values

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(data_df)), stratify_labels)):
        fold_id = fold_idx + 1  # 1-based

        train_pids = patient_ids[train_idx].tolist()
        test_pids = patient_ids[test_idx].tolist()

        # 统计
        train_events = events[train_idx].sum()
        test_events = events[test_idx].sum()
        train_rate = train_events / len(train_idx)
        test_rate = test_events / len(test_idx)

        splits[fold_id] = {
            "train": {"patient_ids": train_pids, "indices": train_idx.tolist()},
            "test": {"patient_ids": test_pids, "indices": test_idx.tolist()},
            "stats": {
                "train_n": len(train_idx), "train_events": int(train_events),
                "train_event_rate": float(train_rate),
                "test_n": len(test_idx), "test_events": int(test_events),
                "test_event_rate": float(test_rate),
            },
        }

        print(f"  Fold {fold_id}: train={len(train_idx)}(事件率={train_rate:.2%}), "
              f"test={len(test_idx)}(事件率={test_rate:.2%})")

    return splits


def build_ncls_splits(
    data_df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 2026,
    label_col: str = "n_stage_label",
    patient_id_col: str = "PatientID",
) -> Dict[int, Dict[str, Dict]]:
    """
    为N分期分类任务生成5折交叉验证拆分。

    按类别标签分层拆分，确保每折各类别比例一致。

    参数:
        data_df: 包含全部数据的DataFrame
        n_folds: 交叉验证折数
        seed: 随机种子
        label_col: 类别标签列名
        patient_id_col: 病人ID列名

    返回:
        Dict: 与build_survival_splits相同的格式
    """
    print(f"[SplitBuilder] 构建N分期分类{n_folds}折交叉验证拆分")

    labels = data_df[label_col].values
    patient_ids = data_df[patient_id_col].values

    uniq, cnt = np.unique(labels, return_counts=True)
    print(f"  类别分布: {dict(zip(uniq.tolist(), cnt.tolist()))}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    splits = {}
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(data_df)), labels)):
        fold_id = fold_idx + 1

        splits[fold_id] = {
            "train": {"patient_ids": patient_ids[train_idx].tolist(), "indices": train_idx.tolist()},
            "test": {"patient_ids": patient_ids[test_idx].tolist(), "indices": test_idx.tolist()},
            "stats": {
                "train_n": len(train_idx), "test_n": len(test_idx),
                "train_dist": {int(k): int(v) for k, v in zip(*np.unique(labels[train_idx], return_counts=True))},
                "test_dist": {int(k): int(v) for k, v in zip(*np.unique(labels[test_idx], return_counts=True))},
            },
        }

    return splits


def save_splits(splits: Dict, output_path: Path) -> None:
    """
    将拆分结果保存为pickle文件（兼容旧格式）。

    参数:
        splits: build_survival_splits()或build_ncls_splits()的返回值
        output_path: 输出.pkl文件路径
    """
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
    N = 300

    # 模拟COAD数据
    mock_df = pd.DataFrame({
        "PatientID": [f"TCGA-{i:04d}" for i in range(N)],
        "OS": np.random.binomial(1, 0.35, N),
        "OS.time": np.random.randint(30, 3650, N).astype(float),
        "age": np.random.randint(30, 90, N).astype(float),
    })

    # 构建生存分析拆分
    splits = build_survival_splits(mock_df, n_folds=5, seed=42)

    # 验证每折事件率是否接近
    print("\n  事件率一致性检查:")
    rates = [s["stats"]["train_event_rate"] for s in splits.values()]
    print(f"    各折训练集事件率: {[f'{r:.2%}' for r in rates]}")
    print(f"    事件率标准差: {np.std(rates):.4f} (越小越一致)")

    print("\n测试通过!")
