# -*- coding: utf-8 -*-
"""
split_builder.py - 病人级5折拆分 (回归任务, 按y分箱分层)。

同一个病人的所有patch必须落在同一折 (以病人ID为最小单元)。
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from sklearn.model_selection import StratifiedKFold, KFold


def build_regression_splits(
    patient_ids: List[str],
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 2026,
    n_bins: int = 5,
) -> Dict[int, Dict]:
    """按连续得分y分箱做分层拆分 (回归任务)。"""
    patient_ids = list(patient_ids)
    y = np.asarray(y, dtype=np.float64)

    # 分箱 (qcut可能因重复值失败, 回退到K折)
    try:
        bins = pd.qcut(pd.Series(y), q=min(n_bins, len(y)), labels=False, duplicates="drop")
        bins = bins.fillna(0).astype(int).values
    except Exception:
        bins = None

    if bins is not None and len(np.unique(bins)) >= 2:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        try:
            fold_iter = list(splitter.split(np.zeros(len(y)), bins))
        except ValueError:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            fold_iter = list(splitter.split(np.zeros(len(y))))
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_iter = list(splitter.split(np.zeros(len(y))))

    splits = {}
    for fold_idx, (train_idx, test_idx) in enumerate(fold_iter):
        fold_id = fold_idx + 1
        splits[fold_id] = {
            "train": {"patient_ids": [patient_ids[i] for i in train_idx],
                      "indices": [int(i) for i in train_idx]},
            "test": {"patient_ids": [patient_ids[i] for i in test_idx],
                     "indices": [int(i) for i in test_idx]},
            "stats": {"train_n": int(len(train_idx)), "test_n": int(len(test_idx))},
        }
    return splits


def save_splits(splits: Dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"cv_splits": splits}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[SplitBuilder] 拆分已保存: {output_path}")
