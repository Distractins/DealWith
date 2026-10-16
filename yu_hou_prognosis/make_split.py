# -*- coding: utf-8 -*-
"""生成5折CV拆分文件，训练前运行一次即可。

根据任务类型选择分层策略:
  - surv     : 按事件(死亡)×生存时间分层
  - ncls     : 按N分期三分类(N0/N1/N2)分层
  - n_binary : 按淋巴结转移二分类(N0 vs N1/N2/N3)分层
"""

import pandas as pd

from src.data_loading.split_builder import (
    build_survival_splits,
    build_ncls_splits,
    save_splits,
)
from src.data_loading.datasets import _map_n_stage_to_binary, _map_n_stage_to_3class
from config.config_loader import load_config

config = load_config()

# 读取基因组数据
csv_path = config.resolve_path(config.data.genomic_csv)
print(f"数据文件: {csv_path}")
df = pd.read_csv(csv_path)
print(f"样本数: {len(df)}")

task = str(config.model.task).lower()

if task in ("ncls", "n_binary"):
    # N分期分类任务: 按类别分层，保证每折 N0/N+ 比例一致
    n_col = "n"
    if n_col not in df.columns:
        raise ValueError(f"CSV缺少N分期列 '{n_col}'。可用列: {list(df.columns)[:20]}...")

    if task == "n_binary":
        df["_strat_label"] = df[n_col].astype(str).apply(_map_n_stage_to_binary)
    else:  # ncls 三分类
        df["_strat_label"] = df[n_col].astype(str).apply(_map_n_stage_to_3class)

    print(f"  分层标签分布: {df['_strat_label'].value_counts().to_dict()}")
    splits = build_ncls_splits(
        df,
        n_folds=5,
        seed=config.model.seed,
        label_col="_strat_label",
        patient_id_col="TCGA ID",
    )
else:
    # 生存分析任务: 按事件×生存时间分层
    splits = build_survival_splits(
        df,
        n_folds=5,
        seed=config.model.seed,
        event_col="event",
        time_col="Survival months",
        patient_id_col="TCGA ID",
    )

# 保存
out_path = config.resolve_path(config.data.split_file)
out_path.parent.mkdir(parents=True, exist_ok=True)
save_splits(splits, out_path)
