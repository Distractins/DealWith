# -*- coding: utf-8 -*-
"""生成5折CV拆分文件，训练前运行一次即可"""

from src.data_loading.split_builder import build_survival_splits, save_splits
from config.config_loader import load_config
import pandas as pd

config = load_config()

# 读取基因组数据
csv_path = config.resolve_path(config.data.genomic_csv)
print(f"数据文件: {csv_path}")
df = pd.read_csv(csv_path)
print(f"样本数: {len(df)}")

splits = build_survival_splits(
    df,
    n_folds=5,
    seed=config.model.seed,
    event_col="event",
    time_col="Survival months",
    patient_id_col="TCGA ID"
)

# 保存
out_path = config.resolve_path(config.data.split_file)
out_path.parent.mkdir(parents=True, exist_ok=True)
save_splits(splits, out_path)