# -*- coding: utf-8 -*-
"""生成按病人ID的5折交叉验证拆分文件 (训练前运行一次)。

Histomolecular任务:
  - classification: 按目标基因高/低表达标签分层 (保证每折类别均衡)
  - regression    : 按表达量分箱分层

同一病人的6张WSI切片始终在同一折 (以TCGA ID为最小拆分单元)。
"""

import pandas as pd

from src.data_loading.split_builder import build_gene_binary_splits, build_gene_regression_splits, save_splits
from src.data_loading.gene_labels import resolve_gene_column, binarize_expression
from config.config_loader import load_config

config = load_config()

csv_path = config.resolve_path(config.data.genomic_csv)
print(f"数据文件: {csv_path}")
df = pd.read_csv(csv_path)
print(f"样本数: {len(df)}")

patient_id_col = config.data.genomic.patient_id_col
suffix = config.data.genomic.rnaseq_suffix
target_gene = config.model.target_gene
task_type = str(config.model.task_type).lower()

gene_col = resolve_gene_column(df, target_gene, suffix)
if gene_col is None:
    raise ValueError(f"未找到目标基因 '{target_gene}' 的表达列。请检查 CSV 或 data.genomic.target_genes。")
print(f"目标基因表达列: {gene_col}")

if task_type == "classification":
    # 用全数据集阈值粗略分层 (正式折内训练集上会重算阈值避免泄露)
    df["_gene_label"] = binarize_expression(df, gene_col, threshold_method=config.model.group_threshold)
    splits = build_gene_binary_splits(
        df, gene_label_col="_gene_label",
        n_folds=config.cross_validation.n_folds,
        seed=config.model.seed,
        patient_id_col=patient_id_col,
    )
else:
    splits = build_gene_regression_splits(
        df, gene_col=gene_col,
        n_folds=config.cross_validation.n_folds,
        seed=config.model.seed,
        patient_id_col=patient_id_col,
    )

out_path = config.resolve_path(config.data.split_file)
out_path.parent.mkdir(parents=True, exist_ok=True)
save_splits(splits, out_path)
print(f"拆分完成: {out_path}")
