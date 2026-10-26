# -*- coding: utf-8 -*-
"""
clinical_baseline.py
============================================================================
临床基线诊断脚本 - 回答「Histomolecular任务的天花板在哪」。

用临床特征 (age_at_diagnosis / gender / tumor_stage / t / n / m / histological_type)
做逻辑回归，预测每个目标基因的高/低表达，输出5折交叉验证的 AUC/ACC/F1。

诊断意义:
    - 若临床特征 AUC 明显 > 0.6 (尤其 histological_type 单独就高)，
      说明形态/临床信号真实存在，图像模型用对 backbone 应该也能学到 -> 值得换病理基础模型。
    - 若临床特征也只有 ~0.55，说明这批数据里该表型本身几乎不可预测。

注意:
    - 分组阈值在每折训练集上计算 (与下游训练一致, 避免泄露)。
    - 只用了病理报告里已有的临床/形态标签，不读取 patch 图像。

用法:
    python clinical_baseline.py
    python clinical_baseline.py --genes REG4 MUC2 OLFM4 DEFA5
============================================================================
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.dummy import DummyClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config_loader import load_config
from src.data_loading.gene_labels import (
    resolve_gene_column,
    compute_group_threshold,
    build_binary_label,
)


# 病理报告中常见的临床/形态标签列
CLINICAL_FEATURES = [
    "age_at_diagnosis", "gender", "tumor_stage",
    "t", "n", "m", "histological_type",
]


def _build_feature_pipeline(df: pd.DataFrame, feat_cols):
    """构建临床特征的预处理管线 (数值标准化 + 分类OneHot)。"""
    numeric = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in feat_cols if c not in numeric]

    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        )

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def _evaluate_gene(df, gene, config, feat_cols):
    """用指定临床特征做5折CV逻辑回归, 返回各指标 (每折训练集算阈值防泄露)。"""
    gene_col = resolve_gene_column(df, gene, config.data.genomic.rnaseq_suffix)
    if gene_col is None:
        print(f"  [跳过] 未找到基因 {gene} 的表达列")
        return None

    values = pd.to_numeric(df[gene_col], errors="coerce")
    valid = values.notna() & df[feat_cols].notna().all(axis=1)
    X = df.loc[valid, feat_cols]
    y_cont = values[valid].values.astype(float)

    if len(y_cont) < 10:
        print(f"  [跳过] 基因 {gene} 有效样本不足")
        return None

    pipe = _build_feature_pipeline(df, feat_cols)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.model.seed)
    strat_label = build_binary_label(y_cont, compute_group_threshold(y_cont, config.model.group_threshold))

    aucs, accs, f1s = [], [], []
    for train_idx, test_idx in skf.split(X, strat_label):
        thr = compute_group_threshold(y_cont[train_idx], config.model.group_threshold)
        y_train = build_binary_label(y_cont[train_idx], thr)
        y_test = build_binary_label(y_cont[test_idx], thr)

        pipe.fit(X.iloc[train_idx], y_train)
        proba = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        pred = pipe.predict(X.iloc[test_idx])

        aucs.append(roc_auc_score(y_test, proba))
        accs.append(accuracy_score(y_test, pred))
        f1s.append(f1_score(y_test, pred, zero_division=0))

    # 多数类基线 (Dummy)
    dummy_aucs = []
    dummy = DummyClassifier(strategy="most_frequent")
    for train_idx, test_idx in skf.split(X, strat_label):
        thr = compute_group_threshold(y_cont[train_idx], config.model.group_threshold)
        y_train = build_binary_label(y_cont[train_idx], thr)
        y_test = build_binary_label(y_cont[test_idx], thr)
        dummy.fit(X.iloc[train_idx], y_train)
        dummy_aucs.append(roc_auc_score(y_test, dummy.predict_proba(X.iloc[test_idx])[:, 1]))

    return {
        "gene": gene,
        "n": int(len(y_cont)),
        "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "acc": float(np.mean(accs)),
        "f1": float(np.mean(f1s)),
        "dummy_auc": float(np.mean(dummy_aucs)),
    }


def main():
    parser = argparse.ArgumentParser(description="临床基线诊断")
    parser.add_argument("--config", "-c", type=str, default="config/default_config.yaml")
    parser.add_argument("--genes", type=str, nargs="+", default=None,
                        help="要评估的基因列表 (默认用配置中的target_genes)")
    args = parser.parse_args()

    config = load_config(args.config)
    df = pd.read_csv(config.resolve_path(config.data.genomic_csv))

    feat_cols = [c for c in CLINICAL_FEATURES if c in df.columns]
    print("=" * 80)
    print("  临床基线诊断 - 逻辑回归预测基因高/低表达 (5折CV)")
    print("=" * 80)
    print(f"  样本总数: {len(df)}")
    print(f"  使用临床特征: {feat_cols}")
    print(f"  其中 histological_type 是病理医生的人工形态判读，最能代表'形态信号天花板'")
    print("=" * 80)

    genes = args.genes if args.genes else config.data.genomic.target_genes
    has_hist = "histological_type" in feat_cols
    rows = []
    for gene in genes:
        print(f"\n  >>> 评估基因 {gene} ...")
        r = _evaluate_gene(df, gene, config, feat_cols)
        if not r:
            continue
        # 单看 histological_type (病理医生形态判读) 的天花板
        if has_hist:
            rh = _evaluate_gene(df, gene, config, ["histological_type"])
            if rh:
                r["hist_auc"] = rh["auc"]
        rows.append(r)
        extra = f"  hist_only_AUC={r['hist_auc']:.3f}" if "hist_auc" in r else ""
        print(f"      AUC={r['auc']:.3f}±{r['auc_std']:.3f}  ACC={r['acc']:.3f}  "
              f"F1={r['f1']:.3f}  (多数类基线AUC={r['dummy_auc']:.3f}){extra}")

    if rows:
        print("\n" + "=" * 80)
        print("  汇总 (临床特征逻辑回归, 5折CV)")
        print("=" * 80)
        summary = pd.DataFrame(rows)
        summary.to_csv(config.get_subdir("results") / "clinical_baseline_results.csv", index=False)
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"\n  结果已保存: {config.get_subdir('results') / 'clinical_baseline_results.csv'}")
        print("\n  判读:")
        print("    - AUC > 0.65: 临床/形态信号强, 图像用对backbone应能学到, 值得换病理基础模型")
        print("    - AUC 0.55~0.65: 信号弱, 换模型后提升有限")
        print("    - AUC ~0.5: 该表型在这批数据里几乎不可预测")
    else:
        print("\n  无有效结果")


if __name__ == "__main__":
    main()
