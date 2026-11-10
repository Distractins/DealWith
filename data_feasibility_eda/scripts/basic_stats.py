# -*- coding: utf-8 -*-
"""① 样本基础统计: 规模 / 粘液占比 / TNM-Stage 分布 / 生存事件 / 删失率。"""
import os
import pandas as pd


def run_basic_stats(df, cohort, out_tab):
    total = len(df)
    n_muc = int(df["is_mucinous"].sum())
    n_event = int(df["event"].eq(1).sum())
    n_censored = int(df["censored"].eq(1).sum())

    summary = {
        "队列": cohort,
        "总样本数": total,
        "粘液腺癌数": n_muc,
        "粘液腺癌占比": round(n_muc / total, 3),
        "死亡事件数": n_event,
        "删失数": n_censored,
        "删失率": round(n_censored / total, 3),
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_tab, "01_basic_stats.csv"), index=False, encoding="utf-8-sig")

    # TNM / Stage 分布 (长表)
    dist_rows = []
    for col in ["tumor_stage", "t", "n", "m"]:
        if col not in df.columns:
            continue
        vc = df[col].astype(str).str.strip().replace("", "缺失").value_counts()
        for k, v in vc.items():
            dist_rows.append({"变量": col, "取值": k, "样本数": int(v)})
    pd.DataFrame(dist_rows).to_csv(
        os.path.join(out_tab, "01_tnm_stage_distribution.csv"),
        index=False, encoding="utf-8-sig")

    summary["_stage_counts"] = {
        col: df[col].value_counts().to_dict() for col in ["tumor_stage", "t", "n", "m"]
        if col in df.columns
    }
    return summary
