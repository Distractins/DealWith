# -*- coding: utf-8 -*-
"""③ 标签可行性评估: 3 个候选任务在单队列上的打分与评级。"""
import pandas as pd


def assess_feasibility(cohort, basic, gene, surv):
    pca = gene["pca"]
    feas = {}

    # ---- 任务A: PC1 粘液分化连续回归 ----
    n = gene["n_complete"]
    evr = pca["evr"]
    auc = pca["auc"]
    mw_p = pca["mw_p"]
    if n >= 100 and evr >= 0.40 and mw_p < 0.01:
        verdict, reason = "推荐", \
            f"连续标签可用: PC1 集中 {evr:.0%} 方差, 粘液/非粘液显著分离 (p={mw_p:.1e}, AUC={auc:.2f}), 样本 n={n} 充足。"
    elif n >= 50 and evr >= 0.30 and mw_p < 0.05:
        verdict, reason = "谨慎尝试", \
            f"标签有中等信号 (p={mw_p:.1e}, AUC={auc:.2f}, EVR={evr:.0%}), 样本/效应量需在图像模型中再验证。"
    else:
        verdict, reason = "不建议", \
            f"信号不足或样本过少 (n={n}, p={mw_p:.1e}, AUC={auc:.2f}, EVR={evr:.0%})。"
    feas["A"] = {"任务": "A: PC1粘液分化回归", "样本数": n, "PC1解释方差比": round(evr, 3),
                 "PC1vs粘液AUC": round(auc, 3), "MannWhitney_p": mw_p, "评级": verdict, "理由": reason}

    # ---- 任务B: 粘液 vs 非粘液 二分类 ----
    n_pos = basic["粘液腺癌数"]
    n_total = basic["总样本数"]
    rate = n_pos / n_total
    if n_pos >= 50:
        verdict, reason = "推荐", f"正例 {n_pos} 个 ({rate:.0%}), 足以支撑二分类深度模型。"
    elif n_pos >= 20:
        verdict, reason = "谨慎尝试", f"正例 {n_pos} 个 ({rate:.0%}), 可用但类别不平衡, 需重加权/重采样。"
    else:
        verdict, reason = "不建议", f"正例仅 {n_pos} 个 ({rate:.0%}), 正例过少, 深度模型极易过拟合。"
    feas["B"] = {"任务": "B: 粘液vs非粘液二分类", "总样本数": n_total, "正例数": n_pos,
                 "正例占比": round(rate, 3), "评级": verdict, "理由": reason}

    # ---- 任务C: 生存预后 Cox ----
    n_event = surv["n_event"]
    censoring = surv["censoring_rate"]
    epv = surv["epv"]
    if epv >= 10 and censoring < 0.70:
        verdict, reason = "推荐", f"事件数 {n_event} (EPV={epv:.1f}) 且删失率 {censoring:.0%}, 可支撑 Cox。"
    elif epv >= 5:
        verdict, reason = "谨慎尝试", f"事件数 {n_event} (EPV={epv:.1f}), 只适合少量协变量的 Cox, 不宜作主任务。"
    else:
        verdict, reason = "不建议", f"事件数 {n_event} (EPV={epv:.1f}), 删失率 {censoring:.0%}, 事件严重不足, 仅可作事后附属验证。"
    feas["C"] = {"任务": "C: 生存预后Cox", "死亡事件数": n_event, "删失率": round(censoring, 3),
                 "EPV": round(epv, 2), "评级": verdict, "理由": reason}

    return feas


def feas_to_table(feas, cohort, out_tab):
    rows = []
    for k in ["A", "B", "C"]:
        d = dict(feas[k])
        d["队列"] = cohort
        rows.append(d)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out_tab}/03_feasibility.csv", index=False, encoding="utf-8-sig")
    return df
