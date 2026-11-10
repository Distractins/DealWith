# -*- coding: utf-8 -*-
"""生成 feasibility_report.md。"""
import os
from loader import MODULE_DIR


def _f(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def generate_report(results, cfg):
    L = []
    L.append("# 数据可行性论证报告 (Data Feasibility EDA)")
    L.append("")
    L.append("> 前置可行性论证: 从数据源头评估 COAD / READ 两套 TCGA 队列在 WSI-MIL 弱监督任务上的学习信号,"
             " 供下游深度学习方向决策, 避免盲目训练。本模块仅做统计与可行性, 不含任何 MIL/Backbone/训练代码。")
    L.append("")

    # ---------------- 1. 统计汇总 ----------------
    L.append("## 1. 两个队列统计汇总")
    L.append("")
    L.append("| 指标 | COAD | READ |")
    L.append("|------|------|------|")
    def _basic(key):
        a = results["COAD"]["basic"][key]
        b = results["READ"]["basic"][key]
        return f"{a} / {b}"
    L.append(f"| 总样本数 | {results['COAD']['basic']['总样本数']} | {results['READ']['basic']['总样本数']} |")
    L.append(f"| 粘液腺癌 (占比) | {results['COAD']['basic']['粘液腺癌数']} ({results['COAD']['basic']['粘液腺癌占比']:.0%}) | "
             f"{results['READ']['basic']['粘液腺癌数']} ({results['READ']['basic']['粘液腺癌占比']:.0%}) |")
    L.append(f"| 死亡事件数 | {results['COAD']['basic']['死亡事件数']} | {results['READ']['basic']['死亡事件数']} |")
    L.append(f"| 删失率 | {results['COAD']['basic']['删失率']:.0%} | {results['READ']['basic']['删失率']:.0%} |")
    L.append(f"| 有完整基因样本 | {results['COAD']['gene']['n_complete']} | {results['READ']['gene']['n_complete']} |")
    L.append("")
    L.append(f"- COAD 基因表达为 log2 尺度, READ 为原始 TPM (本模块对 READ 做 log1p 后仅用于**队列内**分析)。")
    if results["READ"]["missing_genes"]:
        L.append(f"- READ 缺少目标基因: {', '.join(g.replace('_rnaseq','') for g in results['READ']['missing_genes'])}"
                 " (仅 COAD 具备全部 6 基因)。")
    L.append("")

    # ---------------- 2. 任务可行性 ----------------
    L.append("## 2. 候选任务可行性评估")
    L.append("")
    L.append("评级标准: 【推荐】信号强且样本充足; 【谨慎尝试】信号存在但样本/效应量需在图像模型中再验证; 【不建议】信号不足或正例/事件过少。")
    L.append("")
    for task_key, title in [("A", "任务A: 粘液分化 PC1 连续回归 (histomolecular)"),
                            ("B", "任务B: 粘液腺癌 vs 非粘液腺癌 二分类"),
                            ("C", "任务C: 生存预后 Cox 建模")]:
        L.append(f"### {title}")
        L.append("")
        for cname in ["COAD", "READ"]:
            f = results[cname]["feas"][task_key]
            L.append(f"**{cname} — {f['评级']}**  ")
            L.append(f"{f['理由']}  ")
            L.append("")
    L.append("")

    # 基因集选择观察
    core_notes = []
    for cname in ["COAD", "READ"]:
        cp = results[cname]["gene"]["core_pca"]
        if cp["genes"] and set(cp["genes"]) != set(results[cname]["genes"]):
            core_notes.append(
                f"- {cname}: 全基因集 PC1 AUC={results[cname]['gene']['pca']['auc']:.2f} → "
                f"核心4基因 (杯状细胞) AUC={cp['auc']:.2f}, EVR={cp['evr']:.2f}")
    if core_notes:
        L.append("**数据观察 (基因集选择)**: 只保留杯状细胞特异基因 (REG1A/REG4/SPINK4/ITLN1) 做 PCA, "
                 "对粘液标签的区分度更高 (OLFM4=肠干细胞、DEFA5=潘氏细胞, 非粘液特异, 会稀释信号)。"
                 "这与 mucus_mil_regression 现用 4 基因一致。")
        L.extend(core_notes)
        L.append("")

    # ---------------- 3. 推荐方向 ----------------
    L.append("## 3. 推荐研究方向")
    L.append("")
    reco = _recommend(results)
    for line in reco:
        L.append(line)
    L.append("")

    # ---------------- 4. 风险清单 ----------------
    L.append("## 4. 风险清单 (深度学习实验需避开的坑)")
    L.append("")
    risks = [
        "**标签泄露**: 分子标签 (PC1/粘液) 由同一张 WSIs 的 RNA 测序得到, 划分 train/val/test 必须按**病人 ID** (同一病人的所有 patch 同折), 且临床/基因特征严禁在划分前用测试集信息。",
        "**跨队列 PCA 不可混用**: COAD 与 READ 的基因尺度不同 (log2 vs TPM), PC1 必须**各队列独立**拟合; 跨队列统一任务时, 两个队列的 PC1 方向/量纲需对齐说明, 不能共用投影矩阵。",
        "**EPV 不足 (生存)**: 生存任务 events/predictors 需 ≥10 才稳; READ 仅 9 个死亡事件, 任何生存模型都极易过拟合, 只可作事后验证。",
        "**高删失**: 两队列删失率均 >80%, KM 尾部置信区间宽, 生存结论需谨慎表述。",
        "**正例稀少 (二分类)**: READ 粘液仅 13 例, 二分类在 READ 上不可行; 若跨队列统一二分类, 需以 COAD 为主并处理类别不平衡 (重加权/Focal)。",
        "**标签偏态 (回归)**: PC1 呈右偏 (少数高粘液样本), 直接 MSE 回归会被多数样本主导; 建议排序/相关性 (Spearman) 作为主指标, 并报告 MAE/RMSE 的分组表现。",
        "**样本量**: 两队列合计 ~480 例, 5 折 CV 下每折测试集 ~30-60 例, 指标方差大, 需报告 mean±std 而非单次。",
    ]
    for r in risks:
        L.append(f"- {r}")
    L.append("")

    # ---------------- 5. 备选叙事 ----------------
    L.append("## 5. 备选叙事方案")
    L.append("")
    L.append("### 方案1: 双队列同一任务 (COAD + READ 联合)")
    L.append("")
    L.append("**推荐任务: 粘液分化 PC1 连续回归**。两队列都具备该连续标签的学习信号, 可合并扩大样本并验证泛化。"
             "实现要点: 各自独立 PCA 得到 PC1 后, 将 PC1 标准化到同一尺度 (如各自 z-score 或统一分箱), "
             "以 COAD 为训练主队列、READ 为跨队列验证队列, 报告两队列各自的 Spearman/MAE。")
    L.append("")
    L.append("### 方案2: COAD 为主任务, READ 作对照")
    L.append("")
    L.append("以 **COAD** 为主 (样本更多、粘液 48 例、6 基因齐全、事件 50 例), 主任务选 PC1 回归或粘液二分类; "
             "**READ** 作为独立外部队列做泛化验证 (仅 PC1 回归可行, 因 READ 粘液仅 13 例、缺 2 基因)。")
    L.append("")

    # 附录: 产物清单
    L.append("## 附录: 产物清单")
    L.append("")
    L.append("每个队列在 `output/{coad,read}/` 下: ")
    L.append("`figures/` 直方图/箱线图/相关热力图/PC1箱线图/KM曲线; `tables/` 基础统计/基因秩和/相关矩阵/PCA/可行性/生存汇总。")
    L.append("")

    report_path = os.path.join(MODULE_DIR, cfg["report_path"])
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return report_path


def _recommend(results):
    fa = {c: results[c]["feas"]["A"]["评级"] for c in ["COAD", "READ"]}
    fb = {c: results[c]["feas"]["B"]["评级"] for c in ["COAD", "READ"]}
    fc = {c: results[c]["feas"]["C"]["评级"] for c in ["COAD", "READ"]}

    lines = []
    if fa["COAD"] in ("推荐", "谨慎尝试") and fa["READ"] in ("推荐", "谨慎尝试"):
        lines.append("**主推方向 ① — 粘液分化 PC1 连续回归 (双队列)**: 连续标签信号在两队列都存在, "
                     "与「WSI 病理弱监督 MIL + Sentinel Patch 采样」主线高度契合 — 图像直接回归分子分化得分, "
                     "并可用注意力热力图验证模型是否聚焦粘液/杯状细胞区域。")
    if fb["COAD"] in ("推荐", "谨慎尝试"):
        lines.append("**辅推方向 ② — 粘液腺癌二分类 (COAD 单队列)**: COAD 正例足够, 可作辅助任务或与回归任务多任务联合; READ 正例过少, 不作训练。")
    if not lines:
        lines.append("**结论: 当前数据在两个队列上均不具备可稳定学习的图像任务**, 建议先扩充样本或改用半监督/自监督预训练策略。")
    lines.append(f"生存任务 (Cox): COAD={fc['COAD']}, READ={fc['READ']} — 仅作事后附属验证, 不作深度学习训练目标。")
    return lines
