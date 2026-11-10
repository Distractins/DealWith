# -*- coding: utf-8 -*-
"""data_feasibility_eda 主入口: 一键运行 ①-④ 并生成报告。

用法: python scripts/run_eda.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loader import load_config, load_cohort, out_dir, setup_mpl
from basic_stats import run_basic_stats
from gene_eda import run_gene_eda
from feasibility import assess_feasibility, feas_to_table
from survival import run_survival
from report import generate_report


def main():
    cfg = load_config()
    plt = setup_mpl()
    results = {}

    for cohort, c in cfg["cohorts"].items():
        print(f"\n===== 分析队列: {cohort} =====")
        df, genes, missing, gene_unit = load_cohort(cfg, cohort)
        out_fig = out_dir(cfg, cohort, "figures")
        out_tab = out_dir(cfg, cohort, "tables")

        basic = run_basic_stats(df, cohort, out_tab)
        gene = run_gene_eda(df, genes, cfg["core_mucus_genes"], cohort, gene_unit, out_fig, out_tab, plt)
        surv = run_survival(df, cohort, out_fig, out_tab, plt, cfg)
        feas = assess_feasibility(cohort, basic, gene, surv)
        feas_to_table(feas, cohort, out_tab)

        results[cohort] = {
            "cohort": cohort, "genes": genes, "missing_genes": missing,
            "n": len(df), "gene_unit": gene_unit,
            "basic": basic, "gene": gene, "surv": surv, "feas": feas,
        }
        print(f"  n={len(df)}, 基因={[g.replace('_rnaseq','') for g in genes]}, "
              f"缺={[g.replace('_rnaseq','') for g in missing] or '无'}")
        print(f"  可行性: A={feas['A']['评级']}, B={feas['B']['评级']}, C={feas['C']['评级']}")

    report_path = generate_report(results, cfg)
    print(f"\n完成。报告: {report_path}")


if __name__ == "__main__":
    main()
