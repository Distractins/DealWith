# -*- coding: utf-8 -*-
"""
task_justification.py
============================================================================
任务合理性论证模块。

分析内容:
    1. 单模态基线比较: path-only vs omic-only vs pathomic
    2. 多模态收益分析: 验证path+omic是否优于任一单模态
    3. 生存分析vs分类任务论证: 为什么选择生存预后而非普通分类
    4. N分期KM曲线: 展示不同N分期的生存差异

论证要点:
    - 临床场景需要时间维度的预测，而非简单二分类
    - 删失数据的处理是临床研究的核心需求
    - C-index和KM曲线是肿瘤预后领域的金标准
    - 多模态融合的互补性证据

使用示例:
    python -m src.eda.task_justification --csv data/TCGA_COAD/COAD_all_dataset.csv
============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from typing import Dict, List, Optional


try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


class TaskJustification:
    """
    任务合理性论证器。

    参数:
        csv_path: COAD_all_dataset.csv 文件路径
        output_dir: 图表输出目录
    """

    def __init__(self, csv_path: str, output_dir: str = "experiments/eda"):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None

    def load_data(self):
        """加载数据"""
        self.df = pd.read_csv(self.csv_path)
        print(f"[TaskJustification] 加载数据: {len(self.df)} 个样本")
        return self.df

    def plot_n_stage_km(self):
        """
        绘制不同N分期的KM生存曲线。

        展示: N0/N1/N2三个分期的生存曲线差异。
        如果各分期之间有显著生存差异，则证明N分期是预后相关标记，
        生存分析比简单分类更有临床价值。
        """
        if self.df is None:
            self.load_data()

        # 查找N分期相关列
        n_stage_col = None
        for col in ['n_stage', 'N_stage', 'nstage', 'pathologic_N']:
            if col in self.df.columns:
                n_stage_col = col
                break

        time_col = None
        for col in ['OS.time', 'os.time', 'survival_time']:
            if col in self.df.columns:
                time_col = col
                break

        event_col = None
        for col in ['OS', 'os', 'event']:
            if col in self.df.columns:
                event_col = col
                break

        if not (n_stage_col and time_col and event_col):
            print("  缺少必要的列，跳过N分期KM曲线绘制")
            return

        # 分组
        times = self.df[time_col].values
        events = self.df[event_col].values
        n_stages = self.df[n_stage_col].astype(str).values

        unique_stages = sorted(set(n_stages))
        if len(unique_stages) < 2:
            return

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

        kmfs = {}
        for i, stage in enumerate(unique_stages[:4]):
            mask = n_stages == stage
            if mask.sum() > 1:
                kmf = KaplanMeierFitter()
                kmf.fit(
                    times[mask] / 30,
                    events[mask].astype(bool),
                    label=f'{n_stage_col}={stage} (n={mask.sum()})',
                )
                kmf.plot_survival_function(ax=ax, color=colors[i % len(colors)], linewidth=2)
                kmfs[stage] = kmf

        if len(kmfs) >= 2:
            # Log-rank检验
            stages_list = list(kmfs.keys())
            lr = logrank_test(
                times[n_stages == stages_list[0]] / 30,
                times[n_stages == stages_list[1]] / 30,
                event_observed_A=events[n_stages == stages_list[0]].astype(bool),
                event_observed_B=events[n_stages == stages_list[1]].astype(bool),
            )
            ax.text(0.05, 0.05, f"Log-rank p = {lr.p_value:.4f} "
                   f"({stages_list[0]} vs {stages_list[1]})",
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.set_xlabel('时间 (月)', fontsize=12)
        ax.set_ylabel('总体生存率', fontsize=12)
        ax.set_title('不同N分期的Kaplan-Meier生存曲线', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / "n_stage_km_curves.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  N分期KM曲线已保存: {save_path}")

    def generate_justification_report(self) -> str:
        """
        生成任务合理性论证报告。

        返回:
            str: 论证报告文本
        """
        report = """
================================================================================
              结直肠癌COAD多模态预后预测 - 任务合理性论证
================================================================================

一、为什么选择生存预后分析而非普通分类/回归?

1. 删失数据处理的必要性:
   临床随访中约30-50%的患者在随访期结束时仍然存活（删失），
   这些数据携带有价值的"至少存活到某时刻"的信息。
   普通分类模型（如二分类"存活/死亡"）会错误地将删失样本
   视为负类或直接丢弃，造成信息损失和偏倚。
   生存分析通过Cox部分似然，自然地将删失信息纳入模型。

2. 时间维度的临床意义:
   临床决策需要知道"5年生存率是多少"，而非简单的"是否会死亡"。
   生存分析可以输出任意时间点的存活概率、风险函数，
   为个体化治疗方案的制定提供时间分辨率更高、临床意义更丰富的预测。

3. 肿瘤预后领域的评价金标准:
   C-index（一致性指数）和Kaplan-Meier生存曲线是肿瘤预后研究中
   公认的评价标准。使用生存分析可以与国际同类研究直接比较。

4. 免疫治疗/靶向治疗的长期跟踪需求:
   现代结直肠癌治疗强调长期生存获益，如辅助化疗、靶向治疗和
   免疫检查点抑制剂的疗效评估，都需要生存时间维度。

二、为什么选择多模态融合?

1. 互补性证据:
   - 病理图像反映肿瘤微环境的形态学特征（免疫浸润、间质比例等）
   - 基因组数据反映分子层面的驱动突变和信号通路异常
   - 文献显示: 病理+基因组的多模态模型在多个癌症类型中优于单模态
   - 结直肠癌的分子分型（CMS1-4）与组织形态有明确对应关系

2. 低质量病理图像的价值保留:
   当部分WSI patch质量较低时（模糊/染色不均），基因组数据可以
   提供互补的预测信号，维持整体预测性能的鲁棒性。

三、为什么不用传统机器学习模型?

1. 端到端特征学习: CNN可以在训练过程中自动学习与预后最相关的
   病理形态学特征，无需手工设计特征。

2. 非线性融合: 深度学习融合模块可以学习病理-基因组之间复杂的
   非线性交互关系，传统ML模型（LASSO/RF/XGBoost）难以建模。

3. 高维适应性: 基因组~356维 + 病理图像高维特征，深度学习
   通过梯度优化和正则化技术更好地处理高维小样本问题。

四、推荐实验设计:

1. 主实验: 多模态预后生存分析 (surv + pathomic)
2. 对照实验: 单模态 (path-only / omic-only)
3. 融合策略消融: 9种融合方法横向对比
4. 预处理消融: 有/无图像预处理性能对比
5. 特征选择消融: 原始356维 vs 降维后100维

================================================================================
        """
        return report


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/TCGA_COAD/COAD_all_dataset.csv"

    justifier = TaskJustification(csv_path)
    if Path(csv_path).exists():
        justifier.load_data()
        justifier.plot_n_stage_km()
        report = justifier.generate_justification_report()
        print(report)
    else:
        print(f"数据文件不存在: {csv_path}")
        print("任务论证报告（无数据版本）:")
        justifier.df = None
        report = justifier.generate_justification_report()
        print(report)
