# -*- coding: utf-8 -*-
"""
genomic_eda.py
============================================================================
基因组特征探索性数据分析 (EDA) 模块。

分析内容:
    1. 数据概览: 样本量、特征维度、缺失值统计
    2. 生存标签分析: 删失率、中位生存时间、事件分布
    3. 高维特征可视化: PCA降维散点图
    4. 特征方差分布: 各基因特征方差直方图
    5. 单变量Cox回归: Top-20显著基因排名
    6. 生成EDA报告

使用示例:
    python -m src.eda.genomic_eda --csv data/TCGA_COAD/COAD_all_dataset.csv
============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple

try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


class GenomicEDA:
    """
    基因组特征探索性数据分析器。

    参数:
        csv_path: COAD_all_dataset.csv 文件路径
        event_col: 事件标记列名 (默认 "OS")
        time_col: 生存时间列名 (默认 "OS.time")
        output_dir: 输出目录 (用于保存图表)
    """

    def __init__(
        self,
        csv_path: str,
        event_col: str = "OS",
        time_col: str = "OS.time",
        output_dir: str = "experiments/eda",
    ):
        self.csv_path = Path(csv_path)
        self.event_col = event_col
        self.time_col = time_col
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = None
        self.genomic_cols = None

    def load_data(self) -> pd.DataFrame:
        """加载CSV数据"""
        print(f"[GenomicEDA] 加载数据: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        print(f"  样本数: {len(self.df)}, 总列数: {len(self.df.columns)}")
        return self.df

    def identify_genomic_columns(self) -> List[str]:
        """识别基因组特征列（排除临床/ID列）"""
        exclude_patterns = [
            'patient', 'sample', 'bcr', 'os.time', 'os', 'age', 'gender',
            'stage', 'tnm', 'grade', 'histology', 'molecular', 'msi',
            'survival', 'censor', 'event', 'status', 'race', 'ethnicity',
        ]
        self.genomic_cols = [
            c for c in self.df.columns
            if not any(p in c.lower() for p in exclude_patterns)
        ]
        print(f"  识别的基因组特征: {len(self.genomic_cols)} 列")
        return self.genomic_cols

    def basic_summary(self) -> Dict:
        """数据基本概览"""
        print("\n" + "=" * 60)
        print("数据基本概览")
        print("=" * 60)

        # 样本统计
        n_samples = len(self.df)
        n_features = len(self.genomic_cols) if self.genomic_cols else len(self.df.columns)

        # 生存标签
        if self.event_col in self.df.columns:
            events = self.df[self.event_col].values
            n_events = int(np.sum(events == 1))
            n_censored = int(np.sum(events == 0))
            censoring_rate = n_censored / n_samples
        else:
            n_events = n_censored = censoring_rate = None

        # 生存时间
        if self.time_col in self.df.columns:
            times = self.df[self.time_col].dropna().values
            median_time = np.median(times) if len(times) > 0 else None
        else:
            times = None
            median_time = None

        summary = {
            "样本数": n_samples,
            "基因组特征数": n_features,
            "事件数": n_events,
            "删失数": n_censored,
            "删失率": f"{censoring_rate:.1%}" if censoring_rate is not None else "N/A",
            "中位生存时间(天)": f"{median_time:.0f}" if median_time else "N/A",
        }

        for k, v in summary.items():
            print(f"  {k}: {v}")

        # 缺失值统计
        if self.genomic_cols:
            missing = self.df[self.genomic_cols].isnull().sum()
            missing_pct = missing / n_samples
            print(f"\n  缺失值统计 (基因组特征):")
            print(f"    有缺失的列数: {(missing > 0).sum()}")
            print(f"    最大缺失比例: {missing_pct.max():.2%}")
            print(f"    平均缺失比例: {missing_pct.mean():.2%}")

        return summary

    def plot_survival_distribution(self) -> plt.Figure:
        """绘制生存时间分布直方图"""
        if self.time_col not in self.df.columns:
            return None

        times = self.df[self.time_col].dropna().values
        events = self.df[self.event_col].values if self.event_col in self.df.columns else None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 生存时间直方图
        axes[0].hist(times / 30, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        axes[0].axvline(x=np.median(times) / 30, color='red', linestyle='--',
                       label=f'中位: {np.median(times)/30:.0f}月')
        axes[0].set_xlabel('生存时间 (月)')
        axes[0].set_ylabel('患者数')
        axes[0].set_title('生存时间分布')
        axes[0].legend()

        # 删失比例饼图
        if events is not None:
            n_event = int(np.sum(events == 1))
            n_censor = int(np.sum(events == 0))
            axes[1].pie([n_event, n_censor], labels=['死亡', '删失'],
                       colors=['#e74c3c', '#3498db'], autopct='%1.1f%%',
                       explode=(0.05, 0))
            axes[1].set_title(f'事件状态分布 (N={len(events)})')

        plt.tight_layout()
        save_path = self.output_dir / "survival_distribution.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  生存分布图已保存: {save_path}")
        return fig

    def plot_feature_variance(self, top_n: int = 50) -> plt.Figure:
        """绘制基因组特征方差分布"""
        if not self.genomic_cols:
            return None

        variances = self.df[self.genomic_cols].var().sort_values(ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 方差直方图
        axes[0].hist(variances.values, bins=50, color='steelblue', edgecolor='white')
        axes[0].set_xlabel('方差')
        axes[0].set_ylabel('特征数')
        axes[0].set_title(f'基因组特征方差分布 ({len(variances)}个特征)')

        # Top-N高方差特征
        top_var = variances.head(top_n)
        axes[1].barh(range(top_n), top_var.values[::-1], color='coral')
        axes[1].set_xlabel('方差')
        axes[1].set_title(f'Top-{top_n} 高方差特征')

        plt.tight_layout()
        save_path = self.output_dir / "feature_variance.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  特征方差图已保存: {save_path}")
        return fig

    def univariate_cox_ranking(self, top_n: int = 20) -> pd.DataFrame:
        """
        单变量Cox回归显著性排名。

        对每个基因组特征单独拟合Cox模型，按p值排序，
        找出与生存最显著相关的top-N个基因。

        参数:
            top_n: 显示前N个最显著的基因

        返回:
            DataFrame: 排名结果 (基因名, p值, 风险比)
        """
        from lifelines import CoxPHFitter
        import warnings
        warnings.filterwarnings('ignore')

        if self.time_col not in self.df.columns or self.event_col not in self.df.columns:
            print("  缺少生存标签列，跳过单变量Cox分析")
            return None

        if not self.genomic_cols:
            return None

        print(f"\n  单变量Cox回归分析 ({len(self.genomic_cols)}个特征)...")

        results = []
        for col in self.genomic_cols[:min(200, len(self.genomic_cols))]:  # 最多分析200个
            try:
                temp_df = self.df[[col, self.time_col, self.event_col]].dropna()
                temp_df = temp_df.rename(columns={
                    self.time_col: 'time',
                    self.event_col: 'event',
                })
                temp_df['event'] = temp_df['event'].astype(int)

                cph = CoxPHFitter()
                cph.fit(temp_df[['time', 'event', col]], duration_col='time', event_col='event')
                row = cph.summary.loc[col]
                results.append({
                    '基因': col,
                    'p值': float(row['p']),
                    '风险比(HR)': float(np.exp(row['coef'])),
                    '系数': float(row['coef']),
                })
            except Exception:
                continue

        if not results:
            print("  无有效结果")
            return None

        result_df = pd.DataFrame(results).sort_values('p值')

        # 打印Top-N
        print(f"\n  Top-{top_n} 生存相关基因:")
        print(f"  {'基因':<20} {'p值':<12} {'HR':<10}")
        print(f"  {'-'*42}")
        for _, row in result_df.head(top_n).iterrows():
            stars = '***' if row['p值'] < 0.001 else ('**' if row['p值'] < 0.01 else ('*' if row['p值'] < 0.05 else ''))
            print(f"  {row['基因']:<20} {row['p值']:<12.4e} {row['风险比(HR)']:<10.3f} {stars}")

        return result_df

    def run_full_eda(self) -> Dict:
        """运行完整的基因组EDA分析"""
        print("=" * 60)
        print("基因组特征探索性数据分析 (EDA)")
        print("=" * 60)

        self.load_data()
        self.identify_genomic_columns()
        summary = self.basic_summary()
        self.plot_survival_distribution()
        self.plot_feature_variance()
        cox_results = self.univariate_cox_ranking()

        print("\nEDA完成!")
        return {
            "summary": summary,
            "cox_results": cox_results,
        }


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/TCGA_COAD/COAD_all_dataset.csv"

    if Path(csv_path).exists():
        eda = GenomicEDA(csv_path)
        eda.run_full_eda()
    else:
        print(f"数据文件不存在: {csv_path}")
        print("请确保COAD_all_dataset.csv在正确位置")
