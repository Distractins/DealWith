# -*- coding: utf-8 -*-
"""
data_quality_check.py
============================================================================
数据质量预检与任务方向推荐模块。

在训练前对基因组CSV数据进行全面分析，判断:
    1. 生存预后预测是否可行（事件率、样本量、特征维度）
    2. 哪些替代分类任务更适合当前数据
    3. 基因组特征质量（缺失率、方差分布、特征类型）

输出:
    - 数据概况报告
    - batch跳过概率估算
    - 任务方向推荐（N分期 > T分期 > Stage分类 > 生存预后）
    - 特征选择建议

使用:
    python -m src.eda.data_quality_check --csv data/COAD_all_dataset.csv
============================================================================
"""

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class DataQualityChecker:
    """
    基因组数据质量预检器。

    参数:
        csv_path: COAD_all_dataset.csv 文件路径
    """

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.header: List[str] = []
        self.rows: List[List[str]] = []
        self.n_samples: int = 0

        # 关键列索引
        self.event_idx: int = -1
        self.time_idx: int = -1
        self.n_idx: int = -1
        self.m_idx: int = -1
        self.t_idx: int = -1
        self.stage_idx: int = -1
        self.gender_idx: int = -1
        self.age_idx: int = -1

        # 特征列分组
        self.mut_cols: List[int] = []       # _mut 后缀 (二值突变)
        self.rnaseq_cols: List[int] = []    # _rnaseq 后缀 (连续表达)
        self.other_num_cols: List[int] = [] # 其他数值列

        # 分析结果
        self.results: Dict = {}

    def load(self):
        """加载CSV数据"""
        with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            self.header = next(reader)
            self.rows = [row for row in reader]
        self.n_samples = len(self.rows)
        return self

    def _find_columns(self):
        """定位关键列"""
        for i, col in enumerate(self.header):
            cl = col.lower()
            if 'event' == cl:
                self.event_idx = i
            elif 'survival' in cl and 'month' in cl:
                self.time_idx = i
            elif cl == 'n':
                self.n_idx = i
            elif cl == 'm':
                self.m_idx = i
            elif cl == 't':
                self.t_idx = i
            elif 'tumor_stage' in cl:
                self.stage_idx = i
            elif 'gender' in cl:
                self.gender_idx = i
            elif 'age' in cl and 'diagnosis' in cl:
                self.age_idx = i

            # 特征列分组
            if '_mut' in cl:
                self.mut_cols.append(i)
            elif '_rnaseq' in cl:
                self.rnaseq_cols.append(i)
            elif i >= 19 and all(k not in cl for k in [
                'tcga', 'id', 'indexes', 'event', 'censored', 'survival',
                'vital_status', 'survival_source', 'tumor_grade', 'tumor_stage',
                'histological_type', 'suspicious', 'gender', 'age_at_diagnosis',
                'codeletion', 'idh mutation', 't', 'n', 'm',
            ]):
                try:
                    float(self.rows[0][i])
                    self.other_num_cols.append(i)
                except (ValueError, TypeError):
                    pass

    def analyze(self) -> Dict:
        """执行全面分析"""
        self._find_columns()
        results = {}

        # ============================================================
        # 1. 事件率分析
        # ============================================================
        if self.event_idx >= 0:
            events = [int(r[self.event_idx]) for r in self.rows if r[self.event_idx].strip()]
            n_events = sum(events)
            n_censored = len(events) - n_events
            event_rate = n_events / len(events) * 100 if events else 0

            # batch跳过概率 (batch_size=4, 全删失或全事件)
            p_censor = n_censored / len(events)
            p_all_censored = p_censor ** 4  # 4个样本全是删失
            p_all_event = (n_events / len(events)) ** 4  # 4个样本全是事件
            p_batch_skip = p_all_censored + p_all_event

            results['event_analysis'] = {
                'n_total': len(events),
                'n_events': n_events,
                'n_censored': n_censored,
                'event_rate_pct': round(event_rate, 1),
                'p_batch_all_censored_pct': round(p_all_censored * 100, 1),
                'p_batch_all_event_pct': round(p_all_event * 100, 1),
                'p_batch_skip_total_pct': round(p_batch_skip * 100, 1),
                'expected_batches_skipped_per_epoch': round(p_batch_skip * (len(events) // 4), 1),
                'events_per_fold_5cv': round(n_events / 5, 1),
                'verdict': 'ok' if event_rate >= 25 else (
                    'borderline' if event_rate >= 15 else 'not_recommended'
                ),
            }

        # ============================================================
        # 2. 生存时间分布
        # ============================================================
        if self.time_idx >= 0:
            times = []
            for r in self.rows:
                try:
                    t = float(r[self.time_idx])
                    if t >= 0:
                        times.append(t)
                except (ValueError, TypeError):
                    pass
            times.sort()
            n_t = len(times)
            results['time_analysis'] = {
                'n_valid': n_t,
                'min': round(min(times), 1),
                'max': round(max(times), 1),
                'median': round(times[n_t // 2], 1),
                'mean': round(sum(times) / n_t, 1),
                'p25': round(times[int(n_t * 0.25)], 1),
                'p75': round(times[int(n_t * 0.75)], 1),
                'n_zero_time': sum(1 for t in times if t < 0.1),
            }

        # ============================================================
        # 3. N分期分布 (最重要的替代任务标签)
        # ============================================================
        if self.n_idx >= 0:
            n_vals = [r[self.n_idx].strip() for r in self.rows]
            n_counter = Counter(n_vals)
            # 分组为 N0 / N1 / N2
            n0 = sum(v for k, v in n_counter.items() if k == 'N0')
            n1 = sum(v for k, v in n_counter.items() if k.startswith('N1'))
            n2 = sum(v for k, v in n_counter.items() if k.startswith('N2'))

            results['n_stage_analysis'] = {
                'n0': n0,
                'n1': n1,
                'n2': n2,
                'n0_pct': round(n0 / self.n_samples * 100, 1),
                'n1_pct': round(n1 / self.n_samples * 100, 1),
                'n2_pct': round(n2 / self.n_samples * 100, 1),
                'n_binary_no_vs_yes': f'{n0} vs {n1 + n2}',
                'n_binary_balance': f'{n0 / self.n_samples * 100:.1f}% / {(n1 + n2) / self.n_samples * 100:.1f}%',
                'dist_raw': {k: v for k, v in n_counter.most_common()},
            }

        # ============================================================
        # 4. T分期分布
        # ============================================================
        if self.t_idx >= 0:
            t_vals = [r[self.t_idx].strip() for r in self.rows]
            t_counter = Counter(t_vals)
            t1_2 = sum(v for k, v in t_counter.items() if k in ('T1', 'T2', 'Tis'))
            t3_4 = sum(v for k, v in t_counter.items()
                      if k.startswith('T3') or k.startswith('T4'))
            results['t_stage_analysis'] = {
                't1_2': t1_2,
                't3_4': t3_4,
                'dist_raw': {k: v for k, v in t_counter.most_common()},
            }

        # ============================================================
        # 5. M分期分布
        # ============================================================
        if self.m_idx >= 0:
            m_vals = [r[self.m_idx].strip() for r in self.rows]
            m_counter = Counter(m_vals)
            m0 = m_counter.get('M0', 0)
            m1 = sum(v for k, v in m_counter.items() if k.startswith('M1'))
            mx = sum(v for k, v in m_counter.items() if k == 'MX' or k == '')
            results['m_stage_analysis'] = {
                'm0': m0,
                'm1': m1,
                'mx_unknown': mx,
                'dist_raw': {k: v for k, v in m_counter.most_common()},
            }

        # ============================================================
        # 6. Tumor Stage 分布
        # ============================================================
        if self.stage_idx >= 0:
            stage_vals = [r[self.stage_idx].strip() for r in self.rows]
            stage_counter = Counter(stage_vals)
            early = sum(v for k, v in stage_counter.items()
                       if k.startswith('Stage I') and 'V' not in k)
            late = sum(v for k, v in stage_counter.items()
                      if k.startswith('Stage III') or k.startswith('Stage IV'))
            results['tumor_stage_analysis'] = {
                'early_stage': early,
                'late_stage': late,
                'dist_raw': {k: v for k, v in stage_counter.most_common()},
            }

        # ============================================================
        # 7. 基因组特征质量
        # ============================================================
        results['feature_quality'] = {
            'n_mut_features': len(self.mut_cols),
            'n_rnaseq_features': len(self.rnaseq_cols),
            'n_other_numeric': len(self.other_num_cols),
            'n_total_genomic': len(self.mut_cols) + len(self.rnaseq_cols),
            'n_samples': self.n_samples,
            'feature_to_sample_ratio': round(
                (len(self.mut_cols) + len(self.rnaseq_cols)) / self.n_samples, 1
            ),
        }

        # 分析突变特征信息量
        if self.mut_cols:
            mut_freqs = []
            zero_var = 0
            for col_idx in self.mut_cols:
                vals = []
                for r in self.rows:
                    try:
                        vals.append(int(r[col_idx]))
                    except (ValueError, TypeError):
                        pass
                if vals:
                    freq = sum(vals) / len(vals)
                    mut_freqs.append(freq)
                    if freq == 0 or freq == 1:
                        zero_var += 1
            results['feature_quality']['mut_zero_variance'] = zero_var
            results['feature_quality']['mut_median_frequency'] = (
                round(sorted(mut_freqs)[len(mut_freqs) // 2] * 100, 1) if mut_freqs else 0
            )

        # ============================================================
        # 8. 任务方向推荐
        # ============================================================
        recommendations = []

        # 生存预后
        er = results.get('event_analysis', {})
        if er.get('verdict') == 'not_recommended':
            recommendations.append({
                'task': '生存预后 (surv)',
                'score': 2,
                'verdict': '不推荐',
                'reason': (
                    f'事件率仅{er.get("event_rate_pct", "?")}%，'
                    f'batch跳过率约{er.get("p_batch_skip_total_pct", "?")}%，'
                    f'5折CV每折仅{er.get("events_per_fold_5cv", "?")}个事件。'
                    f'Cox模型难以学到有效信号。'
                ),
            })
        elif er.get('verdict') == 'borderline':
            recommendations.append({
                'task': '生存预后 (surv)',
                'score': 4,
                'verdict': '可尝试但风险高',
                'reason': f'事件率{er.get("event_rate_pct", "?")}%偏低，需强正则化和早停。',
            })

        # N分期二分类
        n_info = results.get('n_stage_analysis', {})
        if n_info:
            n_pos = n_info.get('n1', 0) + n_info.get('n2', 0)
            n_neg = n_info.get('n0', 0)
            if n_pos >= 30 and n_neg >= 30:
                recommendations.append({
                    'task': 'N分期二分类 (n0 vs n1/n2) — 淋巴结转移预测',
                    'score': 9,
                    'verdict': '强烈推荐',
                    'reason': (
                        f'{n_neg} vs {n_pos} 样本，比例{n_neg / self.n_samples * 100:.0f}%:'
                        f'{n_pos / self.n_samples * 100:.0f}%，样本量充足。'
                        f'临床意义明确：预测淋巴结转移可指导手术范围。'
                    ),
                })

        # T分期二分类
        t_info = results.get('t_stage_analysis', {})
        if t_info:
            t_early = t_info.get('t1_2', 0)
            t_late = t_info.get('t3_4', 0)
            if t_early >= 30 and t_late >= 30:
                recommendations.append({
                    'task': 'T分期二分类 (t1/t2 vs t3/t4) — 肿瘤侵犯深度预测',
                    'score': 6,
                    'verdict': '推荐',
                    'reason': (
                        f'{t_early} vs {t_late} 样本，比例不平衡但可用。'
                    ),
                })

        # Tumor Stage 二分类
        ts_info = results.get('tumor_stage_analysis', {})
        if ts_info:
            early = ts_info.get('early_stage', 0)
            late = ts_info.get('late_stage', 0)
            if early >= 30 and late >= 30:
                recommendations.append({
                    'task': 'Tumor Stage二分类 (I/II vs III/IV) — 早晚期预测',
                    'score': 7,
                    'verdict': '推荐',
                    'reason': (
                        f'{early} vs {late} 样本，临床分期预测。'
                    ),
                })

        # M分期
        m_info = results.get('m_stage_analysis', {})
        if m_info:
            m1 = m_info.get('m1', 0)
            if m1 < 30:
                recommendations.append({
                    'task': 'M分期二分类 (m0 vs m1) — 远处转移预测',
                    'score': 3,
                    'verdict': '不推荐',
                    'reason': f'M1仅{m1}例，样本严重不足。',
                })

        recommendations.sort(key=lambda x: x['score'], reverse=True)
        results['recommendations'] = recommendations
        results['best_task'] = recommendations[0] if recommendations else None

        self.results = results
        return results

    def print_report(self):
        """打印完整分析报告"""
        if not self.results:
            self.analyze()

        r = self.results

        print()
        print("=" * 70)
        print("  基因组数据质量预检报告 — COAD预后预测可行性评估")
        print("=" * 70)

        # ---- 基本信息 ----
        print(f"\n  CSV文件: {self.csv_path}")
        print(f"  总样本数: {self.n_samples}")
        print(f"  基因组特征: {len(self.mut_cols) + len(self.rnaseq_cols)} "
              f"(突变{len(self.mut_cols)} + RNA-seq{len(self.rnaseq_cols)})")
        print(f"  特征/样本比: {r['feature_quality']['feature_to_sample_ratio']}:1 "
              f"{'⚠️ 严重过参数化!' if r['feature_quality']['feature_to_sample_ratio'] > 10 else ''}")

        # ---- 事件率 ----
        ea = r.get('event_analysis', {})
        if ea:
            print(f"\n{'─' * 50}")
            print(f"  [1] 生存事件分析")
            print(f"{'─' * 50}")
            print(f"  事件数: {ea['n_events']} | 删失数: {ea['n_censored']}")
            print(f"  事件率: {ea['event_rate_pct']}%")
            print(f"  batch_size=4时，全删失batch概率: {ea['p_batch_all_censored_pct']}%")
            print(f"  batch跳过总概率: {ea['p_batch_skip_total_pct']}%")
            print(f"  预计每epoch跳过: ~{ea['expected_batches_skipped_per_epoch']}个batch")
            print(f"  5折CV每折事件数: ~{ea['events_per_fold_5cv']}")

            if ea['verdict'] == 'not_recommended':
                print(f"  ⚠️ 结论: 事件率严重不足，不推荐做生存预后预测")
            elif ea['verdict'] == 'borderline':
                print(f"  ⚠️ 结论: 事件率偏低，生存预后困难但可尝试")

        # ---- 分类任务标签分布 ----
        for label, info_key in [
            ('N分期 (淋巴结转移)', 'n_stage_analysis'),
            ('T分期 (肿瘤侵犯)', 't_stage_analysis'),
            ('M分期 (远处转移)', 'm_stage_analysis'),
            ('Tumor Stage', 'tumor_stage_analysis'),
        ]:
            info = r.get(info_key, {})
            if info:
                print(f"\n{'─' * 50}")
                print(f"  [{label}]")
                print(f"{'─' * 50}")
                dist = info.get('dist_raw', {})
                for k, v in dist.items():
                    print(f"  {k}: {v} ({v / self.n_samples * 100:.1f}%)")

        # ---- 特征质量 ----
        fq = r.get('feature_quality', {})
        if fq:
            print(f"\n{'─' * 50}")
            print(f"  [特征质量]")
            print(f"{'─' * 50}")
            print(f"  突变特征: {fq['n_mut_features']}维 "
                  f"(零方差={fq.get('mut_zero_variance', '?')}, "
                  f"中位频率={fq.get('mut_median_frequency', '?')}%)")
            print(f"  RNA-seq特征: {fq['n_rnaseq_features']}维")
            print(f"  特征/样本比: {fq['feature_to_sample_ratio']}:1")

            if fq['feature_to_sample_ratio'] > 10:
                print(f"  ⚠️ 特征维度远大于样本数! 必须做特征选择。")
                print(f"  建议: 1) 保留突变特征(500维二值)")
                print(f"        2) RNA-seq特征用方差阈值过滤(保留top-500)")
                print(f"        3) 或用PCA降至100-200维")
                print(f"        4) 总特征控制在1000维以内")

        # ---- 任务推荐 ----
        recs = r.get('recommendations', [])
        if recs:
            print(f"\n{'=' * 70}")
            print(f"  ★ 任务方向推荐 (综合评分)")
            print(f"{'=' * 70}")
            for i, rec in enumerate(recs):
                score_bar = '█' * rec['score'] + '░' * (10 - rec['score'])
                print(f"\n  [{rec['verdict']}] {rec['task']}")
                print(f"  评分: {score_bar} ({rec['score']}/10)")
                print(f"  理由: {rec['reason']}")

        # ---- 最终建议 ----
        best = r.get('best_task', {})
        if best:
            print(f"\n{'=' * 70}")
            print(f"  ★★★ 最终建议 ★★★")
            print(f"{'=' * 70}")
            print(f"  推荐任务: {best['task']}")
            print(f"  理由: {best['reason']}")
            print()
            print(f"  实施步骤:")
            print(f"    1. 修改 config: model.task = 'ncls'")
            print(f"    2. 特征处理: 突变500维 + RNA-seq方差top-500 = 1000维")
            print(f"    3. 测试多种融合策略: concat -> caugf -> cross_attention")
            print(f"    4. 评价指标: Accuracy, F1, AUC (替代C-index)")
            print(f"    5. 对照: 先用纯基因组(logistic regression)做baseline")

        print()
        print("=" * 70)
        print("  分析完成")
        print("=" * 70)


# ============================================================
# 命令行入口
# ============================================================
if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/COAD_all_dataset.csv"

    checker = DataQualityChecker(csv_path)
    checker.load()
    checker.analyze()
    checker.print_report()
