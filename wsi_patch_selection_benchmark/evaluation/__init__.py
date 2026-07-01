# -*- coding: utf-8 -*-
"""评估层：批量统计、CSV 输出和报告生成。"""

from evaluation.base_evaluator import BaseEvaluator
from evaluation.batch_stats import BatchEvaluator
from evaluation.csv_writer import (
    write_patch_metrics_csv,
    write_slide_metrics_csv,
    write_method_summary_csv,
    write_paper_tables_csv,
)
from evaluation.report_writer import generate_report

__all__ = [
    "BaseEvaluator",
    "BatchEvaluator",
    "write_patch_metrics_csv",
    "write_slide_metrics_csv",
    "write_method_summary_csv",
    "write_paper_tables_csv",
    "generate_report",
]
