# -*- coding: utf-8 -*-
"""
训练/验证/推理层 - 训练器、测试器、CV运行器、断点管理、GPU管理

导出:
    - train: 单折训练函数
    - test: 模型测试函数
    - MetricLogger: 训练指标追踪器
    - run_cross_validation: 交叉验证主流程
"""

from src.training.trainer import train, test, MetricLogger
from src.training.cv_runner import run_cross_validation
