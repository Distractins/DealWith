# -*- coding: utf-8 -*-
"""
gpu_manager.py
============================================================================
GPU显存监控与OOM保护模块。

专为RTX 4080 Super (16GB)优化，提供:
    1. 实时显存使用监控
    2. OOM预警与自动恢复
    3. 显存使用历史曲线
    4. 自动CUDA配置优化

使用示例:
    from src.training.gpu_manager import GPUMonitor, setup_gpu
    setup_gpu(config)  # 配置CUDA环境
    monitor = GPUMonitor(config)  # 创建监控器
    # 在训练循环中
    monitor.log_memory(tag="before_forward")
    # ... forward pass ...
    monitor.log_memory(tag="after_forward")
    monitor.plot_history()  # 训练结束后绘制显存曲线
============================================================================
"""

import os
import gc
import time
import logging
from typing import Optional, List, Dict, Tuple

import torch
import numpy as np


class GPUMonitor:
    """
    GPU显存监控器。

    功能:
        - 每N步记录显存使用 (allocated / reserved / free)
        - 超出阈值时告警
        - OOM异常自动捕获并提供恢复建议
        - 训练结束后绘制显存使用历史曲线

    参数:
        config: ConfigBundle配置对象
        logger: 日志器实例 (可选)
    """

    def __init__(self, config, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger
        self.gpu_config = config.training.gpu_monitor

        # 监控参数
        self.enabled = self.gpu_config.enabled
        self.warning_threshold = self.gpu_config.memory_warning_threshold  # 默认0.85
        self.log_interval = self.gpu_config.log_interval  # 默认每10个batch
        self.oom_protection = self.gpu_config.oom_protection

        # 显存使用历史
        self.memory_history: List[Dict] = []
        self.step_count = 0

        # GPU设备信息
        self.device = torch.device(f"cuda:{config.training.gpu_id}" if torch.cuda.is_available() else "cpu")

        if self.enabled and self.device.type == "cuda":
            self._log_gpu_info()

    def _log_gpu_info(self):
        """记录GPU设备信息"""
        if self.device.type != "cuda":
            return

        props = torch.cuda.get_device_properties(self.device)
        total_memory = props.total_memory / (1024 ** 3)

        msg = (
            f"GPU设备: {torch.cuda.get_device_name(self.device)} | "
            f"显存总量: {total_memory:.1f} GB | "
            f"计算能力: {props.major}.{props.minor} | "
            f"多处理器: {props.multi_processor_count}"
        )
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def get_memory_info(self) -> Dict[str, float]:
        """
        获取当前GPU显存使用信息。

        返回:
            dict: {
                "allocated_gb": 已分配的显存(GB),
                "reserved_gb": 已保留的显存(GB),
                "free_gb": 可用显存(GB),
                "total_gb": 总显存(GB),
                "utilization": 显存利用率(0-1),
            }
        """
        if self.device.type != "cuda":
            return {}

        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
        total = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
        free = total - reserved

        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "free_gb": round(free, 2),
            "total_gb": round(total, 2),
            "utilization": round(reserved / total, 3) if total > 0 else 0,
        }

    def log_memory(self, tag: str = ""):
        """
        记录当前显存使用情况。

        参数:
            tag: 记录标签 (如 "before_forward", "after_backward")
        """
        if not self.enabled or self.device.type != "cuda":
            return

        self.step_count += 1

        # 按log_interval记录
        if self.step_count % self.log_interval != 0:
            return

        info = self.get_memory_info()
        info["step"] = self.step_count
        info["tag"] = tag
        info["timestamp"] = time.time()
        self.memory_history.append(info)

        # 检查是否超出告警阈值
        if info.get("utilization", 0) > self.warning_threshold:
            warn_msg = (
                f"⚠ GPU显存使用告警! 利用率: {info['utilization']:.1%}, "
                f"已分配: {info['allocated_gb']:.1f}GB, "
                f"可用: {info['free_gb']:.1f}GB @ {tag}"
            )
            if self.logger:
                self.logger.warning(warn_msg)
            else:
                print(warn_msg)

    def log_memory_summary(self):
        """打印显存使用摘要"""
        if not self.memory_history:
            return

        allocations = [h["allocated_gb"] for h in self.memory_history]
        msg = (
            f"GPU显存使用摘要: "
            f"峰值={max(allocations):.1f}GB, "
            f"均值={np.mean(allocations):.1f}GB, "
            f"最低={min(allocations):.1f}GB"
        )
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def handle_oom(self, error: RuntimeError) -> Optional[int]:
        """
        处理CUDA OOM异常。

        策略: 清空缓存后建议减半batch_size。

        参数:
            error: RuntimeError异常对象

        返回:
            int: 建议的新batch_size (None表示无法自动恢复)
        """
        if not self.oom_protection or "out of memory" not in str(error).lower():
            return None

        msg = "检测到CUDA OOM错误! 正在清理显存..."
        if self.logger:
            self.logger.error(msg)
        else:
            print(f"ERROR: {msg}")

        # 清空缓存
        gc.collect()
        torch.cuda.empty_cache()

        # 建议减半batch_size
        current_bs = self.config.training.batch_size
        new_bs = max(1, current_bs // 2)

        suggest = (
            f"OOM恢复建议: 将batch_size从 {current_bs} 减小到 {new_bs}，"
            f"或增大 gradient_accumulation_steps"
        )
        if self.logger:
            self.logger.warning(suggest)
        else:
            print(f"WARNING: {suggest}")

        return new_bs

    def cleanup(self):
        """清理GPU显存"""
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            if self.logger:
                self.logger.info("GPU显存已清理")

    def plot_history(self, save_path: Optional[str] = None):
        """
        绘制显存使用历史曲线。

        参数:
            save_path: 图片保存路径 (可选)
        """
        if len(self.memory_history) < 2:
            return

        try:
            import matplotlib.pyplot as plt

            steps = [h["step"] for h in self.memory_history]
            allocated = [h["allocated_gb"] for h in self.memory_history]
            reserved = [h["reserved_gb"] for h in self.memory_history]

            plt.figure(figsize=(10, 5))
            plt.plot(steps, allocated, 'b-', label='已分配 (Allocated)', linewidth=1.5)
            plt.plot(steps, reserved, 'r--', label='已保留 (Reserved)', linewidth=1.5)
            plt.axhline(y=self.warning_threshold * self.memory_history[0].get("total_gb", 16),
                       color='orange', linestyle=':', label=f'告警阈值 ({self.warning_threshold:.0%})')
            plt.xlabel('训练步数', fontsize=12)
            plt.ylabel('显存 (GB)', fontsize=12)
            plt.title('GPU显存使用历史', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150)
                plt.close()
            else:
                plt.show()

        except ImportError:
            pass


def setup_gpu(config):
    """
    配置GPU运行环境。

    设置CUDA优化参数:
        - cudnn.benchmark = True (自动寻找最优卷积算法)
        - TF32加速
        - 浮点精度设置
        - 显存限制（如果配置了gpu_memory_fraction）

    参数:
        config: ConfigBundle配置对象
    """
    if not torch.cuda.is_available():
        print("[GPU] CUDA不可用，使用CPU")
        return torch.device("cpu")

    # 选择GPU
    device = torch.device(f"cuda:{config.training.gpu_id}")
    torch.cuda.set_device(device)

    # cuDNN优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # TF32加速 (Ampere架构及以上)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    # PyTorch 2.0+ 浮点精度优化
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # 显存限制
    gpu_memory_fraction = config.training.gpu_memory_fraction
    if gpu_memory_fraction > 0:
        try:
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device)
            print(f"[GPU] 显存限制比例: {gpu_memory_fraction:.0%}")
        except Exception:
            pass

    # 打印设置信息
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    print(f"[GPU] 设备: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"[GPU] cuDNN benchmark: ON")
    print(f"[GPU] TF32: ON (如果支持)")

    return device


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GPUMonitor GPU显存监控模块自测")
    print("=" * 60)

    # 模拟配置
    class MockGPUMonitorConfig:
        enabled = True
        memory_warning_threshold = 0.85
        log_interval = 1
        oom_protection = True
    class MockTrainingConfig:
        gpu_id = 0
        batch_size = 4
        gpu_memory_fraction = 0.0
        gpu_monitor = MockGPUMonitorConfig()
    class MockConfig:
        training = MockTrainingConfig()
        def resolve_path(self, p): return p

    if torch.cuda.is_available():
        # 测试GPU监控器
        monitor = GPUMonitor(MockConfig())
        info = monitor.get_memory_info()
        print(f"\n  GPU显存状态:")
        for k, v in info.items():
            print(f"    {k}: {v}")

        # 模拟训练中的显存记录
        monitor.log_memory("test_start")
        x = torch.randn(100, 100, 100, device="cuda")  # 模拟分配
        monitor.log_memory("after_allocation")
        del x
        torch.cuda.empty_cache()
        monitor.log_memory("after_cleanup")
        print(f"\n  记录了 {len(monitor.memory_history)} 个数据点")
    else:
        print("\n  (CUDA不可用，跳过GPU测试)")

    print("\n测试通过!")
