# -*- coding: utf-8 -*-
"""gpu_manager.py - GPU环境配置 (RTX4080 Super 16G)。"""

import torch


def setup_gpu(config) -> torch.device:
    if config.training.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.training.device)

    if device.type == "cuda":
        gpu_id = config.training.gpu_id
        if gpu_id >= torch.cuda.device_count():
            gpu_id = 0
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        frac = config.training.gpu_memory_fraction
        if frac > 0:
            try:
                torch.cuda.set_per_process_memory_fraction(frac, device)
            except Exception:
                pass
        name = torch.cuda.get_device_name(device)
        mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        print(f"[GPU] {name} ({mem:.1f}GB)")
    else:
        print("[GPU] 使用CPU")

    return device


def log_memory(device: torch.device, tag: str = ""):
    if device.type != "cuda":
        return
    a = torch.cuda.memory_allocated(device) / (1024 ** 3)
    r = torch.cuda.memory_reserved(device) / (1024 ** 3)
    print(f"[GPU显存 {tag}] allocated={a:.2f}GB reserved={r:.2f}GB")
