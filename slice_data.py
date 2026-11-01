# -*- coding: utf-8 -*-
"""
slice_data.py - 一键切片 (自动用 dataset_config.yaml 里的 active 数据集)。

用法:
    python slice_data.py

它会:
    1. 读 dataset_config.yaml 的 active (READ / COAD)
    2. 用对应的 wsi_root / output_root 调上游 run_patch_selection.py
    3. patch 输出到 {active} 对应的 outputs 目录
"""

import os
import sys
import subprocess

try:
    import yaml
except ImportError:
    print("需要 PyYAML: pip install pyyaml")
    sys.exit(1)

BASE = os.path.dirname(os.path.abspath(__file__))  # 仓库根
UPSTREAM_DIR = os.path.join(BASE, "wsi_patch_selection_benchmark")


def main():
    cfg_path = os.path.join(BASE, "dataset_config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    active = cfg.get("active", "COAD")
    d = (cfg.get("datasets") or {}).get(active, {})
    if not d:
        print(f"[slice] 错误: dataset_config.yaml 里没有数据集 '{active}'")
        sys.exit(1)

    wsi_root = d.get("wsi_root", "")
    upstream_output = d.get("upstream_output", "")
    output_root = os.path.join(BASE, upstream_output) if upstream_output else "outputs"

    print("=" * 60)
    print(f"  切片数据集: {active}")
    print(f"  WSI 目录 : {wsi_root}")
    print(f"  输出目录 : {output_root}")
    print("=" * 60)

    cmd = [
        sys.executable, "run_patch_selection.py",
        "--wsi-root", wsi_root,
        "--output-root", output_root,
    ]
    subprocess.run(cmd, cwd=UPSTREAM_DIR, check=True)


if __name__ == "__main__":
    main()
