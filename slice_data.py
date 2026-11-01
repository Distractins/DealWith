# -*- coding: utf-8 -*-
"""
slice_data.py - 一键切片 (提示当前数据集, 然后直接跑上游)。

重要:
    - 上游只认 wsi_patch_selection_benchmark/configs/default_config.yaml,
      且忽略命令行参数。所以切换数据集要手动改那个配置文件里的两行:
          wsi_root
          output_root
    - 本脚本【不修改任何配置文件】, 避免运行时写坏配置。

用法:
    1. (切换数据集时) 手动改 configs/default_config.yaml 的 wsi_root / output_root
    2. python slice_data.py   (崩溃了就重跑, 自动续传)
"""

import os
import sys
import subprocess

try:
    import yaml
except ImportError:
    print("需要 PyYAML: pip install pyyaml")
    sys.exit(1)

BASE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.join(BASE, "wsi_patch_selection_benchmark")


def main():
    # 读 dataset_config.yaml 仅用于提示, 不改任何文件
    cfg = yaml.safe_load(open(os.path.join(BASE, "dataset_config.yaml"), encoding="utf-8"))
    active = cfg.get("active", "COAD")
    d = (cfg.get("datasets") or {}).get(active, {})

    print("=" * 60)
    print(f"  当前数据集: {active}")
    print(f"  上游配置文件: {os.path.join(UPSTREAM_DIR, 'configs', 'default_config.yaml')}")
    print(f"  请确认配置里的 wsi_root 指向 {d.get('wsi_root', '?')}")
    print(f"  请确认配置里的 output_root 是 {d.get('upstream_output', '?').split('/')[-1]}")
    print("=" * 60)

    # 直接跑上游 (不改配置)
    r = subprocess.run([sys.executable, "run_patch_selection.py"], cwd=UPSTREAM_DIR)

    print("\n" + "=" * 60)
    if r.returncode == 0:
        print("  切片完成!")
    else:
        print(f"  退出码 {r.returncode} (OpenSlide崩溃/中断, 已完成部分已保存)")
        print("  → 重跑 python slice_data.py 自动续传")
    print("=" * 60)


if __name__ == "__main__":
    main()
