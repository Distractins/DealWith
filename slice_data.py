# -*- coding: utf-8 -*-
"""
slice_data.py - 一键切片 (自动用 dataset_config.yaml 里的 active 数据集)。

关键点:
    1. 上游 run_patch_selection.py 忽略命令行参数(load_config解析空列表),
       只认配置文件 configs/default_config.yaml。所以本脚本直接改写配置文件。
    2. OpenSlide 4.0.1 在 Windows 有崩溃bug(进程退出时tiff cache断言失败),
       但 patch 在崩溃前已经保存, 且上游有 progress.json 断点续跑。
    3. 所以: 跑一次会切一批(可能中途崩), 重跑本脚本自动续传, 多跑几次直到全切完。

用法:
    python slice_data.py
    (崩溃了就再跑一次, 会自动跳过已完成的、继续切剩下的)
"""

import os
import re
import sys
import subprocess

try:
    import yaml
except ImportError:
    print("需要 PyYAML: pip install pyyaml")
    sys.exit(1)

BASE = os.path.dirname(os.path.abspath(__file__))  # 仓库根
UPSTREAM_DIR = os.path.join(BASE, "wsi_patch_selection_benchmark")
CONFIG_FILE = os.path.join(UPSTREAM_DIR, "configs", "default_config.yaml")


def _set_config(wsi_root, output_root):
    """逐行改写上游配置文件的 wsi_root 和 output_root (只动这两行, 其它原样保留)。"""
    with open(CONFIG_FILE, encoding="utf-8") as f:
        lines = f.readlines()
    wsi_fwd = wsi_root.replace("\\", "/")
    out_fwd = output_root.replace("\\", "/")
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("wsi_root:"):
            out.append(f'wsi_root: "{wsi_fwd}"\n')
        elif stripped.startswith("output_root:"):
            out.append(f'output_root: "{out_fwd}"\n')
        else:
            out.append(line)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.writelines(out)


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
    output_root = os.path.join(BASE, d.get("upstream_output", "outputs"))

    print("=" * 60)
    print(f"  切片数据集: {active}")
    print(f"  WSI 目录 : {wsi_root}")
    print(f"  输出目录 : {output_root}")
    print(f"  模式     : 批量切 (崩溃重跑自动续传)")
    print("=" * 60)

    # 改写配置文件 (上游只认配置文件)
    _set_config(wsi_root, output_root)

    # 跑上游一次 (批量)
    r = subprocess.run([sys.executable, "run_patch_selection.py"], cwd=UPSTREAM_DIR)

    print("\n" + "=" * 60)
    if r.returncode == 0:
        print("  切片完成! (正常退出)")
    else:
        print(f"  本次退出码 {r.returncode} (OpenSlide崩溃, 但已完成的patch已保存)")
        print("  → 直接重跑: python slice_data.py  会自动续传剩下的")
    print("=" * 60)


if __name__ == "__main__":
    main()
