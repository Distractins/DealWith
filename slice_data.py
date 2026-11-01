# -*- coding: utf-8 -*-
"""
slice_data.py - 一键切片 (自动用 dataset_config.yaml 里的 active 数据集)。

特点:
    1. 读 dataset_config.yaml 的 active (READ / COAD)
    2. 逐个 WSI 单独调上游 (而非一次跑全部), 坏样本崩溃了就跳过、继续下一张
    3. 断点续跑: 靠上游 progress.json, 已切完的自动跳过

用法:
    python slice_data.py
"""

import os
import sys
import shutil
import glob
import subprocess

try:
    import yaml
except ImportError:
    print("需要 PyYAML: pip install pyyaml")
    sys.exit(1)

BASE = os.path.dirname(os.path.abspath(__file__))  # 仓库根
UPSTREAM_DIR = os.path.join(BASE, "wsi_patch_selection_benchmark")
TMP_DIR = os.path.join(UPSTREAM_DIR, "_single_wsi_tmp")


def _discover_wsis(wsi_root):
    """递归发现所有 .svs (和上游 discover_wsis 行为一致)。"""
    files = glob.glob(os.path.join(wsi_root, "**", "*.svs"), recursive=True)
    files += glob.glob(os.path.join(wsi_root, "**", "*.SVS"), recursive=True)
    return sorted(set(files))


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

    wsi_files = _discover_wsis(wsi_root)
    if not wsi_files:
        print(f"[slice] 没有找到 WSI 文件: {wsi_root}")
        sys.exit(1)

    print("=" * 60)
    print(f"  切片数据集: {active}")
    print(f"  WSI 目录 : {wsi_root}")
    print(f"  WSI 数量 : {len(wsi_files)}")
    print(f"  输出目录 : {output_root}")
    print(f"  模式     : 逐个处理 (坏样本自动跳过)")
    print("=" * 60)

    os.makedirs(TMP_DIR, exist_ok=True)

    ok_count = 0
    fail_count = 0
    failed_files = []

    for i, wsi in enumerate(wsi_files):
        name = os.path.basename(wsi)
        print(f"\n[{i + 1}/{len(wsi_files)}] {name} ...", flush=True)

        # 清空临时目录, 放单个 WSI (硬链接, 不复制大文件)
        for f in glob.glob(os.path.join(TMP_DIR, "*.svs")) + glob.glob(os.path.join(TMP_DIR, "*.SVS")):
            try:
                os.remove(f)
            except OSError:
                pass
        tmp_wsi = os.path.join(TMP_DIR, name)
        try:
            os.link(wsi, tmp_wsi)
        except Exception:
            shutil.copy(wsi, tmp_wsi)

        cmd = [
            sys.executable, "run_patch_selection.py",
            "--wsi-root", TMP_DIR,
            "--output-root", output_root,
        ]
        # 关键: 不设 check=True, 坏样本崩溃只返回非零退出码, 不会中断整个循环
        r = subprocess.run(cmd, cwd=UPSTREAM_DIR)

        if r.returncode == 0:
            ok_count += 1
        else:
            fail_count += 1
            failed_files.append(name)
            print(f"  [跳过坏样本] {name} (退出码 {r.returncode})", flush=True)

    # 汇总
    print("\n" + "=" * 60)
    print(f"  切片完成: 成功 {ok_count}, 失败/跳过 {fail_count}")
    print("=" * 60)

    if failed_files:
        fail_txt = os.path.join(BASE, f"{active}_failed_svs.txt")
        with open(fail_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(failed_files))
        print(f"  失败样本列表已保存: {fail_txt}")
        print(f"  失败样本: {failed_files}")


if __name__ == "__main__":
    main()
