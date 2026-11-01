# -*- coding: utf-8 -*-
"""
slice_data.py - 一键切片 (自动用 dataset_config.yaml 里的 active 数据集)。

关键点:
    1. 上游 run_patch_selection.py 有个 bug: load_config() 解析的是空列表,
       会忽略所有 --wsi-root / --output-root 等命令行参数, 只认配置文件。
    2. 因此本脚本通过【直接改写上游配置文件里的 wsi_root】来逐张切,
       坏样本崩溃就跳过、继续下一张; 切完自动恢复原配置。
    3. 断点续跑靠上游 progress.json。

用法:
    python slice_data.py
"""

import os
import re
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
CONFIG_FILE = os.path.join(UPSTREAM_DIR, "configs", "default_config.yaml")
TMP_DIR = os.path.join(UPSTREAM_DIR, "_single_wsi_tmp")


def _discover_wsis(wsi_root):
    files = glob.glob(os.path.join(wsi_root, "**", "*.svs"), recursive=True)
    files += glob.glob(os.path.join(wsi_root, "**", "*.SVS"), recursive=True)
    return sorted(set(files))


def _set_config_wsi_root(path):
    """改写上游配置文件的 wsi_root (字符串替换, 保留其它内容和注释)。"""
    with open(CONFIG_FILE, encoding="utf-8") as f:
        text = f.read()
    path_fwd = path.replace("\\", "/")
    new_text = re.sub(r'wsi_root:\s*"[^"]*"', f'wsi_root: "{path_fwd}"', text, count=1)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(new_text)


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

    # 备份上游原配置 (结束/出错时恢复)
    with open(CONFIG_FILE, encoding="utf-8") as f:
        orig_config = f.read()

    wsi_files = _discover_wsis(wsi_root)
    if not wsi_files:
        print(f"[slice] 没有找到 WSI 文件: {wsi_root}")
        sys.exit(1)

    print("=" * 60)
    print(f"  切片数据集: {active}")
    print(f"  WSI 目录 : {wsi_root}")
    print(f"  WSI 数量 : {len(wsi_files)}")
    print(f"  模式     : 逐张切 (坏样本自动跳过)")
    print("=" * 60)

    os.makedirs(TMP_DIR, exist_ok=True)

    ok_count = 0
    fail_count = 0
    failed_files = []

    try:
        for i, wsi in enumerate(wsi_files):
            name = os.path.basename(wsi)
            print(f"\n[{i + 1}/{len(wsi_files)}] {name} ...", flush=True)

            # 清空临时目录, 放单个 WSI (硬链接, 不复制大文件)
            for f in (glob.glob(os.path.join(TMP_DIR, "*.svs")) +
                      glob.glob(os.path.join(TMP_DIR, "*.SVS"))):
                try:
                    os.remove(f)
                except OSError:
                    pass
            try:
                os.link(wsi, os.path.join(TMP_DIR, name))
            except Exception:
                shutil.copy(wsi, os.path.join(TMP_DIR, name))

            # 改配置文件 wsi_root 指向临时目录 (上游只认配置文件, 不认命令行参数)
            _set_config_wsi_root(TMP_DIR)

            # 跑上游 (不带命令行参数)
            r = subprocess.run([sys.executable, "run_patch_selection.py"], cwd=UPSTREAM_DIR)

            if r.returncode == 0:
                ok_count += 1
            else:
                fail_count += 1
                failed_files.append(name)
                print(f"  [跳过坏样本] {name} (退出码 {r.returncode})", flush=True)
    finally:
        # 恢复原配置 (无论成功失败)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(orig_config)

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
