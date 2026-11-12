# -*- coding: utf-8 -*-
"""
slice_robust.py - 每个 case 独立子进程切片, 原生崩溃不影响整体。

用法:
    python slice_robust.py

原理:
    逐个 case 用子进程跑 `run_patch_selection.py --only-case <case_id>`。
    单个 case 的原生崩溃(openslide/imagecodecs 内存问题)只杀死该子进程,
    patch 已保存, 父进程继续下一个, 不再需要反复重跑续传。
"""

import os
import subprocess
import sys
from pathlib import Path

UPSTREAM = Path(__file__).resolve().parent
os.chdir(UPSTREAM)
sys.path.insert(0, str(UPSTREAM))

from datasets.wsi_scanner import discover_wsis
from configs.config_loader import load_config
from utils.file_io import load_progress_json


def _sampler_names(config):
    names = config.enabled_samplers
    if names == "all" or (isinstance(names, list) and "all" in names):
        from samplers import list_samplers
        return list_samplers()
    return list(names)


def main():
    config = load_config()
    case_to_slides = discover_wsis(config.wsi_root, config.only_dx1)
    samplers = _sampler_names(config)

    progress_path = Path(config.output_root) / "progress.json"
    prev = load_progress_json(progress_path) or {}
    completed_keys = set(prev.keys())

    skip_file = Path(config.output_root) / "crashed_cases.txt"
    skip = set(skip_file.read_text(encoding="utf-8").split()) if skip_file.exists() else set()

    pending = []
    for case_id in case_to_slides:
        if all(f"{case_id}||{a}" in completed_keys for a in samplers):
            continue
        if case_id not in skip:
            pending.append(case_id)

    n_done = len(completed_keys) // len(samplers)
    print(f"总case={len(case_to_slides)}, 已完成={n_done}, "
          f"待处理={len(pending)}, 崩溃跳过={len(skip)}", flush=True)

    n_ok = n_crash = 0
    for i, case_id in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] {case_id} ...", flush=True)
        r = subprocess.run(
            [sys.executable, "run_patch_selection.py", "--only-case", case_id],
            cwd=str(UPSTREAM),
        )
        if r.returncode == 0:
            n_ok += 1
            print("    OK", flush=True)
        else:
            n_crash += 1
            skip.add(case_id)
            skip_file.write_text("\n".join(sorted(skip)), encoding="utf-8")
            print(f"    崩溃(退出码 {r.returncode}), patch 已保存, 记入跳过名单", flush=True)

    print(f"\n完成: 成功 {n_ok}, 崩溃但已存patch {n_crash}")
    if n_crash:
        print(f"崩溃的 case 已记入 {skip_file}, 下次运行自动跳过")
    else:
        print("全部 case 处理完毕!")


if __name__ == "__main__":
    main()
