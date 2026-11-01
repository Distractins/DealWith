# -*- coding: utf-8 -*-
"""
match_data.py - 匹配基因组和切片数据, 统一成 1:1 对齐。

功能:
    1. 读基因组表 (dataset_config.yaml 里 active 对应的 table_csv)
    2. 扫 patch 目录 (某采样算法) → 得到有patch的病人
    3. 算交集, 输出:
       - 匹配报告 (交集/仅基因组/仅切片)
       - 过滤后的基因组CSV (只留交集病人) → {PROJECT}_processed_matched.csv
    4. 可选 --delete: 删除「仅切片没有基因组」的多余 patch

用法:
    python match_data.py            # 只报告 + 生成过滤CSV
    python match_data.py --delete   # 同时删除多余patch
    python match_data.py --algorithm sentinel
"""

import os
import re
import sys
import glob
import shutil
import argparse

try:
    import yaml
except ImportError:
    print("需要 PyYAML: pip install pyyaml")
    sys.exit(1)

import csv

BASE = os.path.dirname(os.path.abspath(__file__))


def parse_slide_id(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"^(.+)_(\d+)$", name)
    return m.group(1) if m else name


def slide_to_short(slide_id):
    parts = slide_id.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else slide_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="sentinel")
    parser.add_argument("--delete", action="store_true", help="删除仅切片没有基因组的多余patch")
    args = parser.parse_args()

    # 读 dataset_config.yaml
    cfg = yaml.safe_load(open(os.path.join(BASE, "dataset_config.yaml"), encoding="utf-8"))
    active = cfg["active"]
    d = cfg["datasets"][active]

    table_csv = d["table_csv"]
    if not os.path.isabs(table_csv):
        table_csv = os.path.join(BASE, table_csv)
    upstream_output = os.path.join(BASE, d["upstream_output"])
    patch_dir = os.path.join(upstream_output, "patches", args.algorithm)

    # 1) 基因组病人
    genomic_ids = set()
    with open(table_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.get("TCGA ID", "").strip()
            if pid:
                genomic_ids.add(pid)

    # 2) 有patch的病人 (扫算法目录)
    patch_ids = set()
    patch_id_to_files = {}
    for pf in glob.glob(os.path.join(patch_dir, "*.png")):
        slide = parse_slide_id(pf)
        short = slide_to_short(slide)
        patch_ids.add(short)
        patch_id_to_files.setdefault(short, []).append(pf)

    # 3) 交集 / 差集
    both = genomic_ids & patch_ids
    only_genomic = genomic_ids - patch_ids
    only_patch = patch_ids - genomic_ids

    print("=" * 60)
    print(f"  数据匹配报告: {active} (算法={args.algorithm})")
    print("=" * 60)
    print(f"  基因组病人: {len(genomic_ids)}")
    print(f"  有patch病人: {len(patch_ids)}")
    print(f"  ✅ 交集(两边都有): {len(both)}")
    print(f"  ⚠️ 仅基因组(没patch): {len(only_genomic)}")
    print(f"  ⚠️ 仅切片(没基因组): {len(only_patch)}")
    print("=" * 60)

    if only_genomic:
        print(f"  仅基因组样本: {sorted(only_genomic)}")
    if only_patch:
        print(f"  仅切片样本: {sorted(only_patch)}")

    # 4) 生成过滤后的基因组CSV (只留交集)
    out_csv = os.path.join(os.path.dirname(table_csv), f"{active}_processed_matched.csv")
    with open(table_csv, encoding="utf-8") as fin, open(out_csv, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        n = 0
        for row in reader:
            if row.get("TCGA ID", "").strip() in both:
                writer.writerow(row)
                n += 1
    print(f"\n  过滤后基因组CSV已保存: {out_csv} ({n} 个病人)")

    # 5) 可选: 删除多余patch (仅切片)
    if args.delete and only_patch:
        removed = 0
        for pid in only_patch:
            for pf in patch_id_to_files.get(pid, []):
                try:
                    os.remove(pf)
                    removed += 1
                except OSError:
                    pass
        print(f"  已删除多余patch: {removed} 个文件 ({len(only_patch)} 个仅切片病人)")


if __name__ == "__main__":
    main()
