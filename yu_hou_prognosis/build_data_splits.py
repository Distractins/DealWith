# -*- coding: utf-8 -*-
"""
build_data_splits.py
============================================================================
构建完整数据划分文件 — 将make_split.py生成的病人ID划分与上游patch数据、
基因组CSV数据合并，生成cv_runner可直接训练的完整pkl文件。

使用流程:
    1. python make_split.py          # 生成病人ID级划分
    2. python build_data_splits.py   # 构建完整数据划分（本脚本）
    3. python main.py                # 开始训练

输入:
    - make_split.py输出的pkl:  {cv_splits: {1: {train: {patient_ids:[...]}, test:{...}}, ...}}
    - 上游patch目录:           outputs/patches/{algorithm}/*.png
    - 基因组CSV:              data/COAD_all_dataset.csv

输出:
    - 完整数据划分pkl:  data/TCGA_COAD/splits/coad_allst_patient5fold.pkl
      格式: {"fold_1": {"train": {x_path, x_omic, e, t, g}, "test": {...}}, ...}
============================================================================
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional

# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import load_config
from src.data_loading.upstream_reader import UpstreamPatchReader


def build_patient_patch_map(config) -> Dict[str, List[str]]:
    """
    扫描上游patch目录，构建病人ID → patch路径列表的映射。

    处理TCGA ID匹配:
        - CSV中病人ID: TCGA-XX-XXXX (短格式)
        - patch文件名: TCGA-XX-XXXX-01Z-00-DX1_1.png (长格式slide ID)
        - 通过前缀匹配关联

    返回:
        {patient_short_id: [patch_path_1, patch_path_2, ...]}
    """
    reader = UpstreamPatchReader(config)
    patch_dir = reader.get_patch_dir()

    if not patch_dir.exists():
        print(f"[警告] 上游patch目录不存在: {patch_dir}")
        print(f"  将跳过patch路径构建，x_path将为空")
        return {}

    # 扫描所有patch文件，按slide ID前缀分组
    png_files = sorted(patch_dir.glob("*.png"))
    print(f"[扫描] 在 {patch_dir} 中找到 {len(png_files)} 个PNG文件")

    # slide_id → patch路径列表
    slide_patches = defaultdict(list)
    for png_file in png_files:
        slide_id = reader.parse_slide_id(png_file.name)  # e.g., TCGA-XX-XXXX-01Z-00-DX1
        if slide_id:
            slide_patches[slide_id].append(str(png_file.resolve()))

    num_patches = config.upstream.num_patches_per_patient
    print(f"[分组] 识别到 {len(slide_patches)} 个唯一slide ID")

    # 每个slide取前num_patches个patch
    result = {}
    for slide_id, patches in sorted(slide_patches.items()):
        if len(patches) >= num_patches:
            result[slide_id] = sorted(patches)[:num_patches]
        else:
            print(f"  [跳过] {slide_id}: 仅有{len(patches)}个patch (需要>={num_patches})")

    print(f"[结果] {len(result)} 个合格病人 (patch数>={num_patches})")
    return result


def build_full_data_splits(config):
    """
    主函数: 从make_split.py输出构建完整数据划分。
    """
    # ---- 1. 加载病人ID划分 ----
    id_split_path = config.resolve_path(config.data.split_file)
    print(f"[加载] 病人ID划分文件: {id_split_path}")

    if not id_split_path.exists():
        raise FileNotFoundError(
            f"病人ID划分文件不存在: {id_split_path}\n"
            f"请先运行: python make_split.py"
        )

    with open(id_split_path, "rb") as f:
        raw = pickle.load(f)

    # 处理cv_splits格式
    if isinstance(raw, dict) and "cv_splits" in raw:
        id_splits = raw["cv_splits"]
    elif isinstance(raw, dict):
        id_splits = raw
    else:
        raise ValueError(f"无法识别的划分文件格式: {type(raw)}")

    print(f"[信息] 共 {len(id_splits)} 折")

    # ---- 2. 读取基因组CSV ----
    csv_path = config.resolve_path(config.data.genomic_csv)
    print(f"[加载] 基因组数据: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"基因组CSV不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[信息] CSV样本数: {len(df)}")
    print(f"[信息] CSV列: {list(df.columns)[:10]}...")

    # 从CSV中读取的列名（与make_split.py一致）
    patient_id_col = "TCGA ID"
    event_col = "event"
    time_col = "Survival months"

    # 验证列存在
    for col in [patient_id_col, event_col, time_col]:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' 不在CSV中。可用列: {list(df.columns)[:20]}...")

    # 基因组特征列（排除ID列、生存列、分类列等非特征列）
    exclude_patterns = ["TCGA", "ID", "indexes", "event", "censored", "Survival",
                        "vital_status", "survival_source", "tumor_grade", "tumor_stage",
                        "histological_type", "suspicious"]
    feature_cols = [c for c in df.columns
                    if not any(pat.lower() in c.lower() for pat in exclude_patterns)]
    print(f"[信息] 基因组特征维度: {len(feature_cols)}")

    # 构建病人ID → 基因组特征映射（短ID: TCGA-XX-XXXX）
    genomic_map = {}
    survival_map = {}
    for _, row in df.iterrows():
        pid = str(row[patient_id_col]).strip()
        genomic_map[pid] = row[feature_cols].values.astype(np.float32)
        survival_map[pid] = {
            "e": float(row[event_col]),
            "t": float(row[time_col]),
            "g": str(row.get("t", "")),  # 用生存时间作为分组标签（surv任务下可不使用）
        }

    print(f"[信息] 基因组特征映射: {len(genomic_map)} 个病人")

    # ---- 3. 构建patch映射 ----
    patch_map = build_patient_patch_map(config)
    # 按短ID前缀匹配: CSV中 TCGA-XX-XXXX 匹配 patch slide ID TCGA-XX-XXXX-...
    # 构建短ID → 长slide ID列表的映射
    short_to_long = defaultdict(list)
    for slide_id in patch_map:
        # slide_id 如 TCGA-A6-2686-01Z-00-DX1
        # 短ID取前3段: TCGA-A6-2686
        parts = slide_id.split("-")
        if len(parts) >= 3:
            short_id = "-".join(parts[:3])  # TCGA-XX-XXXX
            short_to_long[short_id].append(slide_id)

    # 每个短ID只取第一个长slide ID（一个病人对应一个slide）
    short_to_single = {}
    for short_id, long_ids in sorted(short_to_long.items()):
        if long_ids:
            short_to_single[short_id] = long_ids[0]

    print(f"[匹配] 短ID->长slide映射: {len(short_to_single)} 个")

    # ---- 4. 构建每折的完整数据 ----
    output_folds = {}
    overall_missing_patches = 0
    overall_missing_genomic = 0

    for fold_id in sorted(id_splits.keys()):
        fold_data = id_splits[fold_id]
        fold_key = f"fold_{fold_id}" if isinstance(fold_id, int) else str(fold_id)

        output_folds[fold_key] = {}
        fold_missing_patches = 0
        fold_missing_genomic = 0

        for split_name in ["train", "test"]:
            if split_name not in fold_data:
                continue

            patient_ids = fold_data[split_name].get("patient_ids", [])
            print(f"\n[构建] {fold_key}/{split_name}: {len(patient_ids)} 个病人")

            x_path_list = []   # List[List[str]]
            x_omic_list = []   # np.ndarray
            e_list = []        # np.ndarray
            t_list = []        # np.ndarray
            g_list = []        # List

            for pid in patient_ids:
                pid = str(pid).strip()

                # ---- 基因组数据 ----
                if pid in genomic_map:
                    x_omic_list.append(genomic_map[pid])
                    surv = survival_map[pid]
                    e_list.append(surv["e"])
                    t_list.append(surv["t"])
                    g_list.append(surv["g"])
                else:
                    fold_missing_genomic += 1
                    continue  # 跳过没有基因组数据的病人

                # ---- Patch数据 ----
                if pid in short_to_single:
                    long_id = short_to_single[pid]
                    patches = patch_map.get(long_id, [])
                    if len(patches) >= config.upstream.num_patches_per_patient:
                        x_path_list.append(patches[:config.upstream.num_patches_per_patient])
                    else:
                        fold_missing_patches += 1
                        # 用空列表占位（后续在Dataset中会处理）
                        x_path_list.append([])
                else:
                    fold_missing_patches += 1
                    x_path_list.append([])

            # 转为numpy数组
            n_patients = len(x_omic_list)
            output_folds[fold_key][split_name] = {
                "x_path": x_path_list,
                "x_omic": np.array(x_omic_list, dtype=np.float32) if x_omic_list else np.array([]),
                "e": np.array(e_list, dtype=np.float32) if e_list else np.array([]),
                "t": np.array(t_list, dtype=np.float32) if t_list else np.array([]),
                "g": g_list,
            }

            print(f"  实际病人数: {n_patients}")
            print(f"  缺失patch: {fold_missing_patches}, 缺失基因组: {fold_missing_genomic}")

        overall_missing_patches += fold_missing_patches
        overall_missing_genomic += fold_missing_genomic

    # ---- 5. 统计与验证 ----
    print(f"\n{'='*60}")
    print(f"[汇总]")
    print(f"  总缺失patch病人: {overall_missing_patches}")
    print(f"  总缺失基因组病人: {overall_missing_genomic}")

    # 验证每个fold
    for fold_key in sorted(output_folds.keys()):
        fd = output_folds[fold_key]
        for split_name in ["train", "test"]:
            if split_name in fd:
                data = fd[split_name]
                n = len(data.get("x_omic", []))
                n_patches = len(data.get("x_path", []))
                has_patches = sum(1 for p in data.get("x_path", []) if len(p) > 0)
                print(f"  {fold_key}/{split_name}: {n}病人, "
                      f"有patch={has_patches}/{n_patches}")

    # ---- 6. 保存 ----
    output_path = config.resolve_path(config.data.split_file)
    # 备份原文件
    if output_path.exists():
        backup_path = output_path.with_suffix(".pkl.bak")
        print(f"\n[备份] 原文件 -> {backup_path}")
        output_path.rename(backup_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(output_folds, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n[完成] 完整数据划分已保存: {output_path}")
    print(f"  格式: fold_N -> {{train: {{x_path, x_omic, e, t, g}}, test: {{...}}}}")
    print(f"  cv_runner可直接加载此文件进行训练")

    return output_folds


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    config = load_config()
    build_full_data_splits(config)
