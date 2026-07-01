# -*- coding: utf-8 -*-
"""
upstream_reader.py
============================================================================
上游WSI Patch数据读取模块。

功能:
    1. 自动扫描上游wsi_patch_selection_benchmark输出目录
    2. 根据文件名模式解析slide ID和patch序号
    3. 按病人(slide ID)分组patch文件路径
    4. 构建与数据集兼容的x_path列表

上游输出结构:
    outputs/patches/{algorithm_name}/
    ├── TCGA-A6-2686-01Z-00-DX1_1.png
    ├── TCGA-A6-2686-01Z-00-DX1_2.png
    ├── ...
    ├── TCGA-AA-3518-01Z-00-DX1_1.png
    └── ...

文件命名规则:
    {slide_base}_{idx}.png
    其中slide_base = TCGA-XX-XXXX-01Z-00-DX1（不含扩展名）

使用示例:
    from src.data_loading.upstream_reader import UpstreamPatchReader
    reader = UpstreamPatchReader(config)
    patient_patches = reader.build_patient_patch_map()
    print(f"找到 {len(patient_patches)} 个病人")
============================================================================
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict


class UpstreamPatchReader:
    """
    上游WSI Patch数据读取器。

    负责扫描上游项目输出的patch图像目录，按slide ID分组，
    构建病人-patch映射关系。

    参数:
        config: ConfigBundle配置对象
    """

    def __init__(self, config):
        self.config = config
        self.upstream_config = config.upstream

        # 解析patch根目录路径
        self.patch_root = config.resolve_path(config.upstream.patch_root)

        # 算法名称
        self.algorithm_name = config.upstream.algorithm_name

        # 每个病人需要的patch数量
        self.num_patches = config.upstream.num_patches_per_patient

        # 文件名模式解析
        self.file_pattern = config.upstream.file_pattern  # e.g. "{slide_base}_{idx}.png"

    def get_patch_dir(self) -> Path:
        """获取上游patch输出目录的完整路径"""
        return self.patch_root / self.algorithm_name

    def scan_patch_files(self) -> List[Path]:
        """
        扫描上游目录下的所有PNG patch文件。

        返回:
            List[Path]: patch文件路径列表（按文件名排序）

        异常:
            FileNotFoundError: 上游目录不存在
        """
        patch_dir = self.get_patch_dir()

        if not patch_dir.exists():
            raise FileNotFoundError(
                f"上游patch目录不存在: {patch_dir}\n"
                f"请确保已运行上游wsi_patch_selection_benchmark项目。\n"
                f"如需修改路径，请编辑 config/default_config.yaml 中的 "
                f"upstream.patch_root 和 upstream.algorithm_name。"
            )

        # 扫描所有PNG文件
        png_files = sorted(patch_dir.glob("*.png"))
        print(f"[UpstreamReader] 在 {patch_dir} 中找到 {len(png_files)} 个PNG文件")

        return png_files

    def parse_slide_id(self, filename: str) -> Optional[str]:
        """
        从patch文件名中提取slide ID。

        上游命名规则: TCGA-XX-XXXX-01Z-00-DX1_1.png
        slide ID部分: TCGA-XX-XXXX-01Z-00-DX1 (去掉_patch序号和扩展名)

        参数:
            filename: patch文件名

        返回:
            str: slide ID，解析失败返回None
        """
        # 去掉扩展名
        name = os.path.splitext(filename)[0]

        # 匹配模式: TCGA-XX-XXXX-...-..._数字
        # 去掉最后的 _数字 部分即得到slide ID
        match = re.match(r'^(.+)_(\d+)$', name)
        if match:
            return match.group(1)

        # 如果不匹配预期模式，直接使用文件名（不含扩展名）
        return name

    def parse_patch_index(self, filename: str) -> Optional[int]:
        """
        从patch文件名中提取patch序号。

        参数:
            filename: patch文件名

        返回:
            int: patch序号 (1-based)，解析失败返回None
        """
        name = os.path.splitext(filename)[0]
        match = re.match(r'^.+_(\d+)$', name)
        if match:
            return int(match.group(1))
        return None

    def build_patient_patch_map(self) -> Dict[str, List[str]]:
        """
        构建病人ID到patch文件路径列表的映射。

        扫描上游目录，按slide ID分组，每个病人最多取num_patches个patch。

        返回:
            Dict[str, List[str]]:
                key: slide ID (如 "TCGA-A6-2686-01Z-00-DX1")
                value: patch文件绝对路径列表（按patch序号排序）

        异常:
            FileNotFoundError: 上游目录不存在或为空
        """
        patch_dir = self.get_patch_dir()

        if not patch_dir.exists():
            raise FileNotFoundError(f"上游patch目录不存在: {patch_dir}")

        # 按slide ID分组
        patient_groups = defaultdict(list)

        for png_file in sorted(patch_dir.glob("*.png")):
            slide_id = self.parse_slide_id(png_file.name)
            if slide_id:
                patient_groups[slide_id].append(str(png_file.resolve()))

        # 每个病人取前num_patches个patch（按序号排序）
        result = {}
        skipped_patients = []

        for slide_id, patches in sorted(patient_groups.items()):
            if len(patches) >= self.num_patches:
                result[slide_id] = patches[:self.num_patches]
            else:
                skipped_patients.append((slide_id, len(patches)))

        # 报告统计信息
        print(f"[UpstreamReader] 扫描结果:")
        print(f"  总PNG文件数: {sum(len(v) for v in patient_groups.values())}")
        print(f"  识别的病人数: {len(patient_groups)}")
        print(f"  合格的病人数 (patch数>={self.num_patches}): {len(result)}")
        print(f"  跳过的病人数 (patch不足): {len(skipped_patients)}")

        if skipped_patients and len(skipped_patients) <= 10:
            for sid, cnt in skipped_patients:
                print(f"    - {sid}: 仅有{cnt}个patch")

        if len(result) == 0:
            raise ValueError(
                f"未找到任何合格的病人数据！"
                f"请检查上游patch目录是否为空，或降低 num_patches_per_patient 配置。"
            )

        return result

    def get_patient_list(self) -> List[str]:
        """获取所有合格病人的slide ID列表（排序后）"""
        patient_map = self.build_patient_patch_map()
        return sorted(patient_map.keys())

    def get_patch_count_summary(self) -> Dict[str, int]:
        """
        统计各病人的patch数量分布。

        返回:
            Dict[str, int]: {"total_patients": N, "total_patches": M, ...}
        """
        patient_map = self.build_patient_patch_map()
        total_patches = sum(len(v) for v in patient_map.values())

        return {
            "total_patients": len(patient_map),
            "total_patches": total_patches,
            "patches_per_patient": self.num_patches,
            "patch_dir": str(self.get_patch_dir()),
            "algorithm": self.algorithm_name,
        }


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("UpstreamPatchReader 上游数据读取模块自测")
    print("=" * 60)

    # 测试slide ID解析
    test_filenames = [
        "TCGA-A6-2686-01Z-00-DX1_1.png",
        "TCGA-A6-2686-01Z-00-DX1_2.png",
        "TCGA-AA-3518-01Z-00-DX1_1.png",
        "TCGA-AA-3518-01Z-00-DX1_6.png",
    ]

    # 模拟reader（不需要真实配置）
    class MockConfig:
        class upstream:
            patch_root = "."
            algorithm_name = "grid"
            num_patches_per_patient = 6
            file_pattern = "{slide_base}_{idx}.png"
        def resolve_path(self, p):
            return Path(p)

    reader = UpstreamPatchReader(MockConfig())

    print("\nSlide ID解析测试:")
    for fname in test_filenames:
        slide_id = reader.parse_slide_id(fname)
        idx = reader.parse_patch_index(fname)
        print(f"  {fname}: slide_id={slide_id}, idx={idx}")

    # 验证分组
    groups = defaultdict(list)
    for fname in test_filenames:
        sid = reader.parse_slide_id(fname)
        groups[sid].append(fname)

    print("\n病人分组结果:")
    for sid, files in groups.items():
        print(f"  {sid}: {len(files)}个patch")

    print("\n测试通过!")
