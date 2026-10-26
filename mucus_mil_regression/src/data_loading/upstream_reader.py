# -*- coding: utf-8 -*-
"""
upstream_reader.py - 上游patch读取 (复用wsi_patch_selection_benchmark输出)。

命名规则: {slide_base}_{idx}.png, 如 TCGA-A6-2686-01Z-00-DX1_1.png
按slide ID分组, 建立 短TCGA ID -> patch路径列表 的映射 (每个病人取前num_patches张)。
"""

import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional


class UpstreamPatchReader:
    def __init__(self, config):
        self.config = config
        self.patch_root = config.resolve_path(config.upstream.patch_root)
        self.algorithm_name = config.upstream.algorithm_name
        self.num_patches = config.upstream.num_patches_per_patient

    def get_patch_dir(self) -> Path:
        return self.patch_root / self.algorithm_name

    @staticmethod
    def parse_slide_id(filename: str) -> Optional[str]:
        """从patch文件名提取slide ID (去掉末尾 _序号 与扩展名)。"""
        name = Path(filename).stem
        m = re.match(r"^(.+)_(\d+)$", name)
        return m.group(1) if m else name

    @staticmethod
    def slide_to_short(slide_id: str) -> Optional[str]:
        """长slide ID -> 短TCGA ID (TCGA-A6-2686-01Z-00-DX1 -> TCGA-A6-2686)。"""
        parts = slide_id.split("-")
        return "-".join(parts[:3]) if len(parts) >= 3 else slide_id

    def build_short_to_patches(self) -> Dict[str, List[str]]:
        """
        扫描上游patch目录, 返回 short TCGA ID -> patch路径列表 映射。
        每个slide取前 num_patches 张 (按文件名序号排序)。
        """
        patch_dir = self.get_patch_dir()
        if not patch_dir.exists():
            raise FileNotFoundError(f"上游patch目录不存在: {patch_dir}")

        slide_patches = defaultdict(list)
        for pf in sorted(patch_dir.glob("*.png")):
            sid = self.parse_slide_id(pf.name)
            if sid:
                slide_patches[sid].append(str(pf.resolve()))

        short_to_patches: Dict[str, List[str]] = {}
        for sid, patches in slide_patches.items():
            if len(patches) < self.num_patches:
                continue
            short = self.slide_to_short(sid)
            # 若同一病人有多个slide, 保留第一个 (上游 only_dx1 通常仅一个)
            if short not in short_to_patches:
                short_to_patches[short] = sorted(patches)[: self.num_patches]

        print(f"[UpstreamReader] 算法={self.algorithm_name}, 合格slide={len(slide_patches)}, "
              f"可用病人={len(short_to_patches)}")
        return short_to_patches
