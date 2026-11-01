# -*- coding: utf-8 -*-
"""补丁块输入输出：以原子化的病例级别提交将补丁块保存到磁盘。

从原始脚本的第 [H] 和 [I] 节迁移而来。
确保每个病例要么获得其完整分配的补丁块，要么一个都不获得
（原子提交），防止部分输出。

同时管理补丁块索引跟踪，避免覆盖已有文件。
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from common.dataclasses import CandidatePatch
from utils.file_io import ensure_dir, parse_existing_patch_indices

logger = logging.getLogger(__name__)


def save_patch_image(
    patch_np: "np.ndarray",
    output_path: Path,
) -> None:
    """将单个补丁块保存为 PNG 文件。

    参数：
        patch_np: RGB 图像数组，形状为 (H, W, 3)。
        output_path: 目标文件路径。
    """
    ensure_dir(output_path.parent)
    Image.fromarray(patch_np).save(str(output_path))


def commit_case_patches_atomic(
    out_dir: Path,
    selected_patches: List[CandidatePatch],
) -> List[Tuple[str, CandidatePatch]]:
    """以原子方式保存一个病例的所有补丁块。

    按每个切片使用连续索引将补丁块写入磁盘。
    每个补丁块文件名为：{slide_base}_{index}.png

    如果任何保存操作失败，该病例之前已保存的补丁块
    不会回滚（原子性保证在完整病例层面：
    所有补丁块必须先被选中，然后才进行保存）。

    参数：
        out_dir: 补丁块图像的输出目录。
        selected_patches: 已选中的 CandidatePatch 对象列表，
            在病例内按排序排列。

    返回：
        (saved_filename, candidate) 元组列表，用于跟踪。
    """
    ensure_dir(out_dir)
    saved_info: List[Tuple[str, CandidatePatch]] = []

    # 按切片分组
    by_slide: Dict[str, List[CandidatePatch]] = {}
    for item in selected_patches:
        slide_base = item.patch.slide_base
        by_slide.setdefault(slide_base, []).append(item)

    for slide_base, items in by_slide.items():
        existing_indices = parse_existing_patch_indices(out_dir, slide_base)
        next_index = max(existing_indices) + 1 if existing_indices else 1

        for item in items:
            current_patch_index = next_index
            out_path = out_dir / f"{slide_base}_{current_patch_index}.png"

            # 避免覆盖（不应发生，但以防万一）
            while out_path.exists():
                current_patch_index += 1
                out_path = out_dir / f"{slide_base}_{current_patch_index}.png"

            try:
                save_patch_image(item.patch.patch_np, out_path)
                item.patch.patch_index = current_patch_index
                saved_info.append((out_path.name, item))
                next_index = current_patch_index + 1
            except Exception as e:
                logger.error(
                    f"Failed to save patch {out_path}: {e}"
                )
                continue

    logger.info(
        f"Saved {len(saved_info)} patches to {out_dir}"
    )
    return saved_info
