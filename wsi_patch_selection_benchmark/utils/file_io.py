# -*- coding: utf-8 -*-
"""从原始脚本迁移而来的文件 I/O 工具。

这些函数处理目录创建、CSV 追加、JSON 序列化
以及补丁索引管理——均具备完善的错误处理和日志记录。
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    """创建目录（如果不存在）。

    Args:
        path: 要创建的目录路径。
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise


def append_stats_row(csv_path: Path, row: Dict[str, Any]) -> None:
    """向 CSV 文件追加一行数据。

    如果文件尚不存在，则创建文件并写入表头。
    使用 UTF-8 with BOM 编码以确保 Excel 兼容性。

    Args:
        csv_path: CSV 文件路径。
        row: 表示一行数据的字典。
    """
    try:
        file_exists = csv_path.exists()
        pd.DataFrame([row]).to_csv(
            csv_path,
            mode="a",
            header=not file_exists,
            index=False,
            encoding="utf-8-sig",
        )
    except Exception as e:
        logger.error(f"Failed to append row to {csv_path}: {e}")


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """将字典保存为 JSON 文件。

    Args:
        path: 输出文件路径。
        data: 要序列化的字典。
    """
    try:
        ensure_dir(path.parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {e}")


def parse_existing_patch_indices(out_dir: Path, slide_base: str) -> List[int]:
    """查找某张切片已有的补丁索引。

    Args:
        out_dir: 包含已保存补丁图像的目录。
        slide_base: 切片基础名称。

    Returns:
        排序后的整型补丁索引列表。
    """
    pattern = re.compile(rf"^{re.escape(slide_base)}_(\d+)\.png$", re.IGNORECASE)
    indices = []
    try:
        for p in out_dir.glob(f"{slide_base}_*.png"):
            m = pattern.match(p.name)
            if m:
                indices.append(int(m.group(1)))
    except Exception as e:
        logger.error(f"Error parsing patch indices for {slide_base}: {e}")
    return sorted(indices)


def get_existing_patch_paths_for_slide(
    out_dir: Path, slide_base: str
) -> List[Path]:
    """获取某张切片已有的补丁文件路径。

    Args:
        out_dir: 包含已保存补丁图像的目录。
        slide_base: 切片基础名称。

    Returns:
        排序后的 Path 对象列表。
    """
    pattern = re.compile(rf"^{re.escape(slide_base)}_(\d+)\.png$", re.IGNORECASE)
    paths = []
    try:
        for p in out_dir.glob(f"{slide_base}_*.png"):
            if pattern.match(p.name):
                paths.append(p)
    except Exception as e:
        logger.error(f"Error listing patches for {slide_base}: {e}")
    return sorted(paths)


def save_progress_json(path: Path, data: Dict[str, Any]) -> None:
    """以原子方式将进度数据保存为 JSON。

    首先写入临时文件，然后重命名，以避免在写入过程中
    因断电而导致数据损坏。

    Args:
        path: 目标 JSON 文件路径。
        data: 要序列化的字典。
    """
    tmp_path = path.with_suffix(".tmp")
    try:
        ensure_dir(path.parent)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(path)  # 在 Windows/Linux 上为原子操作
    except Exception as e:
        logger.error(f"Failed to save progress to {path}: {e}")


def load_progress_json(path: Path) -> Dict[str, Any]:
    """从 JSON 文件加载进度数据。

    Args:
        path: 进度 JSON 文件路径。

    Returns:
        进度数据字典；若文件不存在则返回空字典。
    """
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load progress from {path}: {e}")
        return {}


def remove_existing_partial_case_patches(
    out_dir: Path, case_slides: List[Path]
) -> int:
    """移除某病例已有的补丁（用于原子重新处理）。

    Args:
        out_dir: 包含已保存补丁的目录。
        case_slides: 该病例的切片路径列表。

    Returns:
        已删除的文件数量。
    """
    removed = 0
    for slide_path in case_slides:
        slide_base = slide_path.stem
        for p in get_existing_patch_paths_for_slide(out_dir, slide_base):
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass
    return removed
