# -*- coding: utf-8 -*-
"""切片识别与分组工具。

从所有原始脚本的公共工具部分迁移而来。
"""

import logging
import re
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def tcga_case_id_from_string(s: str) -> str:
    """从字符串（文件路径或切片名称）中提取 TCGA 病例 ID。

    匹配模式: TCGA-XX-XXXX

    Args:
        s: 包含 TCGA 病例 ID 的输入字符串。

    Returns:
        提取出的病例 ID 字符串，若未找到则返回空字符串。
    """
    m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", s.upper())
    return m.group(1) if m else ""


def slide_base_from_path(slide_path: Path) -> str:
    """获取切片的基础名称（不含扩展名的文件名）。

    Args:
        slide_path: 切片文件的路径。

    Returns:
        文件名的 stem 部分。
    """
    return slide_path.stem


def build_case_to_slides(all_svs: List[Path]) -> Dict[str, List[Path]]:
    """按 TCGA 患者病例 ID 对切片文件进行分组。

    无法识别 TCGA ID 的切片将被归到
    'NO_CASE::{filename}' 下。

    Args:
        all_svs: .svs 文件路径的列表。

    Returns:
        映射 case_id -> 排序后的切片路径列表的字典。
    """
    case_to_slides: Dict[str, List[Path]] = {}
    for svs in all_svs:
        case_id = tcga_case_id_from_string(str(svs))
        if not case_id:
            case_id = f"NO_CASE::{svs.stem}"
        case_to_slides.setdefault(case_id, []).append(svs)

    # 在每个病例内对切片进行排序，以支持确定性处理
    for case_id in case_to_slides:
        case_to_slides[case_id] = sorted(case_to_slides[case_id])

    logger.info(
        f"Found {len(case_to_slides)} cases from {len(all_svs)} slides"
    )
    return case_to_slides


def count_existing_patches_for_case(
    out_dir: Path, case_slides: List[Path]
) -> int:
    """统计某个患者病例已保存的补丁数量。

    Args:
        out_dir: 包含已保存补丁的输出目录。
        case_slides: 该病例的切片路径列表。

    Returns:
        已有补丁文件的总数。
    """
    from utils.file_io import parse_existing_patch_indices

    total = 0
    for slide_path in case_slides:
        slide_base = slide_base_from_path(slide_path)
        total += len(parse_existing_patch_indices(out_dir, slide_base))
    return total
