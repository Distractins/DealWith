# -*- coding: utf-8 -*-
"""WSI 文件发现与病例分组。

扫描根目录中的 .svs 文件，可选地按 DX1（诊断）
切片进行过滤，并按 TCGA 患者病例 ID 进行分组。
"""

import logging
from pathlib import Path
from typing import Dict, List

from utils.slide_reader import build_case_to_slides

logger = logging.getLogger(__name__)


def discover_wsis(
    wsi_root: str,
    only_dx1: bool = True,
    extensions: tuple = (".svs", ".tiff", ".tif", ".ndpi", ".mrxs"),
) -> Dict[str, List[Path]]:
    """发现 WSI 文件并按病例 ID 进行分组。

    Args:
        wsi_root: 用于搜索 WSI 文件的根目录。
        only_dx1: 如果为 True，则仅包含文件名中含有 'DX1' 的切片。
        extensions: 需要包含的文件扩展名。

    Returns:
        映射 case_id -> 排序后的切片路径列表的字典。
    """
    root = Path(wsi_root)
    if not root.exists():
        logger.error(f"WSI root directory not found: {wsi_root}")
        raise FileNotFoundError(f"WSI root directory not found: {wsi_root}")

    # 查找所有 WSI 文件
    all_svs = []
    for ext in extensions:
        all_svs.extend(root.rglob(f"*{ext}"))
        all_svs.extend(root.rglob(f"*{ext.upper()}"))

    # 去重
    all_svs = sorted(set(all_svs))
    logger.info(f"Found {len(all_svs)} WSI files under {wsi_root}")

    # 过滤 DX1
    if only_dx1:
        dx1_svs = [p for p in all_svs if "DX1" in p.name.upper()]
        logger.info(
            f"DX1 filter: {len(dx1_svs)} slides retained "
            f"(removed {len(all_svs) - len(dx1_svs)})"
        )
        all_svs = dx1_svs

    if len(all_svs) == 0:
        logger.warning(f"No WSI files found in {wsi_root}")
        return {}

    # 按病例 ID 分组
    case_to_slides = build_case_to_slides(all_svs)

    return case_to_slides
