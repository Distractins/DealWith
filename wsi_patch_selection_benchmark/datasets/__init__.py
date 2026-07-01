# -*- coding: utf-8 -*-
"""数据集层: WSI 文件发现与读取。

此模块仅负责读取 WSI 切片和发现 .svs 文件。
此处不包含任何算法逻辑。
"""

from datasets.wsi_reader import WSIReader
from datasets.wsi_scanner import discover_wsis

__all__ = ["WSIReader", "discover_wsis"]
