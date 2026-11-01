# -*- coding: utf-8 -*-
"""图像处理工具函数。

包含 rgb_to_od 和其他不特定于补丁特征提取的
通用图像辅助函数。
"""

import numpy as np


def rgb_to_od(rgb: np.ndarray) -> np.ndarray:
    """将 RGB 图像转换到光学密度（OD）空间。

    OD = -log((RGB + 1) / 256)
    该方法用于组织病理学中的组织-背景分离。

    Args:
        rgb: 输入 RGB 图像，类型为 uint8 的 ndarray。

    Returns:
        光学密度图像，类型为 float32 的 ndarray。
    """
    rgb = rgb.astype(np.float32)
    od = -np.log((rgb + 1.0) / 256.0)
    return od
