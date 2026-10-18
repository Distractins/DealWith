# -*- coding: utf-8 -*-
"""
datasets.py
============================================================================
PyTorch数据集定义 - 病人级弱监督Histomolecular数据集。

任务: 用WSI patch图像预测肠分化/粘液相关RNA-seq基因表达。

数据格式:
    每个样本 = 一个病人 (固定N张patch) + 可选协变量 + 基因标签/表达值。
    病人级弱监督: N个patch通过PathNet提取特征后均值池化得到病人级病理表示。

标签:
    - task_type=classification: g 为高/低表达二分类标签 (int, 0/1)
    - task_type=regression    : g 为连续表达量 (float)

返回元组 (兼容旧接口, 保留e/t用于附属生存分析):
    (x_path, x_grph, x_omic, e, t, g)
      x_path  : [N, 3, H, W] 或空张量
      x_grph  : 空占位张量 (兼容)
      x_omic  : [D] 协变量 或空张量 (mode不含omic时)
      e       : 删失标记 (附属生存分析用)
      t       : 生存时间 (附属生存分析用)
      g       : 基因标签/表达值

使用示例:
    from src.data_loading.datasets import PathomicDataset, build_dataloader
    dataset = PathomicDataset(config, data_split, split="train", mode="path")
    loader = build_dataloader(config, data_split, split="train", mode="path", shuffle=True)
============================================================================
"""

import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# 基础工具函数
# ============================================================

def _empty_tensor() -> torch.Tensor:
    return torch.zeros(1, dtype=torch.float32)


def _validate_mode(mode: str) -> None:
    valid_modes = {"path", "omic", "pathomic"}
    if mode not in valid_modes:
        raise ValueError(
            f"不支持的数据集模式: '{mode}'。支持的模式: {sorted(valid_modes)}"
        )


def _to_float_tensor(x) -> torch.Tensor:
    return torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)


def _to_long_tensor(x) -> torch.Tensor:
    try:
        return torch.tensor(np.asarray(x), dtype=torch.long)
    except (TypeError, ValueError):
        return torch.tensor(0, dtype=torch.long)


# ============================================================
# 主数据集类
# ============================================================

class PathomicDataset(Dataset):
    """
    病人级Histomolecular数据集 (原始patch图像输入)。

    参数:
        config: ConfigBundle配置对象
        data: 数据字典 {"train": {...}, "test": {...}}
        split: 数据集划分 ("train" 或 "test")
        mode: 数据模式 ("path" / "omic" / "pathomic")
    """

    def __init__(self, config, data: Dict, split: str = "train", mode: str = "path"):
        _validate_mode(mode)

        if split not in data:
            raise KeyError(f"划分 '{split}' 不在数据字典中。可用键: {list(data.keys())}")

        self.X_path = data[split]["x_path"]
        self.X_omic = data[split].get("x_omic", np.zeros((len(self.X_path), 0), dtype=np.float32))
        self.e = data[split].get("e", np.zeros(len(self.X_path), dtype=np.float32))
        self.t = data[split].get("t", np.zeros(len(self.X_path), dtype=np.float32))
        self.g = data[split]["g"]

        self.mode = mode
        self.split = split
        self.task_type = str(config.model.task_type).lower()

        self.input_size = config.model.path.input_size
        self.num_patches = config.upstream.num_patches_per_patient

        n = len(self.X_path)
        if not (len(self.X_omic) == len(self.e) == len(self.t) == len(self.g) == n):
            raise ValueError(
                f"数据长度不一致: split={split}, x_path={n}, x_omic={len(self.X_omic)}, "
                f"e={len(self.e)}, t={len(self.t)}, g={len(self.g)}"
            )

        self.transforms = self._build_transforms()

    def _build_transforms(self):
        if self.split == "train":
            tfms = [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
            return transforms.Compose(tfms)
        else:
            tfms = [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
            return transforms.Compose(tfms)

    def _load_patch_bag(self, patch_list) -> torch.Tensor:
        patch_list = list(patch_list)
        if len(patch_list) != self.num_patches:
            raise ValueError(
                f"每个病人必须有恰好 {self.num_patches} 个patch，但收到 {len(patch_list)} 个"
            )

        imgs = []
        for img_path in patch_list:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Patch图像文件不存在: {img_path}")
            with Image.open(img_path) as img:
                img = img.convert("RGB").copy()
            imgs.append(self.transforms(img))

        return torch.stack(imgs, dim=0)  # [N, 3, H, W]

    def __getitem__(self, index: int) -> Tuple:
        single_e = _to_float_tensor(self.e[index])
        single_t = _to_float_tensor(self.t[index])

        # 基因标签: 分类->long, 回归->float
        if self.task_type == "regression":
            single_g = _to_float_tensor(self.g[index])
        else:
            single_g = _to_long_tensor(self.g[index])

        x_grph = _empty_tensor()

        if self.mode in ("path", "pathomic"):
            single_X_path = self._load_patch_bag(self.X_path[index])
        else:
            single_X_path = _empty_tensor()

        if self.mode in ("omic", "pathomic"):
            single_X_omic = _to_float_tensor(self.X_omic[index])
        else:
            single_X_omic = _empty_tensor()

        return (single_X_path, x_grph, single_X_omic, single_e, single_t, single_g)

    def __len__(self) -> int:
        return len(self.X_path)


# ============================================================
# 数据集构建辅助函数
# ============================================================

def build_dataloader(config, data: Dict, split: str, mode: str,
                     shuffle: bool = False, for_test: bool = False):
    dataset = PathomicDataset(config, data, split=split, mode=mode)

    if for_test:
        num_workers, pin_memory, prefetch = 0, False, None
    else:
        num_workers = config.training.num_workers
        pin_memory = config.training.pin_memory
        prefetch = config.training.prefetch_factor if num_workers > 0 else None

    kwargs = {
        "batch_size": config.training.batch_size,
        "shuffle": shuffle,
        "drop_last": (split == "train"),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0 and prefetch is not None:
        kwargs["prefetch_factor"] = prefetch

    return torch.utils.data.DataLoader(dataset, **kwargs)


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PathomicDataset 数据集模块自测")
    print("=" * 60)

    N_patients, N_patches = 8, 3
    mock_x_path = [[f"dummy_{i}_{j}.png" for j in range(N_patches)] for i in range(N_patients)]
    mock_x_omic = np.random.randn(N_patients, 4).astype(np.float32)
    mock_e = np.random.binomial(1, 0.3, N_patients).astype(np.float32)
    mock_t = np.random.randint(30, 3650, N_patients).astype(np.float32)
    mock_g = np.random.choice([0, 1], N_patients)

    mock_data = {"train": {
        "x_path": mock_x_path, "x_omic": mock_x_omic,
        "e": mock_e, "t": mock_t, "g": mock_g,
    }}

    # 用一个最小配置对象驱动
    class _P:
        input_size = 224
        pretrained = ""
    class _Up:
        num_patches_per_patient = N_patches
    class _M:
        task_type = "classification"
        path = _P()
    class _C:
        model = _M()
        upstream = _Up()

    ds = PathomicDataset(_C(), mock_data, split="train", mode="path")
    print(f"  数据集长度: {len(ds)}")
    print(f"  单样本字段数: {len(ds[0])}")

    print("\n数据集结构验证通过!")
