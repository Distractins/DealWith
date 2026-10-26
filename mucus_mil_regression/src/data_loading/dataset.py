# -*- coding: utf-8 -*-
"""
dataset.py - MIL bag数据集 (一个病人=一个bag)。

每个样本: patch_bag [N, 3, H, W] + 临床特征 [D] + 回归标签 y (标量)。
N = num_patches_per_patient, 同一病人的所有patch作为实例, 仅bag级标签。
"""

import os
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MILBagDataset(Dataset):
    def __init__(self, config, data: Dict, split: str = "train", need_patches: bool = True):
        if split not in data:
            raise KeyError(f"划分 '{split}' 不存在: {list(data.keys())}")
        self.x_path = data[split]["x_path"]          # List[List[str]]
        self.x_clinical = data[split].get("x_clinical", np.zeros((len(self.x_path), 0), dtype=np.float32))
        self.y = data[split]["y"]                     # np.ndarray [N] float
        self.patient_ids = data[split].get("patient_ids", [])

        self.split = split
        self.patch_size = config.upstream.patch_size
        self.num_patches = config.upstream.num_patches_per_patient
        self.need_patches = need_patches

        n = len(self.x_path)
        assert len(self.x_clinical) == n and len(self.y) == n, "数据长度不一致"

        self.transforms = self._build_transforms()

    def _build_transforms(self):
        base = [transforms.Resize((self.patch_size, self.patch_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        if self.split == "train":
            base = [transforms.Resize((self.patch_size, self.patch_size)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(base)

    def _load_bag(self, patch_list) -> torch.Tensor:
        patch_list = list(patch_list)
        if len(patch_list) != self.num_patches:
            raise ValueError(f"每个bag需 {self.num_patches} 张patch, 实际 {len(patch_list)}")
        imgs = []
        for p in patch_list:
            if not os.path.exists(p):
                raise FileNotFoundError(f"patch不存在: {p}")
            with Image.open(p) as img:
                img = img.convert("RGB").copy()
            imgs.append(self.transforms(img))
        return torch.stack(imgs, dim=0)  # [N, 3, H, W]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.need_patches:
            bag = self._load_bag(self.x_path[idx])
        else:
            bag = torch.zeros(self.num_patches, 3, self.patch_size, self.patch_size, dtype=torch.float32)
        clinical = torch.tensor(self.x_clinical[idx], dtype=torch.float32)
        y = torch.tensor(float(self.y[idx]), dtype=torch.float32)
        return bag, clinical, y

    def __len__(self) -> int:
        return len(self.x_path)


def build_dataloader(config, data: Dict, split: str, shuffle: bool = False, need_patches: bool = True):
    ds = MILBagDataset(config, data, split=split, need_patches=need_patches)
    return DataLoader(
        ds,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        drop_last=(split == "train"),
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
