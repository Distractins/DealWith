# -*- coding: utf-8 -*-
"""
datasets.py
============================================================================
PyTorch数据集定义 - 病人级弱监督多模态数据集。

从源项目 data_loaders.py 重构而来，主要改动:
    1. 去掉grph(图网络)模式，简化为path/omic/pathomic三种模式
    2. 去掉FastDatasetLoader（预提取特征版本），仅保留原始图像版本
    3. 支持从配置中读取上游patch路径和基因组CSV
    4. 所有路径通过配置管理，无硬编码
    5. 添加完整的图像预处理流水线集成
    6. 添加中文注释

数据格式:
    每个病人 = 固定数量(N)的patch图像 + 1个基因组特征向量 + 生存标签

病人级弱监督:
    训练时每个样本是"病人"，不是单张patch。
    N个patch通过PathNet提取特征后进行均值池化得到病人级病理表示。

使用示例:
    from src.data_loading.datasets import PathomicDataset
    dataset = PathomicDataset(config, data_split, split="train", mode="pathomic")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
============================================================================
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# 允许加载截断的图像（处理损坏的patch）
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# 基础工具函数
# ============================================================

def _empty_tensor() -> torch.Tensor:
    """创建空的占位张量"""
    return torch.zeros(1, dtype=torch.float32)


def _validate_mode(mode: str) -> None:
    """验证数据集模式是否合法"""
    valid_modes = {"path", "omic", "pathomic"}
    if mode not in valid_modes:
        raise ValueError(
            f"不支持的数据集模式: '{mode}'。支持的模式: {sorted(valid_modes)}"
        )


def _to_float_tensor(x) -> torch.Tensor:
    """安全转换为float32张量"""
    return torch.tensor(x, dtype=torch.float32)


def _to_long_tensor(x) -> torch.Tensor:
    """安全转换为int64张量"""
    return torch.tensor(x, dtype=torch.long)


# ============================================================
# N分期标签映射（用于ncls任务）
# ============================================================

def _normalize_n_stage(raw_label) -> str:
    """清洗N分期标签字符串"""
    if raw_label is None:
        return ""
    s = str(raw_label).strip().upper().replace(" ", "")
    return s


def _map_n_stage_to_3class(raw_label) -> int:
    """
    N分期三分类映射:
        N0 -> 0
        N1/N1a/N1b/N1c -> 1
        N2/N2a/N2b -> 2
    """
    s = _normalize_n_stage(raw_label)
    mapping = {
        "N0": 0,
        "N1": 1, "N1A": 1, "N1B": 1, "N1C": 1,
        "N2": 2, "N2A": 2, "N2B": 2,
    }
    if s not in mapping:
        raise ValueError(
            f"不支持的N分期标签: raw_label={raw_label}, normalized={s}。"
            f"期望: {sorted(mapping.keys())}"
        )
    return mapping[s]


def _convert_labels_by_task(task: str, label_mode: str, labels: List) -> List:
    """
    根据任务配置转换标签格式。

    参数:
        task: 任务类型 ("surv" / "ncls")
        label_mode: 标签模式 ("n_binary" / "n_3class")
        labels: 原始标签列表

    返回:
        List[int]: 转换后的整数标签列表
    """
    task = str(task).lower()
    label_mode = str(label_mode).lower()
    converted = []

    for g in labels:
        if task == "ncls":
            if label_mode == "n_3class":
                converted.append(_map_n_stage_to_3class(g))
            else:
                try:
                    converted.append(int(g))
                except Exception:
                    raise ValueError(
                        f"标签转换失败: label_mode={label_mode}, raw_label={g}"
                    )
        else:
            # surv任务：保持原值
            try:
                converted.append(int(g))
            except Exception:
                converted.append(g)

    return converted


# ============================================================
# 主数据集类
# ============================================================

class PathomicDataset(Dataset):
    """
    病人级多模态数据集（原始patch图像输入版本）。

    每个样本包含:
        - x_path: [N, 3, H, W] N个patch图像
        - x_omic: [input_dim] 基因组特征向量
        - e: 删失标记 (1=事件, 0=删失)
        - t: 生存时间
        - g: 标签（surv时为额外标记，ncls时为类别标签）

    train时的数据增强:
        - 随机水平/垂直翻转
        - 归一化 (mean=0.5, std=0.5)

    test时的处理:
        - 仅归一化，不做数据增强

    参数:
        config: ConfigBundle配置对象
        data: 数据字典 {"train": {...}, "test": {...}}
        split: 数据集划分 ("train" 或 "test")
        mode: 数据模式 ("path" / "omic" / "pathomic")
    """

    def __init__(self, config, data: Dict, split: str = "train", mode: str = "pathomic"):
        _validate_mode(mode)

        if split not in data:
            raise KeyError(
                f"划分 '{split}' 不在数据字典中。"
                f"可用键: {list(data.keys())}"
            )

        # 从数据字典中提取各字段
        self.X_path = data[split]["x_path"]    # 病人patch路径列表 (list of list)
        self.X_omic = data[split]["x_omic"]    # 基因组特征 (numpy array)
        self.e = data[split]["e"]               # 删失标记
        self.t = data[split]["t"]               # 生存时间

        # 标签处理
        self.g_raw = data[split]["g"]
        task = config.model.task
        label_mode = getattr(config.model, "label_mode", "n_binary")
        self.g = _convert_labels_by_task(task, label_mode, self.g_raw)

        self.mode = mode
        self.split = split

        # 配置参数
        self.input_size = config.model.path.input_size  # 1024
        self.num_patches = config.upstream.num_patches_per_patient  # 6
        self.force_resize = False  # 默认不做强制缩放

        # 长度一致性检查
        if not (len(self.X_path) == len(self.X_omic) == len(self.e) == len(self.t) == len(self.g)):
            raise ValueError(
                f"数据长度不一致: split={split}, "
                f"x_path={len(self.X_path)}, x_omic={len(self.X_omic)}, "
                f"e={len(self.e)}, t={len(self.t)}, g={len(self.g)}"
            )

        # 构建数据变换
        self.transforms = self._build_transforms()

    def _build_transforms(self):
        """根据split构建数据增强/预处理变换"""
        if self.split == "train":
            # 训练集: 添加数据增强
            tfms = []
            if self.force_resize:
                tfms.append(transforms.Resize((self.input_size, self.input_size)))
            tfms.extend([
                transforms.RandomHorizontalFlip(0.5),    # 随机水平翻转
                transforms.RandomVerticalFlip(0.5),      # 随机垂直翻转
                transforms.ToTensor(),                    # HWC -> CHW, [0,255] -> [0,1]
                transforms.Normalize(                     # 归一化到[-1,1]
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),
            ])
            return transforms.Compose(tfms)
        else:
            # 验证/测试集: 无数据增强
            tfms = []
            if self.force_resize:
                tfms.append(transforms.Resize((self.input_size, self.input_size)))
            tfms.extend([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            return transforms.Compose(tfms)

    def _load_patch_bag(self, patch_list: List[str]) -> torch.Tensor:
        """
        加载一个病人的N个patch图像。

        参数:
            patch_list: 长度为num_patches的路径列表

        返回:
            bag_tensor: [N, 3, H, W] patch图像批次

        异常:
            FileNotFoundError: patch图像文件不存在
            RuntimeError: 图像加载失败
        """
        if not isinstance(patch_list, (list, tuple)):
            raise TypeError(
                f"patch_list应为list/tuple，实际类型: {type(patch_list)}"
            )

        patch_list = list(patch_list)
        if len(patch_list) != self.num_patches:
            raise ValueError(
                f"每个病人必须有恰好 {self.num_patches} 个patch，"
                f"但收到了 {len(patch_list)} 个"
            )

        imgs = []
        for img_path in patch_list:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Patch图像文件不存在: {img_path}")

            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB").copy()
                img = self.transforms(img)
                imgs.append(img)
            except Exception as e:
                raise RuntimeError(f"加载图像失败: {img_path}, 错误: {e}")

        bag_tensor = torch.stack(imgs, dim=0)  # [N, 3, H, W]
        return bag_tensor

    def __getitem__(self, index: int) -> Tuple:
        """
        获取第index个病人的数据。

        返回:
            (x_path, x_grph, x_omic, e, t, g) 元组
            - x_path: [N, 3, H, W] 或空张量（非path模式）
            - x_grph: 空张量（占位，兼容旧接口）
            - x_omic: [input_dim] 或空张量（非omic模式）
            - e: 删失标记
            - t: 生存时间
            - g: 标签
        """
        single_e = _to_float_tensor(self.e[index])
        single_t = _to_float_tensor(self.t[index])
        single_g = _to_long_tensor(self.g[index])

        # 占位张量（兼容旧接口）
        x_grph = _empty_tensor()

        # 加载patch图像
        if self.mode in ("path", "pathomic"):
            patch_list = self.X_path[index]
            single_X_path = self._load_patch_bag(patch_list)
        else:
            single_X_path = _empty_tensor()

        # 加载基因组特征
        if self.mode in ("omic", "pathomic"):
            single_X_omic = _to_float_tensor(self.X_omic[index])
        else:
            single_X_omic = _empty_tensor()

        return (single_X_path, x_grph, single_X_omic, single_e, single_t, single_g)

    def __len__(self) -> int:
        """返回病人总数"""
        return len(self.X_path)


# ============================================================
# 数据集构建辅助函数
# ============================================================

def build_dataloader(config, data: Dict, split: str, mode: str,
                     shuffle: bool = False, for_test: bool = False):
    """
    构建DataLoader的便捷函数。

    参数:
        config: ConfigBundle配置对象
        data: 数据字典
        split: "train" 或 "test"
        mode: "path" / "omic" / "pathomic"
        shuffle: 是否打乱数据
        for_test: 是否为测试模式（禁用多worker）

    返回:
        DataLoader实例
    """
    dataset = PathomicDataset(config, data, split=split, mode=mode)

    if for_test:
        # 测试模式: 单worker, 无pin_memory
        num_workers = 0
        pin_memory = False
        persistent = False
        prefetch = None
    else:
        num_workers = config.training.num_workers
        pin_memory = config.training.pin_memory
        persistent = False
        prefetch = config.training.prefetch_factor if num_workers > 0 else None

    kwargs = {
        "batch_size": config.training.batch_size,
        "shuffle": shuffle,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = False
        if prefetch is not None:
            kwargs["prefetch_factor"] = prefetch

    return torch.utils.data.DataLoader(dataset, **kwargs)


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PathomicDataset 数据集模块自测")
    print("=" * 60)

    # 模拟数据
    N_patients = 20
    N_patches = 6

    # 模拟patch路径（实际使用时这些路径指向真实的PNG文件）
    mock_x_path = [
        [f"dummy_patch_{i}_{j}.png" for j in range(N_patches)]
        for i in range(N_patients)
    ]
    mock_x_omic = np.random.randn(N_patients, 356).astype(np.float32)
    mock_e = np.random.binomial(1, 0.3, N_patients).astype(np.float32)
    mock_t = np.random.randint(30, 3650, N_patients).astype(np.float32)
    mock_g = np.random.choice(["N0", "N1", "N2"], N_patients)

    mock_data = {
        "train": {
            "x_path": mock_x_path,
            "x_omic": mock_x_omic,
            "e": mock_e,
            "t": mock_t,
            "g": mock_g,
        }
    }

    print(f"\n  病人数: {N_patients}")
    print(f"  每病人patch数: {N_patches}")
    print(f"  基因组特征维度: {mock_x_omic.shape[1]}")
    print(f"  删失比例: {1 - mock_e.mean():.1%}")

    # 验证数据长度一致性
    lengths = {
        "x_path": len(mock_data["train"]["x_path"]),
        "x_omic": len(mock_data["train"]["x_omic"]),
        "e": len(mock_data["train"]["e"]),
        "t": len(mock_data["train"]["t"]),
        "g": len(mock_data["train"]["g"]),
    }
    print(f"\n  各字段长度: {lengths}")
    assert len(set(lengths.values())) == 1, "数据长度不一致!"

    print("\n  数据集结构验证通过!")
