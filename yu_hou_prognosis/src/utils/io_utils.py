# -*- coding: utf-8 -*-
"""
io_utils.py
============================================================================
通用IO工具函数。

功能:
    1. 安全的路径创建
    2. 文件存在性检查
    3. JSON/Pickle序列化
    4. 大文件安全复制
    5. 目录大小计算

使用示例:
    from src.utils.io_utils import ensure_dir, safe_save_pickle
    ensure_dir("experiments/default/ckpt")
    safe_save_pickle(data, "experiments/default/results/cv_results.pkl")
============================================================================
"""

import os
import pickle
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建。

    参数:
        path: 目录路径（字符串或Path对象）

    返回:
        Path: 创建的目录路径

    注意:
        与os.makedirs(exist_ok=True)相同，但返回Path对象便于链式调用。
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_exists(path: Union[str, Path]) -> bool:
    """
    检查文件是否存在且为普通文件。

    参数:
        path: 文件路径

    返回:
        bool: 文件存在且为普通文件返回True
    """
    return Path(path).is_file()


def dir_exists(path: Union[str, Path]) -> bool:
    """
    检查目录是否存在。

    参数:
        path: 目录路径

    返回:
        bool: 目录存在返回True
    """
    return Path(path).is_dir()


def safe_save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """
    安全保存对象为pickle文件（先写临时文件，再原子替换）。

    参数:
        obj: 要保存的Python对象
        path: 目标文件路径
    """
    path = Path(path)
    ensure_dir(path.parent)

    # 先写入临时文件
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        # 原子替换
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def safe_load_pickle(path: Union[str, Path]) -> Any:
    """
    安全加载pickle文件。

    参数:
        path: pickle文件路径

    返回:
        Any: 加载的Python对象

    异常:
        FileNotFoundError: 文件不存在
        pickle.UnpicklingError: 文件格式错误
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Pickle文件不存在: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def safe_save_json(obj: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    保存对象为JSON文件。

    参数:
        obj: 要保存的对象（必须可JSON序列化）
        path: 目标文件路径
        indent: JSON缩进空格数
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=str)


def safe_load_json(path: Union[str, Path]) -> Any:
    """
    加载JSON文件。

    参数:
        path: JSON文件路径

    返回:
        Any: 加载的对象
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def copy_file_safe(src: Union[str, Path], dst: Union[str, Path],
                   overwrite: bool = False) -> bool:
    """
    安全复制文件。

    参数:
        src: 源文件路径
        dst: 目标文件路径
        overwrite: 如果目标文件已存在，是否覆盖

    返回:
        bool: 复制成功返回True，跳过返回False

    异常:
        FileNotFoundError: 源文件不存在
    """
    src = Path(src)
    dst = Path(dst)

    if not src.is_file():
        raise FileNotFoundError(f"源文件不存在: {src}")

    if dst.exists() and not overwrite:
        return False

    ensure_dir(dst.parent)
    shutil.copy2(str(src), str(dst))
    return True


def get_dir_size(path: Union[str, Path]) -> int:
    """
    递归计算目录总大小（字节）。

    参数:
        path: 目录路径

    返回:
        int: 目录大小（字节）
    """
    path = Path(path)
    if not path.is_dir():
        return 0

    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def format_size(size_bytes: int) -> str:
    """
    将字节数格式化为人类可读的大小字符串。

    参数:
        size_bytes: 字节数

    返回:
        str: 格式化后的大小字符串（如 "1.23 GB"）
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def list_files_recursive(path: Union[str, Path],
                         pattern: str = "*") -> List[Path]:
    """
    递归列出目录下所有匹配模式的文件。

    参数:
        path: 目录路径
        pattern: 文件匹配模式（glob格式，如 "*.png", "*.csv"）

    返回:
        List[Path]: 匹配的文件路径列表（按修改时间排序）
    """
    path = Path(path)
    if not path.is_dir():
        return []

    files = list(path.rglob(pattern))
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return files


def count_files_by_ext(path: Union[str, Path],
                       extensions: List[str] = None) -> Dict[str, int]:
    """
    统计目录下各扩展名文件的数量。

    参数:
        path: 目录路径
        extensions: 要统计的扩展名列表（如 [".png", ".csv"]）。为None时统计所有类型。

    返回:
        Dict[str, int]: 扩展名->文件数量 的映射
    """
    path = Path(path)
    if not path.is_dir():
        return {}

    counts = {}
    for f in path.rglob("*"):
        if f.is_file():
            ext = f.suffix.lower()
            if extensions is None or ext in extensions:
                counts[ext] = counts.get(ext, 0) + 1

    return counts
