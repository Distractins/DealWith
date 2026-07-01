# -*- coding: utf-8 -*-
"""WSI补丁选择基准测试的枚举定义。"""

from enum import Enum


class QCLevel(str, Enum):
    """候选补丁的质量控制级别。"""

    STRICT = "strict"
    MEDIUM = "medium"
    RELAXED = "relaxed"
    FALLBACK = "fallback"


class AlgorithmName(str, Enum):
    """所有支持的补丁选择算法的唯一标识符。"""

    RANDOM = "random"
    GRID = "grid"
    LARGEST_TISSUE = "largest_tissue"
    STRATIFIED = "stratified"
    KMEANS = "kmeans"
    YOTTIXEL = "yottixel"
    SPLICE = "splice"
    SDM = "sdm"
    SENTINEL = "sentinel"


class Status(str, Enum):
    """切片和病例的处理状态。"""

    OK = "ok"
    NO_TISSUE = "no_tissue"
    NO_CANDIDATE = "no_candidate"
    OPEN_FAILED = "open_failed"
    PROCESSING_FAILED = "processing_failed"
    SKIPPED = "skipped_no_need_for_case"
    NO_PATCH_SAVED = "no_patch_saved"
    PARTIAL = "partial"
