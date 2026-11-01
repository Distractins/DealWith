# -*- coding: utf-8 -*-
"""WSI补丁选择基准测试的通用数据结构。"""

from common.dataclasses import (
    Patch,
    QualityMetrics,
    ScorePack,
    FeatureVector,
    CandidatePatch,
    SlideInfo,
    SlideResult,
    CaseResult,
    AlgorithmResult,
    SlideRecord,
    PatchRecord,
    ConfigBundle,
)
from common.enums import QCLevel, AlgorithmName, Status
from common.constants import (
    QC_THRESHOLDS,
    BASELINE_SCORE_WEIGHTS,
    INNOVATION_SCORE_WEIGHTS,
    FEATURE_VECTOR_INDICES,
    NORMALIZE_SCORE_RANGES,
    QC_QUALITY_WEIGHTS,
    TUMOR_MORPH_WEIGHTS,
    METHOD_COLORS,
)

__all__ = [
    "Patch",
    "QualityMetrics",
    "ScorePack",
    "FeatureVector",
    "CandidatePatch",
    "SlideInfo",
    "SlideResult",
    "CaseResult",
    "AlgorithmResult",
    "SlideRecord",
    "PatchRecord",
    "ConfigBundle",
    "QCLevel",
    "AlgorithmName",
    "Status",
    "QC_THRESHOLDS",
    "BASELINE_SCORE_WEIGHTS",
    "INNOVATION_SCORE_WEIGHTS",
    "FEATURE_VECTOR_INDICES",
    "NORMALIZE_SCORE_RANGES",
    "QC_QUALITY_WEIGHTS",
    "TUMOR_MORPH_WEIGHTS",
    "METHOD_COLORS",
]
