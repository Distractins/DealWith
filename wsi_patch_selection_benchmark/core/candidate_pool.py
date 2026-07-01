# -*- coding: utf-8 -*-
"""统一的候选池构建器 —— 供所有采样器共享。

每个采样器都接收一个预构建的 CandidatePool。没有采样器会独立地重新
扫描 WSI。这是关键的架构设计决策。

流水线: WSI -> 组织掩膜 -> 网格扫描 -> 预筛选（基于组织比例）
          -> 读取图像碎片 -> QC + 评分 -> CandidatePool

从原始脚本的第 [F] 节和第 [H] 节迁移而来。
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import openslide

from common.dataclasses import (
    CandidatePatch,
    Patch,
    QualityMetrics,
    ScorePack,
    FeatureVector,
    ConfigBundle,
)
from common.enums import QCLevel
from core.tissue_mask import build_hybrid_tissue_mask, extract_largest_connected_component
from core.feature_extraction import (
    tissue_ratio_in_patch,
    patch_quality_metrics,
    patch_feature_vector,
)
from core.scoring import (
    patch_passes_qc_by_level,
    compute_final_score,
)
from utils.seed import stable_int_hash

logger = logging.getLogger(__name__)


# ============================================================
# 网格位置生成
# ============================================================

def generate_grid_positions(
    w0: int,
    h0: int,
    patch_size: int,
    stride: int,
) -> List[Tuple[int, int]]:
    """生成滑动窗口网格的左上角坐标位置。

    Args:
        w0: 第 0 级下切片的宽度。
        h0: 第 0 级下切片的高度。
        patch_size: 图像碎片的边长。
        stride: 相邻图像碎片之间的步长。

    Returns:
        (x0, y0) 位置的列表。
    """
    xs = list(range(0, max(1, w0 - patch_size + 1), stride))
    ys = list(range(0, max(1, h0 - patch_size + 1), stride))

    if len(xs) == 0:
        xs = [0]
    if len(ys) == 0:
        ys = [0]

    positions = []
    for y0 in ys:
        for x0 in xs:
            positions.append((x0, y0))
    return positions


# ============================================================
# 按优先级合并候选
# ============================================================

def merge_candidates_by_priority(
    candidate_pack: Dict[str, List[CandidatePatch]],
    target_k: int,
) -> Tuple[List[CandidatePatch], str]:
    """按优先级合并不同 QC 层级的候选。

    优先顺序为 strict > medium > relaxed > fallback。
    返回按分数降序排列的扁平列表。

    Args:
        candidate_pack: 字典，其键为 'strict'、'medium'、
            'relaxed'、'fallback'，映射到 CandidatePatch 列表。
        target_k: 所需的图像碎片目标数量。

    Returns:
        (合并排序后的列表, 所选层级字符串) 元组。
    """
    strict = candidate_pack.get("strict_candidates", [])
    medium = candidate_pack.get("medium_candidates", [])
    relaxed = candidate_pack.get("relaxed_candidates", [])
    fallback = candidate_pack.get("fallback_candidates", [])

    if len(strict) >= target_k:
        return strict, "strict"

    merged = strict + medium
    if len(merged) >= target_k:
        merged = sorted(merged, key=lambda x: x.score_for_ranking, reverse=True)
        return merged, "medium"

    merged = strict + medium + relaxed
    if len(merged) >= target_k:
        merged = sorted(merged, key=lambda x: x.score_for_ranking, reverse=True)
        return merged, "relaxed"

    merged = strict + medium + relaxed + fallback
    merged = sorted(merged, key=lambda x: x.score_for_ranking, reverse=True)
    return merged, "fallback"


# ============================================================
# 候选池（CandidatePool）
# ============================================================

@dataclass
class CandidatePool:
    """单张切片的统一候选池。

    为每张切片构建一次，供该切片的所有采样器共享。
    提供按算法特定需求筛选的过滤视图。

    Attributes:
        slide_base: 切片的基础名称。
        candidates: 所有通过 QC 的候选。
        tissue_mask: 二值组织掩膜（下采样后）。
        ds: 掩膜的下采样因子。
        largest_mask: 最大连通分量的可选掩膜。
    """
    slide_base: str
    candidates: List[CandidatePatch] = field(default_factory=list)
    tissue_mask: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1), dtype=np.uint8)
    )
    ds: int = 32
    largest_mask: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.candidates)

    def __bool__(self) -> bool:
        return len(self.candidates) > 0

    def filtered_by_qc(self, level: QCLevel) -> List[CandidatePatch]:
        """按最低 QC 等级筛选候选。"""
        return [c for c in self.candidates if c.qc_level == level]

    def top_by_score(self, k: int) -> List[CandidatePatch]:
        """返回按分数降序排列的前 K 个候选。"""
        sorted_cands = sorted(
            self.candidates, key=lambda x: x.score_for_ranking, reverse=True
        )
        return sorted_cands[:k]

    def filter_tissue_ratio(self, min_ratio: float) -> List[CandidatePatch]:
        """按最低组织比例筛选候选。"""
        return [c for c in self.candidates if c.metrics.tissue_ratio >= min_ratio]

    def filter_by_mask_region(
        self, mask: np.ndarray, ds: int
    ) -> List[CandidatePatch]:
        """筛选中心点落入二值掩膜内的候选。"""
        filtered = []
        for c in self.candidates:
            mx = c.patch.cx // ds
            my = c.patch.cy // ds
            if 0 <= my < mask.shape[0] and 0 <= mx < mask.shape[1]:
                if mask[my, mx] > 0:
                    filtered.append(c)
        return filtered

    def to_feature_matrix(self) -> np.ndarray:
        """将所有特征向量堆叠成 (N, D) 矩阵。"""
        if len(self.candidates) == 0:
            return np.zeros((0, 8), dtype=np.float32)
        return np.stack([c.feature.values for c in self.candidates])

    def to_morph_feature_matrix(self) -> np.ndarray:
        """将所有形态学特征向量堆叠成 (N, D) 矩阵。"""
        morph_feats = []
        for c in self.candidates:
            if c.morph_feature is not None:
                morph_feats.append(c.morph_feature.values)
            else:
                morph_feats.append(c.feature.values)
        if len(morph_feats) == 0:
            return np.zeros((0, 12), dtype=np.float32)
        return np.stack(morph_feats)


# ============================================================
# 候选池构建器（CandidatePoolBuilder）
# ============================================================

class CandidatePoolBuilder:
    """为单张切片构建 CandidatePool。

    两阶段流水线:
    第一阶段: 网格扫描 -> 按组织比例预筛选 -> 取组织比例最高的前 N 个
    第二阶段: 读取图像碎片 -> QC 指标 -> 评分 -> 候选池

    同时支持随机候选生成（供 Sentinel 采样器使用）。
    """

    def __init__(self, config: ConfigBundle):
        """初始化构建器。

        Args:
            config: 配置包。
        """
        self.config = config

    def build(
        self,
        slide: openslide.OpenSlide,
        image_mask: np.ndarray,
        ds: int,
        slide_base: str,
    ) -> CandidatePool:
        """通过网格扫描构建候选池。

        第一阶段: 网格扫描，按组织比例预筛选。
        第二阶段: 读取高组织比例的图像碎片，计算质量指标和分数。

        Args:
            slide: OpenSlide 对象。
            image_mask: 组织掩膜（来自 build_hybrid_tissue_mask）。
            ds: 掩膜的下采样因子。
            slide_base: 切片的基础名称。

        Returns:
            已构建完毕、可供采样器使用的 CandidatePool。
        """
        w0, h0 = slide.dimensions
        patch_size = self.config.patch_size
        stride = self.config.stride
        preselect_topk = self.config.preselect_topk
        min_tissue = self.config.min_tissue_ratio_preselect

        # 第一阶段: 网格扫描 + 组织比例预筛选
        positions = generate_grid_positions(w0, h0, patch_size, stride)
        logger.debug(
            f"[{slide_base}] Grid positions: {len(positions)} "
            f"(stride={stride}, patch_size={patch_size})"
        )

        pre_candidates = []
        for x0, y0 in positions:
            tr = tissue_ratio_in_patch(image_mask, x0, y0, patch_size, ds)
            if tr >= min_tissue:
                pre_candidates.append((x0, y0, tr))

        if len(pre_candidates) == 0:
            logger.warning(f"[{slide_base}] No tissue candidates found")
            return CandidatePool(
                slide_base=slide_base,
                tissue_mask=image_mask,
                ds=ds,
            )

        # 按组织比例降序排列，取前 N 个
        pre_candidates = sorted(pre_candidates, key=lambda x: x[2], reverse=True)
        pre_candidates = pre_candidates[:preselect_topk]
        logger.debug(
            f"[{slide_base}] Preselected {len(pre_candidates)} candidates "
            f"(tissue_ratio >= {min_tissue})"
        )

        # 第二阶段: 读取图像碎片并计算完整指标
        candidates = []
        for x0, y0, tr in pre_candidates:
            try:
                patch_img = slide.read_region(
                    (x0, y0), 0, (patch_size, patch_size)
                ).convert("RGB")
                patch_np = np.array(patch_img)

                # 计算质量指标
                metrics_dict = patch_quality_metrics(
                    patch_np,
                    tissue_ratio=tr,
                    enable_gland_irregularity=self.config.enable_gland_irregularity,
                )

                # 计算最终分数
                score_dict = compute_final_score(
                    metrics=metrics_dict,
                    innovation_weight=self.config.innovation_weight,
                )

                # 构建特征向量
                feat_vec = patch_feature_vector(metrics_dict)

                # 确定 QC 等级
                qc_level = QCLevel.FALLBACK
                if patch_passes_qc_by_level(metrics_dict, "strict"):
                    qc_level = QCLevel.STRICT
                elif patch_passes_qc_by_level(metrics_dict, "medium"):
                    qc_level = QCLevel.MEDIUM
                elif patch_passes_qc_by_level(metrics_dict, "relaxed"):
                    qc_level = QCLevel.RELAXED

                # 构建 CandidatePatch
                patch = Patch(
                    x0=x0,
                    y0=y0,
                    cx=x0 + patch_size // 2,
                    cy=y0 + patch_size // 2,
                    patch_np=patch_np,
                    slide_base=slide_base,
                    slide_path="",  # 将由调用者设置
                )

                quality = QualityMetrics(**{
                    k: metrics_dict[k]
                    for k in QualityMetrics.__dataclass_fields__
                    if k in metrics_dict
                })

                scores = ScorePack(**{
                    k: score_dict[k]
                    for k in ScorePack.__dataclass_fields__
                    if k in score_dict
                })

                candidate = CandidatePatch(
                    patch=patch,
                    metrics=quality,
                    scores=scores,
                    feature=FeatureVector(values=feat_vec),
                    qc_level=qc_level,
                )
                candidates.append(candidate)

            except Exception as e:
                logger.debug(f"[{slide_base}] Error reading patch at ({x0},{y0}): {e}")
                continue

        # 先按 QC 等级、再按分数排序
        qc_order = {QCLevel.STRICT: 0, QCLevel.MEDIUM: 1,
                     QCLevel.RELAXED: 2, QCLevel.FALLBACK: 3}
        candidates = sorted(
            candidates,
            key=lambda c: (qc_order.get(c.qc_level, 3), -c.score_for_ranking),
        )

        # 构建最大连通分量掩膜
        largest_mask = extract_largest_connected_component(image_mask)

        pool = CandidatePool(
            slide_base=slide_base,
            candidates=candidates,
            tissue_mask=image_mask,
            ds=ds,
            largest_mask=largest_mask,
        )

        logger.info(
            f"[{slide_base}] CandidatePool built: {len(candidates)} candidates "
            f"(strict={sum(1 for c in candidates if c.qc_level == QCLevel.STRICT)}, "
            f"medium={sum(1 for c in candidates if c.qc_level == QCLevel.MEDIUM)}, "
            f"relaxed={sum(1 for c in candidates if c.qc_level == QCLevel.RELAXED)}, "
            f"fallback={sum(1 for c in candidates if c.qc_level == QCLevel.FALLBACK)})"
        )

        return pool

    def build_random(
        self,
        slide: openslide.OpenSlide,
        image_mask: np.ndarray,
        ds: int,
        slide_base: str,
    ) -> CandidatePool:
        """通过随机采样构建候选池（供 Sentinel 采样器使用）。

        从组织掩膜区域中随机采样位置，计算每个候选的质量指标和分数。

        Args:
            slide: OpenSlide 对象。
            image_mask: 组织掩膜。
            ds: 下采样因子。
            slide_base: 切片的基础名称。

        Returns:
            包含随机生成候选的 CandidatePool。
        """
        w0, h0 = slide.dimensions
        patch_size = self.config.patch_size
        max_tries = self.config.max_tries
        candidate_pool_size = self.config.candidate_pool_size
        innovation_weight = self.config.innovation_weight
        enable_gland = self.config.enable_gland_irregularity
        seed = self.config.seed

        # 从组织掩膜中获取组织像素坐标
        ys, xs = np.where(image_mask > 0)
        if len(xs) == 0:
            logger.warning(f"[{slide_base}] No tissue in mask for random sampling")
            return CandidatePool(
                slide_base=slide_base,
                tissue_mask=image_mask,
                ds=ds,
            )

        rng_seed = seed + stable_int_hash(slide_base, mod=10 ** 6)
        rng = random.Random(rng_seed)

        strict_candidates = []
        medium_candidates = []
        relaxed_candidates = []
        fallback_candidates = []

        tries = 0
        target_size = max(candidate_pool_size, self.config.patches_per_case * 12)

        while tries < max_tries and (
            len(strict_candidates) + len(medium_candidates)
            + len(relaxed_candidates)
        ) < target_size:
            tries += 1

            # 随机选取一个组织像素
            idx = rng.randrange(0, len(xs))
            mx, my = int(xs[idx]), int(ys[idx])
            cx, cy = mx * ds, my * ds

            x0 = int(cx - patch_size // 2)
            y0 = int(cy - patch_size // 2)

            if x0 < 0 or y0 < 0 or (x0 + patch_size) > w0 or (y0 + patch_size) > h0:
                continue

            tr = tissue_ratio_in_patch(image_mask, x0, y0, patch_size, ds)
            if tr <= 0.05:
                continue

            try:
                patch_img = slide.read_region(
                    (x0, y0), 0, (patch_size, patch_size)
                ).convert("RGB")
                patch_np = np.array(patch_img)

                metrics_dict = patch_quality_metrics(
                    patch_np,
                    tissue_ratio=tr,
                    enable_gland_irregularity=enable_gland,
                )

                score_dict = compute_final_score(
                    metrics=metrics_dict,
                    innovation_weight=innovation_weight,
                )

                feat_vec = patch_feature_vector(metrics_dict)

                # 构建候选
                patch = Patch(
                    x0=x0, y0=y0,
                    cx=x0 + patch_size // 2,
                    cy=y0 + patch_size // 2,
                    patch_np=patch_np,
                    slide_base=slide_base,
                    slide_path="",
                )
                quality = QualityMetrics(**{
                    k: metrics_dict[k]
                    for k in QualityMetrics.__dataclass_fields__
                    if k in metrics_dict
                })
                scores = ScorePack(**{
                    k: score_dict[k]
                    for k in ScorePack.__dataclass_fields__
                    if k in score_dict
                })
                candidate = CandidatePatch(
                    patch=patch,
                    metrics=quality,
                    scores=scores,
                    feature=FeatureVector(values=feat_vec),
                )

                # 按 QC 等级分类
                if patch_passes_qc_by_level(metrics_dict, "strict"):
                    candidate.qc_level = QCLevel.STRICT
                    strict_candidates.append(candidate)
                elif patch_passes_qc_by_level(metrics_dict, "medium"):
                    candidate.qc_level = QCLevel.MEDIUM
                    medium_candidates.append(candidate)
                elif patch_passes_qc_by_level(metrics_dict, "relaxed"):
                    candidate.qc_level = QCLevel.RELAXED
                    relaxed_candidates.append(candidate)
                elif metrics_dict["tissue_ratio"] >= 0.20:
                    candidate.qc_level = QCLevel.FALLBACK
                    fallback_candidates.append(candidate)

            except Exception as e:
                logger.debug(f"[{slide_base}] Error in random candidate: {e}")
                continue

        # 每层按分数降序排列
        strict_candidates.sort(key=lambda c: c.score_for_ranking, reverse=True)
        medium_candidates.sort(key=lambda c: c.score_for_ranking, reverse=True)
        relaxed_candidates.sort(key=lambda c: c.score_for_ranking, reverse=True)
        fallback_candidates.sort(key=lambda c: c.score_for_ranking, reverse=True)

        # 合并: 先 strict，再 medium，以此类推
        all_candidates = (
            strict_candidates + medium_candidates
            + relaxed_candidates + fallback_candidates
        )

        largest_mask = extract_largest_connected_component(image_mask)

        pool = CandidatePool(
            slide_base=slide_base,
            candidates=all_candidates,
            tissue_mask=image_mask,
            ds=ds,
            largest_mask=largest_mask,
        )

        logger.info(
            f"[{slide_base}] Random pool built: {len(all_candidates)} candidates "
            f"from {tries} attempts"
        )

        return pool
