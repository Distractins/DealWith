# -*- coding: utf-8 -*-
"""Sentinel采样器：Sentinel感知候选块选择（Sentinel-aware Patch Selection, SAPS）。

迁移自：dataprocessing/wsi_patch/wsi_to_patch_idea.py

保留的原始创新点：
1. 联合质量-形态评分
2. 通过 tumor_bias 指标实现肿瘤偏向选择
3. 多重约束选择：质量 + 空间多样性 + 特征多样性

策略：
1. 通过从组织掩码区域随机采样生成候选池
   （所有其他采样器均采用基于网格的方式）
2. 应用三级QC过滤（严格 > 中等 > 宽松 > 回退）
3. 按QC优先级合并候选块
4. 施加空间+特征多样性约束
5. 选择 Top-K 个多样性候选块

与 Grid 采样器的关键区别：
- 随机候选生成，而非均匀网格扫描
- 更大的 candidate_pool_size 以增加探索范围
- 在 final_score 排序中使用 innovation_weight
"""

import logging
from typing import List

from common.dataclasses import CandidatePatch, ConfigBundle
from core.candidate_pool import CandidatePool
from core.diversity import select_diverse_topk
from samplers.base_sampler import BaseSampler
from samplers import register_sampler

logger = logging.getLogger(__name__)


@register_sampler
class SentinelSampler(BaseSampler):
    """Sentinel感知候选块选择（SAPS）。

    与 Grid/LargestTissue 的关键区别在于 Sentinel 使用
    随机候选生成（而非网格扫描），从而探索更多样化的
    空间位置。结合三级QC过滤和多样性约束，能够产生
    具有诊断相关性的区域选择。

    注意：应使用 CandidatePoolBuilder.build_random() 方法
    （而非 build()）为此采样器生成候选池。这由
    run_patch_selection.py 中的流水线编排器负责处理。
    """

    name = "Sentinel (SAPS)"

    @staticmethod
    def algorithm_name() -> str:
        return "sentinel"

    def select_patches(
        self,
        candidate_pool: CandidatePool,
        num_patches: int,
    ) -> List[CandidatePatch]:
        """通过 Sentinel 感知选择方式选择候选块。

        Sentinel 算法期望候选块已按
        final_score = baseline + innovation_weight * innovation 进行评分。
        多样性选择随后按 final_score（而非仅 baseline_score）排序，
        赋予肿瘤形态学更大的权重。

        Args:
            candidate_pool: 预构建的候选池（应通过 build_random()
                构建以实现随机候选生成）。
            num_patches: 目标数量（K）。

        Returns:
            已选的带肿瘤偏向的多样性高质量候选块列表。
        """
        if len(candidate_pool) == 0:
            logger.warning(
                f"[{candidate_pool.slide_base}] Sentinel: empty pool"
            )
            return []

        # 按 final_score 排序（包含 innovation/形态学）
        # 这是关键区别：Sentinel 使用 innovation_weight > 0
        sorted_candidates = sorted(
            candidate_pool.candidates,
            key=lambda c: c.scores.final_score,
            reverse=True,
        )

        logger.debug(
            f"[{candidate_pool.slide_base}] Sentinel: "
            f"{len(sorted_candidates)} candidates sorted by final_score "
            f"(strict={sum(1 for c in sorted_candidates if c.qc_level.value == 'strict')}, "
            f"medium={sum(1 for c in sorted_candidates if c.qc_level.value == 'medium')})"
        )

        # 施加空间+特征多样性约束
        # Sentinel 默认使用更激进的多样性阈值
        selected = select_diverse_topk(
            candidates=sorted_candidates,
            topk=num_patches,
            patch_size=self.config.patch_size,
            min_center_distance_ratio=self.config.min_center_distance_ratio,
            min_feature_distance=self.config.min_feature_distance,
        )

        # 记录选择统计信息
        if selected:
            avg_final = sum(c.scores.final_score for c in selected) / len(selected)
            avg_tumor_bias = sum(c.metrics.tumor_bias for c in selected) / len(selected)
            logger.info(
                f"[{candidate_pool.slide_base}] Sentinel: "
                f"selected {len(selected)}/{num_patches} patches "
                f"(avg_final_score={avg_final:.3f}, "
                f"avg_tumor_bias={avg_tumor_bias:.3f})"
            )
        else:
            logger.warning(
                f"[{candidate_pool.slide_base}] Sentinel: "
                f"no patches selected"
            )

        return selected
