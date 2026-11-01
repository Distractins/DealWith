# -*- coding: utf-8 -*-
"""WSI 补丁选择基准测试的工具模块。"""

from utils.logger import setup_logging, get_logger
from utils.seed import set_global_seed, stable_int_hash, deterministic_rng_for_slide
from utils.file_io import (
    ensure_dir,
    append_stats_row,
    save_json,
    parse_existing_patch_indices,
    get_existing_patch_paths_for_slide,
    remove_existing_partial_case_patches,
    save_progress_json,
    load_progress_json,
)
from utils.slide_reader import (
    tcga_case_id_from_string,
    slide_base_from_path,
    build_case_to_slides,
    count_existing_patches_for_case,
)
from utils.math_utils import clip01, minmax_scale, cosine_similarity, pairwise_l2_dist
from utils.timer import Timer, timed

__all__ = [
    "setup_logging",
    "get_logger",
    "set_global_seed",
    "stable_int_hash",
    "deterministic_rng_for_slide",
    "ensure_dir",
    "append_stats_row",
    "save_json",
    "parse_existing_patch_indices",
    "get_existing_patch_paths_for_slide",
    "build_case_to_slides",
    "count_existing_patches_for_case",
    "remove_existing_partial_case_patches",
    "save_progress_json",
    "load_progress_json",
    "tcga_case_id_from_string",
    "slide_base_from_path",
    "clip01",
    "minmax_scale",
    "cosine_similarity",
    "pairwise_l2_dist",
    "Timer",
    "timed",
]
