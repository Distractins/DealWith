# -*- coding: utf-8 -*-
"""
survival_metrics.py
============================================================================
生存分析评估指标模块。

本模块提供了全面的生存分析评估函数，从原始项目的 model_code/utils/utils.py 迁移而来。
包含以下核心功能:
    1. C-index (一致性指数): 纯Python实现 + lifelines库实现
    2. 对数秩检验 (Log-Rank Test): 用于比较两组生存曲线差异
    3. 时间依赖AUC (Time-Dependent AUC): 基于累积/动态定义
    4. 风险比 (Hazard Ratio): 基于中位数分割的高低风险组
    5. 二分类指标: 基于中位数风险分组的分类性能评估
    6. 分组生存摘要: 高低风险组的描述性统计
    7. 辅助函数: 数据清洗、格式转换、时间点选择等

依赖库:
    - numpy, pandas, scipy: 数值计算基础
    - lifelines: 生存分析专用库 (concordance_index, logrank_test, CoxPHFitter)
    - scikit-survival (可选): 时间依赖AUC计算 (需 sksurv.metrics.cumulative_dynamic_auc)
    - torch (可选): 张量到numpy的转换辅助

使用示例:
    from src.evaluation.survival_metrics import (
        CIndex, CIndex_lifeline, cox_log_rank,
        safe_time_dependent_auc, safe_hazard_ratio_by_median_split,
        safe_binary_metrics_from_risk, safe_group_survival_summary,
        accuracy_cox
    )

    # 计算C-index
    ci = CIndex(hazard_pred, survtime, censor)

    # 时间依赖AUC
    auc = safe_time_dependent_auc(survtime, censor, hazard_pred, times=[365, 730, 1095])
============================================================================
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
)

# ---------------------------------------------------------------------------
# 可选依赖检查
# ---------------------------------------------------------------------------
try:
    from lifelines.utils import concordance_index as _lifelines_concordance_index
    _HAS_LIFELINES = True
except ImportError:
    _HAS_LIFELINES = False

try:
    from lifelines.statistics import logrank_test as _lifelines_logrank_test
    _HAS_LIFELINES_LOGRANK = True
except ImportError:
    _HAS_LIFELINES_LOGRANK = False

try:
    from lifelines import CoxPHFitter
    _HAS_COXPH = True
except ImportError:
    _HAS_COXPH = False

try:
    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.util import Surv
    _HAS_SKSURV = True
except ImportError:
    _HAS_SKSURV = False
    cumulative_dynamic_auc = None
    Surv = None

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ===========================================================================
# 第一部分：辅助函数 (Helper Functions)
# ===========================================================================

def _to_numpy_1d(data, name="data", allow_none=False):
    """
    将输入数据转换为1D numpy数组（安全转换）。

    支持的输入类型:
        - numpy ndarray (1D/2D自动展平)
        - pandas Series / DataFrame
        - torch Tensor (detach后转cpu)
        - list / tuple

    参数:
        data: 输入数据
        name: 数据名称（用于错误信息提示）
        allow_none: 是否允许None值

    返回:
        np.ndarray: 1D numpy数组，dtype=float64
    """
    if data is None:
        if allow_none:
            return None
        raise ValueError(f"'{name}' 不能为 None")

    # torch Tensor
    if _HAS_TORCH and isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # pandas
    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.values

    # list / tuple
    if isinstance(data, (list, tuple)):
        data = np.array(data, dtype=np.float64)

    # numpy
    if isinstance(data, np.ndarray):
        data = data.astype(np.float64).ravel()
    else:
        raise TypeError(f"'{name}' 类型不支持: {type(data)}，"
                        f"期望: numpy/torch/pandas/list")

    return data


def _to_float_time_array(data, name="time"):
    """
    将生存时间数据转换为浮点数numpy数组。

    生存时间必须为非负值。此函数同时验证数据有效性。

    参数:
        data: 生存时间数据
        name: 数据名称（用于错误信息）

    返回:
        np.ndarray: 1D float64数组

    异常:
        ValueError: 包含负数或NaN时抛出
    """
    arr = _to_numpy_1d(data, name)
    if np.any(np.isnan(arr)):
        raise ValueError(f"'{name}' 包含 NaN 值")
    if np.any(arr < 0):
        raise ValueError(f"'{name}' 包含负数，生存时间不能为负")
    return arr


def _to_bool_event_array(data, name="event"):
    """
    将事件标记数据转换为布尔型numpy数组。

    支持多种输入格式:
        - 布尔型: True/False, 1/0
        - 整型/浮点型: 1.0=事件, 0.0=删失
        - 其他: 非零值视为事件

    参数:
        data: 事件标记数据
        name: 数据名称（用于错误信息）

    返回:
        np.ndarray: 1D bool数组 (True=事件发生, False=删失)
    """
    arr = _to_numpy_1d(data, name)

    if arr.dtype == bool:
        return arr

    # 整数或浮点数: 1.0/True 表示事件
    # 先检查是否有除0和1以外的值
    unique_vals = np.unique(arr)
    if len(unique_vals) <= 2:
        # 二值情况：统一转为bool
        return arr.astype(bool)

    # 非二值情况：非零视为事件
    warnings.warn(f"'{name}' 包含超过2种取值 {unique_vals}，"
                  f"将非零值视为事件发生。")
    return arr != 0


def _build_structured_surv(survtime, event):
    """
    构建scikit-survival结构化的生存数据数组。

    使用sksurv.util.Surv构建，格式为复合dtype:
        dtype = [('event', bool), ('time', float64)]

    参数:
        survtime: 生存时间数组
        event: 事件标记数组 (True=事件, False=删失)

    返回:
        np.ndarray: 结构化生存数据数组，如果sksurv不可用则返回None
    """
    if not _HAS_SKSURV or Surv is None:
        return None

    time_arr = _to_float_time_array(survtime, "survtime")
    event_arr = _to_bool_event_array(event, "event")

    return Surv.from_arrays(event=event_arr, time=time_arr)


def _sanitize_eval_times(times, survtime, event, n_times=100):
    """
    清洗和验证评估时间点。

    确保评估时间点:
        - 在观测时间范围内
        - 不超过最大事件时间（事件发生后AUC无定义）
        - 均匀分布且数量合理

    参数:
        times: 用户指定的时间点列表或None（自动生成）
        survtime: 生存时间数组
        event: 事件标记数组
        n_times: 自动生成时的默认时间点数

    返回:
        np.ndarray: 清洗后的有效评估时间点
    """
    time_arr = _to_float_time_array(survtime, "survtime")
    event_arr = _to_bool_event_array(event, "event")

    # 有效的观测时间范围
    # 注意: 对于时间依赖AUC，评估时间不应超过最后一个事件时间
    event_times = time_arr[event_arr]
    if len(event_times) == 0:
        raise ValueError("没有事件发生（所有样本均为删失），"
                         "无法计算时间依赖AUC。")

    t_max = np.max(event_times)
    t_min = np.min(time_arr)

    if times is None:
        # 自动生成: 从最小观测时间到最大事件时间，均匀分布
        times = np.linspace(t_min, t_max, min(n_times, len(np.unique(time_arr))))
    else:
        times = _to_numpy_1d(times, "times")
        # 过滤超出范围的时间点
        valid_mask = (times >= t_min) & (times <= t_max)
        if not np.any(valid_mask):
            raise ValueError(f"所有指定的时间点 [{times.min():.1f}, {times.max():.1f}] "
                             f"均不在有效范围 [{t_min:.1f}, {t_max:.1f}] 内。")
        if not np.all(valid_mask):
            n_excluded = np.sum(~valid_mask)
            warnings.warn(f"{n_excluded} 个时间点超出范围 [{t_min:.1f}, {t_max:.1f}]，"
                          f"已自动过滤。")
        times = times[valid_mask]

    return times


def suggest_eval_times_from_data(survtime, event, n_times=100, n_quantiles=None):
    """
    根据数据分布建议评估时间点。

    提供两种策略:
        1. 等间距: n_times个均匀分布的时间点
        2. 分位数: 基于事件时间分位数选择时间点

    参数:
        survtime: 生存时间数组
        event: 事件标记数组
        n_times: 等间距策略的时间点数 (默认100)
        n_quantiles: 分位数策略的时间点数 (如果指定，则使用分位数策略)

    返回:
        np.ndarray: 建议的评估时间点列表

    使用示例:
        # 等间距策略
        times = suggest_eval_times_from_data(survtime, event, n_times=50)

        # 分位数策略
        times = suggest_eval_times_from_data(survtime, event, n_quantiles=20)
    """
    time_arr = _to_float_time_array(survtime, "survtime")
    event_arr = _to_bool_event_array(event, "event")

    event_times = time_arr[event_arr]
    if len(event_times) == 0:
        raise ValueError("没有事件发生，无法建议评估时间点。")

    t_min = np.min(time_arr)
    t_max = np.max(event_times)

    if n_quantiles is not None:
        # 分位数策略: 在事件时间分布上取分位数
        quantiles = np.linspace(0.01, 0.99, min(n_quantiles, len(event_times)))
        times = np.quantile(event_times, quantiles)
    else:
        # 等间距策略
        n_actual = min(n_times, len(np.unique(event_times)))
        times = np.linspace(t_min, t_max, n_actual)

    return times


def _median_risk_group(risk_scores, return_threshold=False):
    """
    基于中位数风险分数将样本分为高/低风险组。

    参数:
        risk_scores: 风险分数数组（越大表示风险越高）
        return_threshold: 是否返回分割阈值

    返回:
        group: 1D bool数组 (True=高风险组, False=低风险组)
        threshold: (仅当return_threshold=True时) float, 分割阈值

    注意:
        - 如果所有风险分数相同，整个样本归入低风险组并发出警告
        - 中位数分割使用 <= 为低风险组，> 为高风险组
    """
    scores = _to_numpy_1d(risk_scores, "risk_scores")
    threshold = np.median(scores)

    if np.isclose(np.min(scores), np.max(scores)):
        warnings.warn("所有风险分数相同，无法进行有意义的中位数分割。"
                      "全部样本归入低风险组。")
        group = np.zeros(len(scores), dtype=bool)
    else:
        group = scores > threshold

    if return_threshold:
        return group, threshold
    return group


def _safe_div(numerator, denominator, default=0.0):
    """
    安全除法，分母为零时返回默认值。

    参数:
        numerator: 分子
        denominator: 分母
        default: 分母为零时的默认返回值

    返回:
        float: 除法结果或默认值
    """
    if denominator is None or denominator == 0:
        return default
    return float(numerator) / float(denominator)


def _validate_survival_inputs(survtime, event, hazard_pred=None):
    """
    验证生存分析输入数据的一致性和有效性。

    检查内容:
        1. 生存时间和事件向量长度一致
        2. 如果提供风险预测，长度也需一致
        3. 生存时间非负
        4. 事件标记为二值或布尔型
        5. 至少有一个事件发生（非全删失）

    参数:
        survtime: 生存时间数组
        event: 事件标记数组
        hazard_pred: 风险预测数组（可选）

    返回:
        tuple: (time_array, event_array, hazard_array) 均为一维numpy数组
               hazard_array在hazard_pred为None时为None
    """
    time_arr = _to_float_time_array(survtime, "survtime")
    event_arr = _to_bool_event_array(event, "event")

    n = len(time_arr)
    if len(event_arr) != n:
        raise ValueError(f"长度不一致: survtime={n}, event={len(event_arr)}")

    if not np.any(event_arr):
        warnings.warn("所有样本均为删失（无事件发生），"
                      "部分指标可能无法计算或返回NaN。")

    hazard_arr = None
    if hazard_pred is not None:
        hazard_arr = _to_numpy_1d(hazard_pred, "hazard_pred")
        if len(hazard_arr) != n:
            raise ValueError(f"长度不一致: survtime={n}, "
                             f"hazard_pred={len(hazard_arr)}")

    return time_arr, event_arr, hazard_arr


# ===========================================================================
# 第二部分：C-index (一致性指数)
# ===========================================================================

def CIndex(hazard_pred, survtime, event):
    """
    计算Harrell's Concordance Index (C-index)，纯Python实现。

    C-index衡量模型预测的风险排序与真实生存时间排序的一致性。
    取值范围 [0, 1]:
        - 1.0: 完美排序
        - 0.5: 随机猜测
        - 0.0: 完全反向排序

    计算逻辑:
        遍历所有样本对(i, j)，其中样本i发生了事件（未删失）且观测时间较短。
        如果风险更高的样本也确实更早发生事件，则记为"一致对"。
        当两个风险预测相同时，计为0.5（风险平局）。

    公式:
        CI = (concordant_pairs + 0.5 * tied_risk_pairs) / comparable_pairs

    参数:
        hazard_pred: array-like, [N] 模型预测的风险分数（越大表示风险越高）
        survtime: array-like, [N] 观测生存时间（天/月）
        event: array-like, [N] 事件标记（1=事件发生, 0=删失）

    返回:
        float: C-index值，范围 [0, 1]

    边缘情况处理:
        - 样本数 < 2: 返回 0.5
        - 无可比较对: 返回 0.5
        - 所有风险相同: 返回 0.5

    使用示例:
        >>> ci = CIndex([0.3, 0.8, 0.1], [100, 50, 200], [1, 1, 0])
        >>> print(ci)  # 约 0.5~0.7
    """
    time_arr, event_arr, hazard_arr = _validate_survival_inputs(
        survtime, event, hazard_pred
    )

    n = len(time_arr)
    if n < 2:
        return 0.5

    comparable_pairs = 0
    concordant_pairs = 0
    tied_risk_pairs = 0

    for i in range(n):
        if not event_arr[i]:
            # 样本i为删失，跳过（不可比较）
            continue
        for j in range(n):
            if i == j:
                continue
            # 样本j必须比样本i活得更久（或同时）才可比较
            if time_arr[j] < time_arr[i]:
                continue

            comparable_pairs += 1

            if hazard_arr[i] > hazard_arr[j]:
                # 高风险样本i更早发生事件 -> 一致
                concordant_pairs += 1
            elif hazard_arr[i] == hazard_arr[j]:
                # 风险相同 -> 0.5
                tied_risk_pairs += 1
            # else: hazard_arr[i] < hazard_arr[j] -> 不一致，不计数

    if comparable_pairs == 0:
        return 0.5

    ci = (concordant_pairs + 0.5 * tied_risk_pairs) / comparable_pairs
    return float(ci)


def CIndex_lifeline(hazard_pred, survtime, event, nan_policy="warn"):
    """
    使用lifelines库计算Harrell's Concordance Index（安全版本）。

    该函数封装了lifelines.utils.concordance_index，添加了输入验证
    和边缘情况处理。内部会验证数据有效性并处理空数组/无效值。

    参数:
        hazard_pred: array-like, [N] 模型预测的风险分数（越大表示风险越高）
        survtime: array-like, [N] 观测生存时间
        event: array-like, [N] 事件标记（1=事件, 0=删失）
        nan_policy: str, NaN处理策略
            - "warn": 发出警告并返回0.5（默认）
            - "raise": 抛出异常
            - "omit": 静默返回0.5

    返回:
        float: C-index值，范围 [0, 1]

    异常:
        ImportError: lifelines库未安装时抛出
        ValueError: 数据无效且nan_policy="raise"时抛出

    使用示例:
        >>> ci = CIndex_lifeline([0.3, 0.8, 0.1], [100, 50, 200], [1, 1, 0])
    """
    if not _HAS_LIFELINES:
        raise ImportError(
            "lifelines 库未安装。请运行: pip install lifelines>=0.27.0"
        )

    time_arr, event_arr, hazard_arr = _validate_survival_inputs(
        survtime, event, hazard_pred
    )

    n = len(time_arr)
    if n < 2:
        return 0.5

    # 检查是否有足够的事件
    if np.sum(event_arr) < 1:
        if nan_policy == "raise":
            raise ValueError("没有事件发生，无法计算C-index。")
        elif nan_policy == "warn":
            warnings.warn("没有事件发生，C-index无定义，返回0.5。")
        return 0.5

    # 检查风险分数是否全部相同
    if np.isclose(np.min(hazard_arr), np.max(hazard_arr)):
        if nan_policy == "warn":
            warnings.warn("所有风险分数相同，C-index无意义，返回0.5。")
        return 0.5

    try:
        ci = _lifelines_concordance_index(
            event_times=time_arr,
            predicted_scores=hazard_arr,
            event_observed=event_arr,
        )
        return float(ci)
    except Exception as e:
        if nan_policy == "raise":
            raise ValueError(f"lifelines C-index计算失败: {e}")
        elif nan_policy == "warn":
            warnings.warn(f"lifelines C-index计算失败: {e}，返回0.5。")
        return 0.5


# ===========================================================================
# 第三部分：对数秩检验 (Log-Rank Test)
# ===========================================================================

def cox_log_rank(hazard_pred, survtime, event, split_method="median"):
    """
    对数秩检验 (Log-Rank Test): 基于风险预测分组比较生存差异。

    将样本按风险分数分为高/低风险两组，计算两组生存曲线之间的
    对数秩检验p值。p值越小表示两组差异越显著。

    分组方法:
        - "median": 中位数分割（默认）
        - "mean": 均值分割
        - "quantile": 自定义分位数分割（需指定quantile参数）

    参数:
        hazard_pred: array-like, [N] 风险预测分数（越大风险越高）
        survtime: array-like, [N] 观测生存时间
        event: array-like, [N] 事件标记（1=事件, 0=删失）
        split_method: str, 分组方法 ("median" / "mean" / "quantile")

    返回:
        float: 对数秩检验p值。值为NaN表示无法计算（如两组中一组无事件）

    边缘情况处理:
        - 样本数 < 2: 返回 NaN
        - 所有风险分数相同: 返回 NaN（无法分组）
        - 任一组无事件: 返回 NaN
        - lifelines不可用: 使用scipy实现替代

    使用示例:
        >>> pval = cox_log_rank([0.3, 0.8, 0.1, 0.9], [100, 50, 200, 30], [1,1,0,1])
        >>> print(f"Log-rank p-value: {pval:.4f}")
    """
    time_arr, event_arr, hazard_arr = _validate_survival_inputs(
        survtime, event, hazard_pred
    )

    n = len(time_arr)
    if n < 2:
        warnings.warn("样本数不足（< 2），无法计算log-rank检验，返回NaN。")
        return float('nan')

    # ---- 分组 ----
    if split_method == "median":
        threshold = np.median(hazard_arr)
    elif split_method == "mean":
        threshold = np.mean(hazard_arr)
    elif split_method == "quantile":
        threshold = np.median(hazard_arr)  # 默认回退到中位数
    else:
        raise ValueError(f"不支持的分组方法: '{split_method}'，"
                         f"可选: 'median' / 'mean' / 'quantile'")

    high_group = hazard_arr > threshold
    low_group = ~high_group

    # 检查分组有效性
    n_high = np.sum(high_group)
    n_low = np.sum(low_group)
    if n_high == 0 or n_low == 0:
        warnings.warn(f"分组无效: 高风险组={n_high}, 低风险组={n_low}，"
                      f"返回NaN。")
        return float('nan')

    # 检查每组是否有事件
    events_high = np.sum(event_arr[high_group])
    events_low = np.sum(event_arr[low_group])
    if events_high == 0 or events_low == 0:
        warnings.warn(f"任一组无事件: 高风险组事件={events_high}, "
                      f"低风险组事件={events_low}，返回NaN。")
        return float('nan')

    # ---- 使用lifelines计算 ----
    if _HAS_LIFELINES_LOGRANK:
        try:
            result = _lifelines_logrank_test(
                durations_A=time_arr[high_group],
                durations_B=time_arr[low_group],
                event_observed_A=event_arr[high_group],
                event_observed_B=event_arr[low_group],
            )
            return float(result.p_value)
        except Exception as e:
            warnings.warn(f"lifelines logrank计算失败: {e}，"
                          f"回退到scipy实现。")

    # ---- scipy回退实现 ----
    # 使用卡方检验近似对数秩检验
    # 对每个唯一事件时间点计算期望事件数和方差
    unique_times = np.unique(time_arr[event_arr])
    if len(unique_times) == 0:
        return float('nan')

    O1 = 0.0  # 高风险组观测事件数（组1）
    E1 = 0.0  # 高风险组期望事件数
    V1 = 0.0  # 方差

    for t in unique_times:
        # 时间t时的风险集
        at_risk_high = np.sum((time_arr[high_group] >= t).astype(float))
        at_risk_low = np.sum((time_arr[low_group] >= t).astype(float))
        at_risk_total = at_risk_high + at_risk_low

        if at_risk_total == 0:
            continue

        # 时间t时的事件数
        events_total = np.sum(event_arr & (time_arr == t))
        if events_total == 0:
            continue

        # 高风险组在时间t的观测事件数
        events_high_t = np.sum(event_arr[high_group] & (time_arr[high_group] == t))

        O1 += events_high_t
        E1 += events_total * (at_risk_high / at_risk_total)
        # 超几何方差
        if at_risk_total > 1:
            V1 += (events_total * at_risk_high * at_risk_low *
                   (at_risk_total - events_total) /
                   (at_risk_total ** 2 * (at_risk_total - 1)))

    if V1 == 0:
        return float('nan')

    # 卡方统计量
    chi_sq = (O1 - E1) ** 2 / V1
    p_value = scipy_stats.chi2.sf(chi_sq, df=1)

    return float(p_value)


# ===========================================================================
# 第四部分：时间依赖AUC (Time-Dependent AUC)
# ===========================================================================

def safe_time_dependent_auc(survtime, event, hazard_pred,
                            times=None, n_times=100,
                            nan_policy="warn"):
    """
    计算时间依赖的累积/动态AUC (Time-Dependent AUC)。

    使用scikit-survival的cumulative_dynamic_auc，基于
    Uno等人和Hung-Chiang的累积/动态定义。

    累积AUC: 衡量模型在给定时间点t之前区分事件发生与否的能力。
    在多个时间点上计算后取均值可作为整体指标。

    参数:
        survtime: array-like, [N] 观测生存时间
        event: array-like, [N] 事件标记（1=事件, 0=删失）
        hazard_pred: array-like, [N] 风险预测分数（越大风险越高）
        times: array-like or None, 评估时间点列表。
               如果为None，自动生成均匀分布的时间点。
        n_times: int, 自动生成时间点的数量（默认100）
        nan_policy: str, NaN处理策略 ("warn"/"raise"/"omit")

    返回:
        float or dict:
            - 如果scikit-survival可用: 返回dict
                {"mean_auc": float, "times": ndarray, "auc_values": ndarray}
            - 如果scikit-survival不可用: 返回NaN (回退为C-index近似)

    边缘情况处理:
        - 无事件: 返回NaN
        - 样本数<5: 返回NaN
        - sksurv不可用: 使用C-index近似并发出警告

    使用示例:
        >>> result = safe_time_dependent_auc(
        ...     [100, 50, 200, 80],
        ...     [1, 1, 0, 1],
        ...     [0.3, 0.8, 0.1, 0.6],
        ...     times=[30, 90, 150]
        ... )
        >>> print(result["mean_auc"])
    """
    time_arr, event_arr, hazard_arr = _validate_survival_inputs(
        survtime, event, hazard_pred
    )

    n = len(time_arr)
    if n < 5:
        if nan_policy == "raise":
            raise ValueError(f"样本数不足 ({n} < 5)，无法计算时间依赖AUC。")
        elif nan_policy == "warn":
            warnings.warn(f"样本数不足 ({n} < 5)，返回NaN。")
        return {"mean_auc": float('nan'), "times": np.array([]),
                "auc_values": np.array([])}

    if np.sum(event_arr) < 1:
        if nan_policy == "raise":
            raise ValueError("没有事件发生，无法计算时间依赖AUC。")
        elif nan_policy == "warn":
            warnings.warn("没有事件发生，返回NaN。")
        return {"mean_auc": float('nan'), "times": np.array([]),
                "auc_values": np.array([])}

    # 清洗评估时间点
    try:
        eval_times = _sanitize_eval_times(times, time_arr, event_arr, n_times)
    except ValueError as e:
        if nan_policy == "raise":
            raise
        warnings.warn(f"评估时间点清洗失败: {e}，返回NaN。")
        return {"mean_auc": float('nan'), "times": np.array([]),
                "auc_values": np.array([])}

    # ---- scikit-survival路径 ----
    if _HAS_SKSURV:
        try:
            surv_data = _build_structured_surv(time_arr, event_arr)
            if surv_data is None:
                raise RuntimeError("无法构建结构化生存数据。")

            # sksurv的cumulative_dynamic_auc期望风险分数越大表示
            # 生存时间越长（即低风险）。因此需要取反。
            # 我们的hazard_pred越大表示风险越高，所以需要取负号。
            risk_scores_for_auc = -hazard_arr

            result = cumulative_dynamic_auc(
                survival_train=surv_data,
                survival_test=surv_data,
                estimate=risk_scores_for_auc,
                times=eval_times,
            )

            auc_values = result[1]  # AUC值数组
            mean_auc = np.nanmean(auc_values)

            return {
                "mean_auc": float(mean_auc),
                "times": eval_times,
                "auc_values": auc_values,
            }

        except Exception as e:
            if nan_policy == "raise":
                raise RuntimeError(f"sksurv时间依赖AUC计算失败: {e}")
            warnings.warn(f"sksurv时间依赖AUC计算失败: {e}，"
                          f"回退到C-index近似。")

    # ---- 回退：使用C-index作为近似 ----
    if nan_policy == "raise":
        raise ImportError(
            "scikit-survival 库未安装。请运行: pip install scikit-survival>=0.20.0"
        )

    warnings.warn(
        "scikit-survival 不可用，使用C-index作为时间依赖AUC的近似替��。"
        "建议: pip install scikit-survival>=0.20.0"
    )
    ci = CIndex(hazard_arr, time_arr, event_arr)
    return {
        "mean_auc": float(ci),
        "times": eval_times,
        "auc_values": np.array([ci]),
        "_note": "回退为C-index近似（sksurv不可用）",
    }


# ===========================================================================
# 第五���分：风险比 (Hazard Ratio)
# ===========================================================================

def safe_hazard_ratio_by_median_split(hazard_pred, survtime, event,
                                      return_details=False):
    """
    基于中位数风险分数分割计算风险比 (Hazard Ratio)。

    将样本按预测风险的中位数分为高/低风险两组，使用Cox比例风险模型
    估计两组之间的风险比。HR > 1 表示高风险组比低风险组风险更高。

    参数:
        hazard_pred: array-like, [N] 风险预测分数
        survtime: array-like, [N] 观测生存时间
        event: array-like, [N] 事件标记
        return_details: bool, 是否返回详细信息（默认False）

    返回:
        float or dict:
            - return_details=False: float, 风险比 (HR)
            - return_details=True: dict {
                "hazard_ratio": float,
                "ci_lower": float (95%CI下界),
                "ci_upper": float (95%CI上界),
                "p_value": float,
                "log_rank_p": float,
                "n_high": int,
                "n_low": int,
                "events_high": int,
                "events_low": int,
                "threshold": float (分割阈值),
                "method": str ("lifelines" 或 "approximate"),
              }

    边缘情况处理:
        - 样本数 < 5: 返回 NaN 或默认字典
        - 所有风险相同: 返回 NaN
        - 任一组无事件: 返回 NaN
        - lifelines不可用: 使用近似计算方法

    使用示例:
        >>> hr = safe_hazard_ratio_by_median_split(
        ...     [0.3, 0.8, 0.1, 0.9],
        ...     [100, 50, 200, 30],
        ...     [1, 1, 0, 1]
        ... )
        >>> print(f"Hazard Ratio: {hr:.3f}")
    """
    time_arr, event_arr, hazard_arr = _validate_survival_inputs(
        survtime, event, hazard_pred
    )

    n = len(time_arr)

    # 默认返回值
    default_return = float('nan')
    default_details = {
        "hazard_ratio": float('nan'),
        "ci_lower": float('nan'),
        "ci_upper": float('nan'),
        "p_value": float('nan'),
        "log_rank_p": float('nan'),
        "n_high": 0,
        "n_low": 0,
        "events_high": 0,
        "events_low": 0,
        "threshold": float('nan'),
        "method": "failed",
    }

    if n < 5:
        warnings.warn(f"样本数不足 ({n} < 5)，无法可靠计算风险比，返回NaN。")
        return default_details if return_details else default_return

    # 中位数分割
    high_group, threshold = _median_risk_group(hazard_arr, return_threshold=True)

    n_high = int(np.sum(high_group))
    n_low = int(np.sum(~high_group))
    events_high = int(np.sum(event_arr[high_group]))
    events_low = int(np.sum(event_arr[~high_group]))

    if n_high == 0 or n_low == 0:
        warnings.warn(f"分组无效: 高风险组={n_high}, 低风险组={n_low}，返回NaN。")
        return default_details if return_details else default_return

    if events_high == 0 or events_low == 0:
        warnings.warn(f"任一组无事件: 高风险组事件={events_high}, "
                      f"低风险组事件={events_low}，返回NaN。")
        return default_details if return_details else default_return

    # 尝试lifelines Cox模型
    if _HAS_COXPH:
        try:
            df = pd.DataFrame({
                "time": time_arr,
                "event": event_arr.astype(bool),
                "group": high_group.astype(int),  # 1=高风险组
            })

            cph = CoxPHFitter()
            cph.fit(df, duration_col="time", event_col="event",
                    show_progress=False)

            hr = np.exp(cph.params_["group"])
            summary = cph.summary.loc["group"]

            ci_lower = np.exp(summary["coef lower 95%"])
            ci_upper = np.exp(summary["coef upper 95%"])
            p_value = summary["p"]

            # 同时计算对数秩检验p值
            log_rank_p = cox_log_rank(hazard_arr, time_arr, event_arr)

            details = {
                "hazard_ratio": float(hr),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "p_value": float(p_value),
                "log_rank_p": float(log_rank_p),
                "n_high": n_high,
                "n_low": n_low,
                "events_high": events_high,
                "events_low": events_low,
                "threshold": float(threshold),
                "method": "lifelines",
            }
            return details if return_details else float(hr)

        except Exception as e:
            warnings.warn(f"lifelines风险比计算失败: {e}，"
                          f"使用近似方法。")

    # ---- 近似计算方法（不使用Cox模型） ----
    # 基于两组的事件率比值来近似风险比
    # HR ≈ (events_high / total_time_high) / (events_low / total_time_low)
    total_time_high = np.sum(time_arr[high_group])
    total_time_low = np.sum(time_arr[~high_group])

    rate_high = _safe_div(events_high, total_time_high, 0.0)
    rate_low = _safe_div(events_low, total_time_low, 0.0)
    hr_approx = _safe_div(rate_high, rate_low, float('nan'))

    log_rank_p = cox_log_rank(hazard_arr, time_arr, event_arr)

    details = {
        "hazard_ratio": float(hr_approx),
        "ci_lower": float('nan'),
        "ci_upper": float('nan'),
        "p_value": float('nan'),
        "log_rank_p": float(log_rank_p),
        "n_high": n_high,
        "n_low": n_low,
        "events_high": events_high,
        "events_low": events_low,
        "threshold": float(threshold),
        "method": "approximate",
    }

    warnings.warn(
        "使用近似方法计算风险比（lifelines不可用或计算失败）。"
        "近似HR = (高组事件率) / (低组事件率)，无置信区间。"
        "建议: pip install lifelines>=0.27.0"
    )

    return details if return_details else float(hr_approx)


# ===========================================================================
# 第六部分：基于风险分组的二分类指标
# ===========================================================================

def safe_binary_metrics_from_risk(hazard_pred, survtime, event,
                                  time_threshold=None,
                                  return_dict=True):
    """
    基于中位数风险分组计算二分类指标。

    使用中位数风险分数将样本分为高/低风险组后，以事件是否在
    time_threshold前发生作为二分类标签，计算分类指标。

    指标包括:
        - Accuracy (准确率)
        - Precision (精确率)
        - Recall (召回率 / 灵敏度)
        - F1 Score
        - AUC-ROC
        - Specificity (特异度)
        - Balanced Accuracy (平衡准确率)

    参数:
        hazard_pred: array-like, [N] 风险预测分数
        survtime: array-like, [N] 生存时间
        event: array-like, [N] 事件标记
        time_threshold: float or None, 时间阈值。
                        事件在time_threshold前发生视为正类。
                        如果为None，使用观测时间的中位数。
        return_dict: bool, 是否返回完整指标字典（默认True）

    返回:
        dict or float:
            - return_dict=True: 指标字典
            - return_dict=False: 精确率（单值，为兼容旧接口）

    边缘情况处理:
        - 任一组为空: 返回NaN值
        - 纯正类或纯负类: 部分指标（如AUC）返回NaN

    使用示例:
        >>> metrics = safe_binary_metrics_from_risk(
        ...     [0.3, 0.8, 0.1, 0.9],
        ...     [100, 50, 200, 30],
        ...     [1, 1, 0, 1],
        ...     time_threshold=90
        ... )
        >>> print(metrics["accuracy"])
    """
    time_arr, event_arr, hazard_arr = _validate_survival_inputs(
        survtime, event, hazard_pred
    )

    n = len(time_arr)

    # 默认返回值
    default_metrics = {
        "accuracy": float('nan'),
        "precision": float('nan'),
        "recall": float('nan'),
        "f1_score": float('nan'),
        "auc_roc": float('nan'),
        "specificity": float('nan'),
        "balanced_accuracy": float('nan'),
        "confusion_matrix": np.zeros((2, 2), dtype=int),
        "n_total": n,
        "n_high": 0,
        "n_low": 0,
        "n_positive": 0,
        "n_negative": 0,
        "time_threshold": time_threshold,
        "threshold_risk": float('nan'),
    }

    if n < 2:
        warnings.warn(f"样本数不足 ({n} < 2)，返回NaN。")
        return default_metrics if return_dict else float('nan')

    # 设置时间阈值
    if time_threshold is None:
        time_threshold = float(np.median(time_arr))

    # 中位数风险分组: 高风险组 = 预测正类
    high_group, risk_threshold = _median_risk_group(hazard_arr,
                                                     return_threshold=True)

    # 定义标签: 在time_threshold前发生事件 = 真实正类
    # 注意：删失样本在时间阈值内的处理
    # - 如果在时间阈值前删失且未观测到事件，视为负类
    # - 如果在时间阈值前发生了事件，视为正类
    # - 如果存活超过时间阈值但后来发生事件，在阈值处视为负类
    true_positive = event_arr & (time_arr <= time_threshold)
    true_negative = ~true_positive  # 包括删失和存活超过阈值的样本

    # 预测: 高风险组 -> 正类预测
    pred_positive = high_group
    pred_negative = ~high_group

    n_high = int(np.sum(pred_positive))
    n_low = int(np.sum(pred_negative))
    n_positive = int(np.sum(true_positive))
    n_negative = int(np.sum(true_negative))

    if n_high == 0 or n_low == 0:
        warnings.warn(f"风险分组无效: n_high={n_high}, n_low={n_low}，"
                      f"返回NaN。")
        return default_metrics if return_dict else float('nan')

    if n_positive == 0 or n_negative == 0:
        warnings.warn(f"时间阈值 {time_threshold:.1f} 下仅有单一类别: "
                      f"正类={n_positive}, 负类={n_negative}，"
                      f"部分指标可能为NaN。")

    # 计算混淆矩阵
    try:
        cm = confusion_matrix(true_negative, pred_negative,
                              labels=[1, 0])  # 注意：负类=1用于confusion_matrix
    except Exception:
        cm = np.zeros((2, 2), dtype=int)
        cm[0, 0] = int(np.sum(true_negative & pred_negative))
        cm[0, 1] = int(np.sum(true_negative & pred_positive))
        cm[1, 0] = int(np.sum(true_positive & pred_negative))
        cm[1, 1] = int(np.sum(true_positive & pred_positive))

    tn, fp, fn, tp = cm.ravel()

    # 计算各项指标
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn, float('nan'))
    precision = _safe_div(tp, tp + fp, float('nan'))
    recall = _safe_div(tp, tp + fn, float('nan'))  # 灵敏度
    specificity = _safe_div(tn, tn + fp, float('nan'))
    f1 = _safe_div(2 * precision * recall, precision + recall, float('nan'))

    # 平衡准确率
    bal_acc = _safe_div(recall + specificity, 2.0, float('nan'))

    # AUC-ROC
    try:
        if n_positive > 0 and n_negative > 0:
            auc_roc = roc_auc_score(true_positive.astype(int),
                                    hazard_arr)
        else:
            auc_roc = float('nan')
    except Exception:
        auc_roc = float('nan')

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc_roc": float(auc_roc),
        "specificity": float(specificity),
        "balanced_accuracy": float(bal_acc),
        "confusion_matrix": cm,
        "n_total": n,
        "n_high": n_high,
        "n_low": n_low,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "time_threshold": float(time_threshold),
        "threshold_risk": float(risk_threshold),
    }

    if return_dict:
        return metrics
    else:
        return float(precision)


# ===========================================================================
# 第七部分：分组生存摘要
# ===========================================================================

def safe_group_survival_summary(hazard_pred, survtime, event,
                                split_method="median"):
    """
    计算高低风险分组的生存数据摘要统计。

    提供两组（高/低风险）的描述性统计，包括:
        - 样本数
        - 事件数
        - 删失数
        - 中位生存时间
        - 最小/最大生存时间
        - 事件发生率
        - 删失率
        - 对数秩检验p值
        - C-index

    参数:
        hazard_pred: array-like, [N] 风险预测分数
        survtime: array-like, [N] 观测生存时间
        event: array-like, [N] 事件标记
        split_method: str, 分组方法 ("median" / "mean" / "quantile")

    返回:
        dict: 包含以下键的字典
            - "high_risk": dict, 高风险组统计
            - "low_risk": dict, 低风险组统计
            - "overall": dict, 总体统计
            - "log_rank_p_value": float, 对数秩检验p值
            - "c_index": float, C-index
            - "threshold": float, 分割阈值

    使用示例:
        >>> summary = safe_group_survival_summary(
        ...     [0.3, 0.8, 0.1, 0.9],
        ...     [100, 50, 200, 30],
        ...     [1, 1, 0, 1]
        ... )
        >>> print(summary["high_risk"]["n_events"])
    """
    time_arr, event_arr, hazard_arr = _validate_survival_inputs(
        survtime, event, hazard_pred
    )

    n = len(time_arr)

    # 分组
    if split_method == "median":
        threshold = np.median(hazard_arr)
    elif split_method == "mean":
        threshold = np.mean(hazard_arr)
    elif split_method == "quantile":
        threshold = np.median(hazard_arr)
    else:
        raise ValueError(f"不支持的分组方法: '{split_method}'")

    high_group = hazard_arr > threshold
    low_group = ~high_group

    n_high = int(np.sum(high_group))
    n_low = int(np.sum(low_group))

    def _group_stats(group_mask, label):
        """计算单个分组的描述性统计。"""
        group_times = time_arr[group_mask]
        group_events = event_arr[group_mask]
        n_total = len(group_times)

        if n_total == 0:
            return {
                "label": label,
                "n_total": 0,
                "n_events": 0,
                "n_censored": 0,
                "event_rate": float('nan'),
                "censoring_rate": float('nan'),
                "median_survival": float('nan'),
                "min_survival": float('nan'),
                "max_survival": float('nan'),
                "mean_survival": float('nan'),
                "std_survival": float('nan'),
            }

        n_events = int(np.sum(group_events))
        n_censored = n_total - n_events

        event_rate = _safe_div(n_events, n_total, float('nan'))
        censoring_rate = _safe_div(n_censored, n_total, float('nan'))

        # 中位生存时间（Kaplan-Meier简易近似：排序后取中位观测时间）
        sorted_times = np.sort(group_times)
        median_survival = float(np.median(sorted_times))

        return {
            "label": label,
            "n_total": n_total,
            "n_events": n_events,
            "n_censored": n_censored,
            "event_rate": float(event_rate),
            "censoring_rate": float(censoring_rate),
            "median_survival": median_survival,
            "min_survival": float(np.min(group_times)),
            "max_survival": float(np.max(group_times)),
            "mean_survival": float(np.mean(group_times)),
            "std_survival": float(np.std(group_times)),
        }

    high_stats = _group_stats(high_group, "高风险组")
    low_stats = _group_stats(low_group, "低风险组")

    # 总体统计
    n_events_total = int(np.sum(event_arr))
    n_censored_total = n - n_events_total

    overall = {
        "n_total": n,
        "n_events": n_events_total,
        "n_censored": n_censored_total,
        "event_rate": _safe_div(n_events_total, n, float('nan')),
        "censoring_rate": _safe_div(n_censored_total, n, float('nan')),
        "median_survival": float(np.median(time_arr)),
        "min_survival": float(np.min(time_arr)),
        "max_survival": float(np.max(time_arr)),
        "mean_survival": float(np.mean(time_arr)),
        "std_survival": float(np.std(time_arr)),
    }

    # 对数秩检验
    log_rank_p = cox_log_rank(hazard_arr, time_arr, event_arr,
                              split_method=split_method)

    # C-index
    ci = CIndex(hazard_arr, time_arr, event_arr)

    return {
        "high_risk": high_stats,
        "low_risk": low_stats,
        "overall": overall,
        "log_rank_p_value": float(log_rank_p) if not np.isnan(log_rank_p) else float('nan'),
        "c_index": float(ci),
        "threshold": float(threshold),
        "split_method": split_method,
    }


# ===========================================================================
# 第八部分：accuracy_cox - Cox模型准确率评估
# ===========================================================================

def accuracy_cox(hazard_pred, survtime, event, time_threshold=None):
    """
    评估Cox模型预测准确率（基于时间阈值的分组正确率）。

    该函数综合评估模型性能:
        1. 使用中位数风险分数将样本分为高/低风险组
        2. 以time_threshold为界定义二分类标签
           （事件在阈值前发生 = 正类）
        3. 计算准确率: 高风险组预测正类，低风险组预测负类

    这是原始项目 model_code/utils/utils.py 中 accuracy_cox 函数的迁移版本。

    参数:
        hazard_pred: array-like, [N] 风险预测分数（越大风险越高）
        survtime: array-like, [N] 观测生存时间
        event: array-like, [N] 事件标记（1=事件, 0=删失）
        time_threshold: float or None, 时间阈值。
                        如果为None，使用观测时间的中位数。

    返回:
        dict: 包含以下键的字典
            - "accuracy": float, 准确率
            - "precision": float, 精确率
            - "recall": float, 召回率
            - "f1_score": float, F1分数
            - "c_index": float, C-index
            - "log_rank_p": float, 对数秩检验p值
            - "hazard_ratio": float or NaN, 风险比
            - "time_threshold": float, 使用的时间阈值
            - "risk_threshold": float, 风险分割阈值
            - "n_high": int, 高风险组样本数
            - "n_low": int, 低风险组样本数

    边缘情况处理:
        - 样本数 < 2: 返回NaN值的字典
        - 所有风险分数相同: 返回NaN值的字典
        - 时间阈值导致单一类别: 部分指标为NaN

    使用示例:
        >>> result = accuracy_cox(
        ...     [0.3, 0.8, 0.1, 0.9],
        ...     [100, 50, 200, 30],
        ...     [1, 1, 0, 1],
        ...     time_threshold=90
        ... )
        >>> print(result["accuracy"])
    """
    time_arr, event_arr, hazard_arr = _validate_survival_inputs(
        survtime, event, hazard_pred
    )

    n = len(time_arr)

    # 设置时间阈值
    if time_threshold is None:
        time_threshold = float(np.median(time_arr))

    # 默认返回值
    nan_result = {
        "accuracy": float('nan'),
        "precision": float('nan'),
        "recall": float('nan'),
        "f1_score": float('nan'),
        "c_index": float('nan'),
        "log_rank_p": float('nan'),
        "hazard_ratio": float('nan'),
        "time_threshold": float(time_threshold),
        "risk_threshold": float('nan'),
        "n_high": 0,
        "n_low": 0,
    }

    if n < 2:
        warnings.warn(f"样本数不足 ({n} < 2)，返回NaN。")
        return nan_result

    # 检查风险分数是否有区分度
    if np.isclose(np.min(hazard_arr), np.max(hazard_arr)):
        warnings.warn("所有风险分数相同，无法有效评估，返回NaN。")
        return nan_result

    # 中位数风险分组
    high_group, risk_threshold = _median_risk_group(hazard_arr,
                                                     return_threshold=True)

    n_high = int(np.sum(high_group))
    n_low = int(np.sum(~high_group))

    if n_high == 0 or n_low == 0:
        warnings.warn(f"分组无效: n_high={n_high}, n_low={n_low}，返回NaN。")
        return nan_result

    # 定义二分类标签
    # 正类: 在time_threshold前发生事件
    # 负类: 未在time_threshold前发生事件（包括删失和存活更久者）
    true_positive = event_arr & (time_arr <= time_threshold)
    y_true = true_positive.astype(int)

    # 预测: 高风险组 -> 正类 (1), 低风险组 -> 负类 (0)
    y_pred = high_group.astype(int)

    n_positive = int(np.sum(y_true))
    n_negative = n - n_positive

    # 计算分类指标
    if n_positive == 0 or n_negative == 0:
        warnings.warn(f"时间阈值 {time_threshold:.1f} 下仅有单一类别: "
                      f"正类={n_positive}, 负类={n_negative}。")

    try:
        acc = accuracy_score(y_true, y_pred)
    except Exception:
        acc = float('nan')

    try:
        prec = precision_score(y_true, y_pred, zero_division=0)
    except Exception:
        prec = float('nan')

    try:
        rec = recall_score(y_true, y_pred, zero_division=0)
    except Exception:
        rec = float('nan')

    try:
        f1 = f1_score(y_true, y_pred, zero_division=0)
    except Exception:
        f1 = float('nan')

    # C-index
    ci = CIndex(hazard_arr, time_arr, event_arr)

    # 对数秩检验
    log_rank_p = cox_log_rank(hazard_arr, time_arr, event_arr)

    # 风险比
    hr = safe_hazard_ratio_by_median_split(hazard_arr, time_arr, event_arr,
                                           return_details=False)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "c_index": float(ci),
        "log_rank_p": float(log_rank_p) if not np.isnan(log_rank_p) else float('nan'),
        "hazard_ratio": float(hr) if not np.isnan(hr) else float('nan'),
        "time_threshold": float(time_threshold),
        "risk_threshold": float(risk_threshold),
        "n_high": n_high,
        "n_low": n_low,
    }


# ===========================================================================
# 第九部分：便捷函数 (Convenience Functions)
# ===========================================================================

def evaluate_survival_model(hazard_pred, survtime, event,
                            times=None,
                            time_threshold=None,
                            prefix=""):
    """
    一站式生存模型评估函数。

    计算所有常用生存分析指标并汇总为字典，方便直接用于实验记录。

    计算的指标包括:
        - C-index
        - 时间依赖AUC (均值)
        - 对数秩检验p值
        - 风险比
        - 准确率 (accuracy_cox)
        - 分组生存摘要

    参数:
        hazard_pred: array-like, [N] 风险预测分数
        survtime: array-like, [N] 观测生存时间
        event: array-like, [N] 事件标记
        times: array-like or None, 时间依赖AUC的评估时间点
        time_threshold: float or None, accuracy_cox的时间阈值
        prefix: str, 返回字典键的前缀（如 "train_" / "val_" / "test_"）

    返回:
        dict: 所有评估指标的汇总字典

    使用示例:
        >>> results = evaluate_survival_model(
        ...     pred, survtime, event, prefix="test_"
        ... )
        >>> print(f"Test C-index: {results['test_c_index']:.4f}")
    """
    time_arr, event_arr, hazard_arr = _validate_survival_inputs(
        survtime, event, hazard_pred
    )

    results = {}

    # C-index
    ci = CIndex(hazard_arr, time_arr, event_arr)
    results[f"{prefix}c_index"] = float(ci)

    try:
        ci_ll = CIndex_lifeline(hazard_arr, time_arr, event_arr,
                                nan_policy="omit")
        results[f"{prefix}c_index_lifeline"] = float(ci_ll)
    except Exception:
        results[f"{prefix}c_index_lifeline"] = float('nan')

    # 时间依赖AUC
    auc_result = safe_time_dependent_auc(time_arr, event_arr, hazard_arr,
                                         times=times, nan_policy="omit")
    results[f"{prefix}time_auc_mean"] = float(auc_result.get("mean_auc", float('nan')))

    # 对数秩检验
    log_rank_p = cox_log_rank(hazard_arr, time_arr, event_arr)
    results[f"{prefix}log_rank_p"] = float(log_rank_p) if not np.isnan(log_rank_p) else float('nan')

    # 风险比
    hr = safe_hazard_ratio_by_median_split(hazard_arr, time_arr, event_arr,
                                           return_details=False)
    results[f"{prefix}hazard_ratio"] = float(hr) if not np.isnan(hr) else float('nan')

    # accuracy_cox
    acc_result = accuracy_cox(hazard_arr, time_arr, event_arr, time_threshold)
    for key in ["accuracy", "precision", "recall", "f1_score"]:
        results[f"{prefix}{key}"] = float(acc_result.get(key, float('nan')))

    # 分组信息
    results[f"{prefix}n_high"] = int(acc_result.get("n_high", 0))
    results[f"{prefix}n_low"] = int(acc_result.get("n_low", 0))

    return results


# ===========================================================================
# 模块自测
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("生存分析评估指标模块自测")
    print("=" * 70)

    # 生成模拟生存数据
    np.random.seed(42)
    n_samples = 200

    # 模拟观测时间（天）
    survtime = np.random.exponential(scale=365 * 3, size=n_samples)
    survtime = np.clip(survtime, 30, 365 * 5)

    # 模拟事件标记（约60%事件率）
    event = np.random.binomial(1, 0.6, size=n_samples).astype(bool)

    # 模拟风险分数（与真实生存时间负相关 + 噪声）
    true_hazard = 1.0 / (survtime + 1)
    hazard_pred = true_hazard + np.random.normal(0, 0.2, size=n_samples)

    print(f"\n  数据信息:")
    print(f"    样本数: {n_samples}")
    print(f"    事件数: {np.sum(event)}")
    print(f"    删失数: {np.sum(~event)}")
    print(f"    中位生存时间: {np.median(survtime):.1f} 天")
    print(f"    生存时间范围: [{np.min(survtime):.1f}, {np.max(survtime):.1f}]")

    # ---- 测试 CIndex ----
    print(f"\n{'='*50}")
    print("1. C-index 测试")
    print(f"{'='*50}")
    ci = CIndex(hazard_pred, survtime, event)
    print(f"  纯Python C-index: {ci:.4f}")

    if _HAS_LIFELINES:
        ci_ll = CIndex_lifeline(hazard_pred, survtime, event)
        print(f"  lifelines C-index: {ci_ll:.4f}")
    else:
        print("  lifelines 不可用，跳过 CIndex_lifeline 测试")

    # ---- 测试 Log-Rank ----
    print(f"\n{'='*50}")
    print("2. 对数秩检验测试")
    print(f"{'='*50}")
    lr_p = cox_log_rank(hazard_pred, survtime, event)
    print(f"  Log-rank p-value: {lr_p:.6f}")
    print(f"  显著 (p<0.05): {lr_p < 0.05}")

    # ---- 测试 时间依赖AUC ----
    print(f"\n{'='*50}")
    print("3. 时间依赖AUC测试")
    print(f"{'='*50}")
    auc_result = safe_time_dependent_auc(survtime, event, hazard_pred,
                                         n_times=20)
    print(f"  Mean AUC: {auc_result.get('mean_auc', 'N/A')}")
    if isinstance(auc_result.get('mean_auc'), float):
        print(f"  Mean AUC: {auc_result['mean_auc']:.4f}")

    # ---- 测试 风险比 ----
    print(f"\n{'='*50}")
    print("4. 风险比测试")
    print(f"{'='*50}")
    hr = safe_hazard_ratio_by_median_split(hazard_pred, survtime, event,
                                           return_details=False)
    print(f"  Hazard Ratio: {hr:.4f}")

    hr_details = safe_hazard_ratio_by_median_split(hazard_pred, survtime, event,
                                                    return_details=True)
    print(f"  方法: {hr_details.get('method', 'N/A')}")
    print(f"  高风险组: n={hr_details['n_high']}, 事件={hr_details['events_high']}")
    print(f"  低风险组: n={hr_details['n_low']}, 事件={hr_details['events_low']}")

    # ---- 测试 二分类指标 ----
    print(f"\n{'='*50}")
    print("5. 二分类指标测试")
    print(f"{'='*50}")
    bin_metrics = safe_binary_metrics_from_risk(
        hazard_pred, survtime, event, time_threshold=365
    )
    for k, v in bin_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")

    # ---- 测试 分组摘要 ----
    print(f"\n{'='*50}")
    print("6. 分组生存摘要测试")
    print(f"{'='*50}")
    summary = safe_group_survival_summary(hazard_pred, survtime, event)
    print(f"  高风险组事件率: {summary['high_risk']['event_rate']:.3f}")
    print(f"  低风险组事件率: {summary['low_risk']['event_rate']:.3f}")
    print(f"  Log-rank p: {summary['log_rank_p_value']:.6f}")
    print(f"  C-index: {summary['c_index']:.4f}")

    # ---- 测试 accuracy_cox ----
    print(f"\n{'='*50}")
    print("7. accuracy_cox 测试")
    print(f"{'='*50}")
    acc_cox_result = accuracy_cox(hazard_pred, survtime, event,
                                  time_threshold=365 * 2)
    for k, v in acc_cox_result.items():
        print(f"  {k}: {v}")

    # ---- 测试 一站式评估 ----
    print(f"\n{'='*50}")
    print("8. 一站式评估测试")
    print(f"{'='*50}")
    all_results = evaluate_survival_model(hazard_pred, survtime, event,
                                          prefix="test_")
    for k, v in all_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # ---- 测试 边缘情况 ----
    print(f"\n{'='*50}")
    print("9. 边缘情况测试")
    print(f"{'='*50}")

    # 空数据
    print("  空数据 C-index:", CIndex([], [], []))
    print("  单样本 C-index:", CIndex([0.5], [100], [1]))

    # 全删失
    all_censored = np.zeros(10, dtype=bool)
    print(f"  全删失 Log-rank p: {cox_log_rank(np.random.randn(10), np.arange(10)*100, all_censored)}")

    # 全事件
    all_event = np.ones(10, dtype=bool)
    print(f"  全事件 C-index: {CIndex(np.random.randn(10), np.arange(10)*100, all_event):.4f}")

    # 全相同风险
    same_risk = np.ones(10) * 0.5
    print(f"  全相同风险 C-index: {CIndex(same_risk, np.arange(10)*100, all_event):.4f}")

    # 测试辅助函数
    print(f"\n{'='*50}")
    print("10. 辅助函数测试")
    print(f"{'='*50}")

    # _to_numpy_1d
    from_list = _to_numpy_1d([1, 2, 3], "test")
    print(f"  list->numpy: {from_list.shape}, dtype={from_list.dtype}")

    # suggest_eval_times
    suggested = suggest_eval_times_from_data(survtime, event, n_times=5)
    print(f"  建议评估时间点: {suggested}")

    # _safe_div
    print(f"  安全除法 5/2: {_safe_div(5, 2)}")
    print(f"  安全除法 5/0: {_safe_div(5, 0, default=-1)}")
    print(f"  安全除法 0/0: {_safe_div(0, 0, default=float('nan'))}")

    print(f"\n{'='*70}")
    print("所有自测完成!")
    print(f"{'='*70}")
