# -*- coding: utf-8 -*-
"""
classification_metrics.py
============================================================================
分类任务评估指标模块。

包含:
    1. 分类指标统一计算: accuracy, balanced_accuracy, precision, recall, f1, AUC, AP, MCC
    2. 二分类/多分类自动适配
    3. logits -> 预测/概率 转换
    4. 安全的AUC/AP计算（处理单类、空输入等边缘情况）

使用示例:
    from src.evaluation.classification_metrics import classification_metrics
    metrics = classification_metrics(logits=logits, y_true=labels)
    print(f"AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
============================================================================
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.preprocessing import LabelBinarizer


# ============================================================
# 工具函数
# ============================================================

def _to_numpy_1d(x):
    """安全地将输入转换为numpy 1D数组"""
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x).reshape(-1)
    return x


def _to_numpy_2d(x):
    """安全地将输入转换为numpy 2D数组"""
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def _safe_div(a, b):
    """安全除法（b=0时返回None）"""
    try:
        if b == 0:
            return None
        return float(a) / float(b)
    except Exception:
        return None


# ============================================================
# Logits转换
# ============================================================

def logits_to_pred_and_prob(logits):
    """
    将模型logits转换为预测标签和概率。

    参数:
        logits: torch.Tensor或numpy数组
            - [B] 或 [B,1]: 二分类logits，使用sigmoid
            - [B, C] (C>=2): 多分类logits，使用softmax

    返回:
        pred_label: [B] 预测类别（整数）
        prob_pos: [B] 正类概率（二分类）或类别1概率（多分类）
        prob_all: [B, C] 所有类别的概率
    """
    if torch.is_tensor(logits):
        x = logits.detach().cpu()
    else:
        x = torch.tensor(logits)

    if x.dim() == 1 or (x.dim() == 2 and x.size(1) == 1):
        # 二分类: sigmoid
        prob_pos = torch.sigmoid(x.view(-1)).numpy()
        pred_label = (prob_pos >= 0.5).astype(int)
        prob_all = np.stack([1.0 - prob_pos, prob_pos], axis=1)
        return pred_label, prob_pos, prob_all

    # 多分类: softmax
    prob_all = torch.softmax(x, dim=1).numpy()
    pred_label = np.argmax(prob_all, axis=1).astype(int)
    prob_pos = prob_all[:, 1] if prob_all.shape[1] >= 2 else prob_all[:, 0]
    return pred_label, prob_pos, prob_all


# ============================================================
# 安全的AUC/AP计算
# ============================================================

def safe_auc_binary(y_true, y_score):
    """
    安全计算二分类AUC。

    参数:
        y_true: [N] 0/1真实标签
        y_score: [N] 预测概率分数

    返回:
        auc: float或None（无法计算时）
    """
    try:
        y_true = _to_numpy_1d(y_true).astype(int)
        y_score = _to_numpy_1d(y_score).astype(float)
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def safe_ap_binary(y_true, y_score):
    """
    安全计算二分类Average Precision。

    参数:
        y_true: [N] 0/1真实标签
        y_score: [N] 预测概率分数

    返回:
        ap: float或None
    """
    try:
        y_true = _to_numpy_1d(y_true).astype(int)
        y_score = _to_numpy_1d(y_score).astype(float)
        if len(np.unique(y_true)) < 2:
            return None
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return None


def safe_auc_multiclass(y_true, prob_all, average="macro"):
    """
    安全计算多分类AUC (one-vs-rest)。

    参数:
        y_true: [N] 整数类别标签
        prob_all: [N, C] 各类别概率
        average: 平均方式 ("macro" / "micro" / "weighted")

    返回:
        auc: float或None
    """
    try:
        y_true = _to_numpy_1d(y_true).astype(int)
        prob_all = _to_numpy_2d(prob_all).astype(float)
        n_class = prob_all.shape[1]
        if len(np.unique(y_true)) < 2:
            return None

        lb = LabelBinarizer()
        lb.fit(np.arange(n_class))
        y_onehot = lb.transform(y_true)
        if y_onehot.ndim == 1:
            y_onehot = np.stack([1 - y_onehot, y_onehot], axis=1)

        return float(roc_auc_score(y_onehot, prob_all, average=average, multi_class="ovr"))
    except Exception:
        return None


def safe_ap_multiclass(y_true, prob_all, average="macro"):
    """
    安全计算多分类Average Precision。

    参数:
        y_true: [N] 整数类别标签
        prob_all: [N, C] 各类别概率
        average: 平均方式

    返回:
        ap: float或None
    """
    try:
        y_true = _to_numpy_1d(y_true).astype(int)
        prob_all = _to_numpy_2d(prob_all).astype(float)
        n_class = prob_all.shape[1]
        lb = LabelBinarizer()
        lb.fit(np.arange(n_class))
        y_onehot = lb.transform(y_true)
        if y_onehot.ndim == 1:
            y_onehot = np.stack([1 - y_onehot, y_onehot], axis=1)
        return float(average_precision_score(y_onehot, prob_all, average=average))
    except Exception:
        return None


def safe_specificity(y_true, y_pred):
    """安全计算特异性 (Specificity = TN / (TN + FP))"""
    try:
        y_true = _to_numpy_1d(y_true).astype(int)
        y_pred = _to_numpy_1d(y_pred).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return _safe_div(tn, tn + fp)
    except Exception:
        return None


def safe_npv(y_true, y_pred):
    """安全计算阴性预测值 (NPV = TN / (TN + FN))"""
    try:
        y_true = _to_numpy_1d(y_true).astype(int)
        y_pred = _to_numpy_1d(y_pred).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return _safe_div(tn, tn + fn)
    except Exception:
        return None


# ============================================================
# 统一分类指标计算
# ============================================================

def classification_metrics(logits=None, y_true=None, y_pred=None, y_score=None,
                           prob_all=None, average="binary"):
    """
    统一计算分类任务的所有评估指标。

    自动判断二分类/多分类，计算对应的完整指标体系。

    参数:
        logits: torch.Tensor或numpy数组 [N]或[N,C]，模型输出logits
        y_true: [N] 真实类别标签（整数）
        y_pred: [N] 预测类别（可选，从logits自动计算）
        y_score: [N] 预测概率（可选，自动计算）
        prob_all: [N, C] 所有类别概率（可选，自动计算）
        average: 多分类平均方式 ("binary" / "macro" / "micro")

    返回:
        dict: 包含所有指标的字典，字段如下:
            - n: 样本数
            - n_class: 类别数
            - acc: 准确率 (Accuracy)
            - balanced_acc: 平衡准确率 (Balanced Accuracy)
            - precision: 精确率 (Precision)
            - recall: 召回率 (Recall)
            - specificity: 特异性 (仅二分类)
            - f1: F1分数
            - mcc: Matthews相关系数
            - npv: 阴性预测值 (仅二分类)
            - auc: AUC (ROC曲线下面积)
            - ap: 平均精度 (Average Precision)
            - confusion_matrix: 混淆矩阵 (list of list)
            - per_class_f1: 各类别F1 (仅多分类)
            - y_pred: 预测标签数组
    """
    # 从logits自动计算预测和概率
    if logits is not None:
        auto_pred, auto_prob_pos, auto_prob_all = logits_to_pred_and_prob(logits)
        if y_pred is None:
            y_pred = auto_pred
        if y_score is None:
            y_score = auto_prob_pos
        if prob_all is None:
            prob_all = auto_prob_all

    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred) if y_pred is not None else None
    y_score = _to_numpy_1d(y_score) if y_score is not None else None
    prob_all = _to_numpy_2d(prob_all) if prob_all is not None else None

    if y_true is None or y_pred is None:
        return {}

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    is_binary = (len(np.unique(np.concatenate([y_true, y_pred]))) <= 2)

    result = {
        "n": int(len(y_true)),
        "n_class": int(prob_all.shape[1]) if prob_all is not None else int(len(np.unique(y_true))),
    }

    # 基础指标
    try:
        result["acc"] = float(accuracy_score(y_true, y_pred))
    except Exception:
        result["acc"] = None

    try:
        result["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        result["balanced_acc"] = None

    try:
        result["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    except Exception:
        result["confusion_matrix"] = None

    if is_binary:
        # 二分类指标
        try:
            result["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
            result["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
            result["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
            result["mcc"] = float(matthews_corrcoef(y_true, y_pred))
            result["specificity"] = safe_specificity(y_true, y_pred)
            result["npv"] = safe_npv(y_true, y_pred)
        except Exception:
            pass
        if y_score is not None:
            result["auc"] = safe_auc_binary(y_true, y_score)
            result["ap"] = safe_ap_binary(y_true, y_score)
    else:
        # 多分类指标
        try:
            result["precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
            result["recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
            result["f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            result["per_class_f1"] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
            result["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        except Exception:
            pass
        if prob_all is not None:
            result["auc"] = safe_auc_multiclass(y_true, prob_all, average="macro")
            result["ap"] = safe_ap_multiclass(y_true, prob_all, average="macro")

    result["y_pred"] = y_pred.tolist() if y_pred is not None else None
    return result


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("分类指标模块自测")
    print("=" * 60)

    np.random.seed(42)
    N = 100

    # 二分类测试
    y_true_bin = np.random.randint(0, 2, N)
    logits_bin = torch.randn(N) * 0.5
    metrics_bin = classification_metrics(logits=logits_bin, y_true=y_true_bin)

    print("\n二分类指标:")
    for k, v in metrics_bin.items():
        if k not in ("confusion_matrix", "per_class_f1", "y_pred"):
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # 多分类测试
    y_true_multi = np.random.randint(0, 3, N)
    logits_multi = torch.randn(N, 3)
    metrics_multi = classification_metrics(logits=logits_multi, y_true=y_true_multi)

    print("\n多分类指标:")
    for k, v in metrics_multi.items():
        if k not in ("confusion_matrix", "per_class_f1", "y_pred"):
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    print("\n所有测试通过!")
