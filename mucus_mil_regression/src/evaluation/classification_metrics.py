# -*- coding: utf-8 -*-
"""classification_metrics.py - 二分类评估指标。

输入: logits + 0/1 标签; 返回 {auc, acc, f1, sens, spec, n}。
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix


def _to_numpy(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def classification_metrics(logits, target) -> dict:
    logits = _to_numpy(logits)
    target = _to_numpy(target)
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))   # sigmoid
    pred = (prob >= 0.5).astype(int)

    result = {"n": int(len(target))}
    if len(target) < 2:
        for k in ["auc", "acc", "f1", "sens", "spec"]:
            result[k] = float("nan")
        return result

    if len(np.unique(target)) > 1:
        try:
            result["auc"] = float(roc_auc_score(target, prob))
        except Exception:
            result["auc"] = float("nan")
    else:
        result["auc"] = float("nan")

    result["acc"] = float(accuracy_score(target, pred))
    try:
        result["f1"] = float(f1_score(target, pred, zero_division=0))
    except Exception:
        result["f1"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(target, pred, labels=[0, 1]).ravel()
    result["sens"] = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")   # 正例召回
    result["spec"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")   # 负例特异度
    return result
