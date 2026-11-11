# -*- coding: utf-8 -*-
"""
diagnose_features.py - 线性探针: 定位粘液二分类失败的原因。

不做 MIL / 不做训练, 直接把每个病人的 patch 特征做逻辑回归, 看特征里有没有信号。

运行:  python diagnose_features.py
输出:
  1) 原始像素探针 (颜色/纹理, 6维)      —— 是否有任何肉眼可见的颜色差异
  2) 冻结ResNet探针 (ImageNet 2048维)   —— ImageNet 特征里是否有黏液信号

判读:
  - ResNet探针 AUC > 0.70  -> 特征没问题, 是 MIL 训练塌缩了 (修训练/损失)
  - ResNet探针 AUC ~0.50  -> 特征没信号, 去查 patch 内容 / 换 UNI
  - 原始像素探针高但ResNet低 -> ImageNet 特征不适用, 换病理基础模型
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from config.config_loader import load_config
from src.networks.backbone import build_backbone
from src.data_loading.upstream_reader import UpstreamPatchReader


def _transform():
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std)])


def _extract(backbone, patch_list, device, tf):
    feats, raw = [], []
    for p in patch_list:
        img = Image.open(p).convert("RGB")
        # 原始像素特征: 缩小到 64x64 后的 RGB 均值+标准差 (6维)
        arr = np.asarray(img.resize((64, 64)), dtype=np.float32) / 255.0
        raw.append(np.concatenate([arr.reshape(-1, 3).mean(0), arr.reshape(-1, 3).std(0)]))
        with torch.no_grad():
            x = tf(img).unsqueeze(0).to(device)
            feats.append(backbone(x).cpu().numpy().reshape(-1))
    return np.mean(np.stack(feats), axis=0), np.mean(np.stack(raw), axis=0)


def _probe(X, y, name):
    if len(np.unique(y)) < 2 or len(y) < 20:
        print(f"  {name}: 样本/类别不足, 跳过")
        return
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(class_weight="balanced", max_iter=3000)
        clf.fit(X[tr], y[tr])
        if len(np.unique(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    print(f"  {name}: n={len(y)} (正例{int(y.sum())})  AUC={np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")


def main():
    config = load_config("config/default_config.yaml", cli_args=[])
    config.project.task = "classification"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"table={config.resolve_path(config.data.table_csv)}")
    print(f"patch={config.resolve_path(config.upstream.patch_root) / config.upstream.algorithm_name}")

    backbone, feat_dim = build_backbone(config)
    backbone.eval().to(device)
    tf = _transform()

    table = pd.read_csv(config.resolve_path(config.data.table_csv), encoding="utf-8-sig")
    pid_col = config.data.patient_id_col
    hist = config.data.histology_col
    table[pid_col] = table[pid_col].astype(str).str.strip()
    labels = dict(zip(table[pid_col],
                      table[hist].astype(str).str.lower()
                      .str.contains(config.data.mucinous_keyword).astype(int)))

    patch_map = UpstreamPatchReader(config).build_short_to_patches()

    X_res, X_raw, y = [], [], []
    n_fail = 0
    for pid, patches in patch_map.items():
        if pid not in labels:
            continue
        try:
            f_res, f_raw = _extract(backbone, patches, device, tf)
        except Exception:
            n_fail += 1
            continue
        X_res.append(f_res)
        X_raw.append(f_raw)
        y.append(labels[pid])

    y = np.asarray(y, dtype=int)
    print(f"\n有效病人(有patch+标签): {len(y)}  正例={int(y.sum())}  负例={int((y == 0).sum())}  失败={n_fail}\n")

    _probe(np.stack(X_raw), y, "原始像素探针 (颜色6维)")
    _probe(np.stack(X_res), y, "冻结ResNet探针 (2048维)")
    print("\n判读见脚本头部注释。")


if __name__ == "__main__":
    main()
