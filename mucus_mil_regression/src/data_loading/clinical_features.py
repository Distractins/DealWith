# -*- coding: utf-8 -*-
"""
clinical_features.py - 临床协变量编码。

数值特征标准化 + 分类特征OneHot, 输出固定维度向量。
训练集上fit, 再transform到train/val/test (避免泄露)。
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import List, Optional


class ClinicalEncoder:
    """临床特征编码器 (数值标准化 + 分类OneHot)。"""

    def __init__(self, feature_cols: List[str]):
        self.feature_cols = list(feature_cols)
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.preprocessor: Optional[ColumnTransformer] = None
        self.output_dim = 0

    def _split_cols(self, df: pd.DataFrame):
        self.numeric_cols = [c for c in self.feature_cols if c in df.columns
                             and pd.api.types.is_numeric_dtype(df[c])]
        self.categorical_cols = [c for c in self.feature_cols if c in df.columns
                                 and c not in self.numeric_cols]

    def fit(self, df: pd.DataFrame) -> "ClinicalEncoder":
        self._split_cols(df)
        transformers = []
        if self.numeric_cols:
            transformers.append(("num", StandardScaler(), self.numeric_cols))
        if self.categorical_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                                      sparse_output=False), self.categorical_cols))
        self.preprocessor = ColumnTransformer(transformers, remainder="drop")
        # 分类缺失值填充为"NA", 数值缺失填充为均值(由scaler处理, 这里先填0)
        df_clean = df.copy()
        for c in self.categorical_cols:
            df_clean[c] = df_clean[c].astype(str).fillna("NA")
        for c in self.numeric_cols:
            df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce").fillna(0.0)
        self.preprocessor.fit(df_clean)
        # 估计输出维度
        dummy = self.preprocessor.transform(df_clean.iloc[:1])
        self.output_dim = int(dummy.shape[1])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise RuntimeError("ClinicalEncoder未fit, 请先调用 fit()")
        df_clean = df.copy()
        for c in self.categorical_cols:
            df_clean[c] = df_clean[c].astype(str).fillna("NA")
        for c in self.numeric_cols:
            df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce").fillna(0.0)
        return self.preprocessor.transform(df_clean).astype(np.float32)
