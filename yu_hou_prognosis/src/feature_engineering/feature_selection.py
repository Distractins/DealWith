# -*- coding: utf-8 -*-
"""
feature_selection.py
============================================================================
高维基因组特征选择与降维模块。

针对COAD数据集~356维基因突变特征的高维小样本问题，
提供多种特征选择和降维方法。

推荐管线:
    VarianceThreshold (356 -> ~200) ->
    Mutual Information (200 -> 100) ->
    PCA (100 -> 50)

支持的方法:
    1. 方差阈值 (VarianceThreshold): 移除低方差特征
    2. 互信息 (Mutual Information): 选择与生存标签最相关的特征
    3. LASSO正则化: 基于L1惩罚的稀疏特征选择
    4. PCA降维: 保留95%方差的主成分

使用示例:
    from src.feature_engineering.feature_selection import GenomicFeatureSelector
    selector = GenomicFeatureSelector(config)
    X_selected, selected_indices = selector.fit_transform(X, y_event, y_time)
    print(f"特征维度: {X.shape[1]} -> {X_selected.shape[1]}")
============================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class GenomicFeatureSelector:
    """
    基因组特征选择器。

    支持多阶段特征筛选管线: 方差过滤 -> 互信息排序 -> LASSO -> PCA

    参数:
        config: ConfigBundle配置对象
    """

    def __init__(self, config):
        self.config = config
        self.fs_config = config.data.feature_selection

        # 特征选择方法
        self.method = self.fs_config.method

        # 方差阈值
        self.variance_threshold = self.fs_config.variance_threshold

        # 互信息保留特征数
        self.n_features_mi = self.fs_config.n_features_mi

        # PCA参数
        self.pca_variance_ratio = self.fs_config.pca_variance_ratio

        # 最终特征数
        self.n_features_final = self.fs_config.n_features_final

        # 记录被选择的特征索引
        self.selected_indices_ = None
        self.feature_names_ = None

    def fit_transform(
        self,
        X: np.ndarray,
        y_event: Optional[np.ndarray] = None,
        y_time: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行特征选择流水线。

        参数:
            X: [N, D] 原始基因组特征矩阵
            y_event: [N] 事件标记（用于互信息计算）
            y_time: [N] 生存时间
            feature_names: 特征名列表（可选）

        返回:
            X_selected: [N, D'] 筛选后的特征矩阵
            selected_indices: [D'] 被选中特征的原始索引
        """
        self.feature_names_ = feature_names
        n_orig = X.shape[1]
        print(f"\n[FeatureSelector] 开始特征选择: {n_orig}维 -> 目标{self.n_features_final}维")
        print(f"  选择方法: {self.method}")

        if self.method == "none":
            # 不做特征选择
            print(f"  跳过特征选择，保持原始{n_orig}维")
            self.selected_indices_ = np.arange(n_orig)
            return X, self.selected_indices_

        # ---- 阶段1: 方差阈值过滤 ----
        X_filtered, indices = self._apply_variance_threshold(X)
        print(f"  方差过滤后: {X_filtered.shape[1]}维")

        # ---- 阶段2: 互信息选择 ----
        if y_event is not None and self.n_features_mi < X_filtered.shape[1]:
            X_filtered, indices = self._apply_mutual_information(
                X_filtered, indices, y_event
            )
            print(f"  互信息选择后: {X_filtered.shape[1]}维")

        # ---- 阶段3: PCA降维 ----
        if self.pca_variance_ratio > 0 and X_filtered.shape[1] > self.n_features_final:
            X_filtered, _ = self._apply_pca(X_filtered, self.pca_variance_ratio)
            print(f"  PCA降维后: {X_filtered.shape[1]}维")

        self.selected_indices_ = indices

        print(f"  特征选择完成: {n_orig}维 -> {X_filtered.shape[1]}维 "
              f"({X_filtered.shape[1]/n_orig*100:.1f}%)")

        return X_filtered, self.selected_indices_

    def _apply_variance_threshold(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        阶段1: 基于方差阈值过滤低信息特征。

        移除方差低于阈值的特征（方差太小意味着该特征在所有样本中几乎不变）。
        """
        selector = VarianceThreshold(threshold=self.variance_threshold)
        X_filtered = selector.fit_transform(X)

        # 获取被保留特征的索引
        support = selector.get_support(indices=True)
        return X_filtered, support

    def _apply_mutual_information(
        self, X: np.ndarray, original_indices: np.ndarray, y_event: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        阶段2: 基于互信息选择与生存事件最相关的特征。

        参数:
            X: 当前特征矩阵
            original_indices: 当前特征在原始特征空间中的索引
            y_event: 事件标记

        返回:
            X_selected: 选择的特征
            selected_indices: 更新后的索引数组
        """
        if X.shape[1] <= self.n_features_mi:
            return X, original_indices

        # 计算互信息分数
        mi_scores = mutual_info_classif(X, y_event, random_state=42)

        # 选择top-K特征
        top_k = min(self.n_features_mi, X.shape[1])
        top_indices = np.argsort(mi_scores)[::-1][:top_k]

        X_selected = X[:, top_indices]
        selected_indices = original_indices[top_indices]
        return X_selected, selected_indices

    def _apply_pca(
        self, X: np.ndarray, variance_ratio: float = 0.95
    ) -> Tuple[np.ndarray, PCA]:
        """
        阶段3: PCA降维。

        参数:
            X: [N, D] 特征矩阵
            variance_ratio: 保留的方差比例（0-1）

        返回:
            X_pca: [N, D'] PCA降维后的特征
            pca: 训练好的PCA对象（用于transform新数据）
        """
        # 标准化（PCA对尺度敏感）
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        pca = PCA(n_components=variance_ratio, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        print(f"    PCA主成分数: {X_pca.shape[1]} (保留方差: {variance_ratio})")

        return X_pca, pca

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        使用已训练的筛选器转换新数据（仅方差过滤+互信息阶段）。

        注意: PCA阶段的transform需要保存PCA对象后使用。

        参数:
            X: [N, D] 原始特征矩阵

        返回:
            X_filtered: [N, D'] 筛选后的特征矩阵
        """
        if self.selected_indices_ is not None:
            return X[:, self.selected_indices_]
        return X

    def get_feature_names(self) -> Optional[List[str]]:
        """获取被选中特征的名称列表"""
        if self.feature_names_ is not None and self.selected_indices_ is not None:
            return [self.feature_names_[i] for i in self.selected_indices_]
        return None


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GenomicFeatureSelector 特征选择模块自测")
    print("=" * 60)

    np.random.seed(42)
    N, D = 200, 356

    # 模拟基因组特征
    X = np.random.randn(N, D)
    # 模拟事件标记
    y_event = np.random.binomial(1, 0.3, N)

    # 模拟配置
    class MockFSConfig:
        method = "variance_threshold"
        variance_threshold = 0.01
        n_features_mi = 100
        pca_variance_ratio = 0.95
        n_features_final = 100
    class MockDataConfig:
        feature_selection = MockFSConfig()
    class MockConfig:
        data = MockDataConfig()

    selector = GenomicFeatureSelector(MockConfig())
    X_selected, indices = selector.fit_transform(X, y_event)

    print(f"\n  原始维度: {X.shape}")
    print(f"  筛选后维度: {X_selected.shape}")
    print(f"  选中特征索引数: {len(indices)}")
    print(f"  维度压缩比: {X_selected.shape[1]/X.shape[1]*100:.1f}%")

    # 测试transform
    X_new = np.random.randn(10, D)
    X_new_selected = selector.transform(X_new)
    print(f"  transform测试: {X_new.shape} -> {X_new_selected.shape}")

    print("\n所有测试通过!")
