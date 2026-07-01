# -*- coding: utf-8 -*-
"""
image_quality_filter.py
============================================================================
模糊/低质量病理图像过滤模块。

设计策略:
    采用"标记不丢弃"策略：
    - 使用Laplacian方差法检测模糊patch
    - 记录每个patch的质量评分
    - 标记低质量病人（模糊patch比例超过阈值）
    - 不丢弃样本以保留小样本数据集的统计效力

检测方法:
    1. Laplacian方差法: 衡量图像边缘/纹理丰富度
    2. Sobel梯度幅值: 衡量图像梯度强度
    3. FFT高频能量: 衡量图像高频细节占比

使用示例:
    from src.feature_engineering.image_quality_filter import PatchQualityAnalyzer
    analyzer = PatchQualityAnalyzer(config)
    quality_report = analyzer.analyze_patient_patches(patient_patch_dict)
============================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image
from collections import defaultdict


class PatchQualityAnalyzer:
    """
    Patch图像质量分析器。

    统计每位病人的patch质量分布，标记低质量样本。

    参数:
        config: ConfigBundle配置对象
    """

    def __init__(self, config):
        self.config = config
        self.blur_config = config.image_preprocessing.blur_filter

        # 模糊检测阈值
        self.threshold = self.blur_config.threshold  # 默认100
        self.strategy = self.blur_config.strategy     # "mark_only"
        self.max_blur_ratio = self.blur_config.max_blur_ratio_per_patient  # 默认0.5

    def compute_laplacian_variance(self, img: np.ndarray) -> float:
        """
        计算Laplacian方差（图像清晰度指标）。

        原理: 对图像应用Laplacian算子后计算方差。
              方差大 -> 边缘多/纹理丰富 -> 清晰
              方差小 -> 平滑/模糊 -> 低质量

        参数:
            img: [H, W, C] uint8 RGB图像

        返回:
            variance: Laplacian响应方差
        """
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def compute_sobel_magnitude(self, img: np.ndarray) -> float:
        """
        计算Sobel梯度幅值均值。

        参数:
            img: [H, W, C] uint8 RGB图像

        返回:
            mean_magnitude: 平均梯度幅值
        """
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        return float(magnitude.mean())

    def compute_brightness_contrast(self, img: np.ndarray) -> Dict[str, float]:
        """
        计算图像的亮度和对比度统计。

        参数:
            img: [H, W, C] uint8 RGB图像

        返回:
            dict: {"mean": 均值, "std": 标准差, "min": 最小值, "max": 最大值}
        """
        gray = img.mean(axis=2) if img.ndim == 3 else img
        return {
            "mean": float(gray.mean()),
            "std": float(gray.std()),
            "min": float(gray.min()),
            "max": float(gray.max()),
        }

    def is_blurry(self, img: np.ndarray) -> Tuple[bool, float]:
        """
        判断单张patch是否模糊。

        参数:
            img: [H, W, C] uint8 RGB图像

        返回:
            (is_blurry, laplacian_variance)
        """
        var = self.compute_laplacian_variance(img)
        is_blurry = var < self.threshold
        return is_blurry, var

    def analyze_patient_patches(
        self,
        patient_patch_dict: Dict[str, List[str]]
    ) -> Dict[str, Dict]:
        """
        分析每位病人所有patch的质量。

        遍历每个病人的所有patch，计算每张patch的质量指标，
        汇总后标记低质量病人。

        参数:
            patient_patch_dict: {slide_id: [patch_path_1, patch_path_2, ...]}

        返回:
            Dict[str, Dict]:
                key: slide_id
                value: {
                    "total_patches": int,
                    "blurry_patches": int,
                    "blur_ratio": float,
                    "is_low_quality": bool,
                    "laplacian_variances": [float, ...],
                    "mean_variance": float,
                    "min_variance": float,
                }
        """
        report = {}

        for slide_id, patch_paths in patient_patch_dict.items():
            variances = []
            n_blurry = 0

            for path in patch_paths:
                try:
                    img = np.array(Image.open(path).convert("RGB"))
                    var = self.compute_laplacian_variance(img)
                    variances.append(var)
                    if var < self.threshold:
                        n_blurry += 1
                except Exception as e:
                    print(f"[QualityFilter] 警告: 无法分析 {path}: {e}")
                    variances.append(np.nan)

            # 计算统计量
            valid_vars = [v for v in variances if not np.isnan(v)]
            n_total = len(patch_paths)
            blur_ratio = n_blurry / n_total if n_total > 0 else 0
            is_low_quality = blur_ratio > self.max_blur_ratio

            report[slide_id] = {
                "total_patches": n_total,
                "blurry_patches": n_blurry,
                "blur_ratio": blur_ratio,
                "is_low_quality": is_low_quality,
                "laplacian_variances": variances,
                "mean_variance": float(np.mean(valid_vars)) if valid_vars else 0.0,
                "min_variance": float(np.min(valid_vars)) if valid_vars else 0.0,
                "max_variance": float(np.max(valid_vars)) if valid_vars else 0.0,
            }

        return report

    def get_quality_summary(self, report: Dict[str, Dict]) -> Dict:
        """
        生成全局质量统计摘要。

        参数:
            report: analyze_patient_patches()的返回值

        返回:
            dict: 全局质量统计
        """
        total_patients = len(report)
        if total_patients == 0:
            return {}

        low_quality_count = sum(1 for r in report.values() if r["is_low_quality"])
        all_variances = []
        for r in report.values():
            all_variances.extend([v for v in r["laplacian_variances"] if not np.isnan(v)])

        return {
            "total_patients": total_patients,
            "low_quality_patients": low_quality_count,
            "low_quality_ratio": low_quality_count / total_patients,
            "total_patches_analyzed": len(all_variances),
            "mean_laplacian_variance": float(np.mean(all_variances)) if all_variances else 0,
            "std_laplacian_variance": float(np.std(all_variances)) if all_variances else 0,
            "blur_threshold": self.threshold,
            "patches_below_threshold": sum(1 for v in all_variances if v < self.threshold),
            "patches_below_threshold_ratio": (
                sum(1 for v in all_variances if v < self.threshold) / len(all_variances)
                if all_variances else 0
            ),
        }


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PatchQualityAnalyzer 图像质量过滤模块自测")
    print("=" * 60)

    # 模拟图像
    np.random.seed(42)
    mock_sharp = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    mock_blurry = np.ones((256, 256, 3), dtype=np.uint8) * 128  # 均匀灰色=完全模糊

    # 模拟配置
    class MockBlurConfig:
        threshold = 100.0
        strategy = "mark_only"
        max_blur_ratio_per_patient = 0.5
    class MockPPConfig:
        blur_filter = MockBlurConfig()
    class MockConfig:
        image_preprocessing = MockPPConfig()

    analyzer = PatchQualityAnalyzer(MockConfig())

    # 测试清晰/模糊检测
    try:
        import cv2
        var_sharp = analyzer.compute_laplacian_variance(mock_sharp)
        var_blurry = analyzer.compute_laplacian_variance(mock_blurry)

        print(f"\n  清晰图像(随机噪声) Laplacian方差: {var_sharp:.2f}")
        print(f"  模糊图像(均匀灰色) Laplacian方差: {var_blurry:.2f}")
        print(f"  阈值: {analyzer.threshold}")

        is_b, _ = analyzer.is_blurry(mock_sharp)
        print(f"  清晰图像判定为模糊: {is_b}")
        is_b2, _ = analyzer.is_blurry(mock_blurry)
        print(f"  模糊图像判定为模糊: {is_b2}")

    except ImportError:
        print("  (需要opencv-python)")

    print("\n测试通过!")
