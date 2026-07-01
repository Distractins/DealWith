# -*- coding: utf-8 -*-
"""
image_quality_eda.py
============================================================================
病理图像质量探索性数据分析 (EDA) 模块。

分析内容:
    1. 抽样显示patch图像网格
    2. 质量指标分布: Laplacian方差/亮度/对比度
    3. 模糊样本检测与标记
    4. 颜色分布分析 (RGB均值/标准差)
    5. 病人间patch一致性分析
    6. 生成图像质量评估报告

使用场景:
    运行上游WSI切块任务后，对生成的patch图像进行质量评估，
    识别低质量图像对模型训练的影响。

使用示例:
    python -m src.eda.image_quality_eda --patch_dir ../wsi_patch_selection_benchmark/outputs/patches/grid
============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from PIL import Image


try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


class ImageQualityEDA:
    """
    病理Patch图像质量探索性数据分析器。

    参数:
        patch_dir: 上游输出的patch图像目录
        output_dir: EDA输出目录
        blur_threshold: Laplacian方差模糊阈值 (默认100)
    """

    def __init__(
        self,
        patch_dir: str,
        output_dir: str = "experiments/eda",
        blur_threshold: float = 100.0,
    ):
        self.patch_dir = Path(patch_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.blur_threshold = blur_threshold

        # 质量指标缓存
        self.quality_metrics: Dict[str, List[Dict]] = defaultdict(list)

    def scan_patches(self) -> List[Path]:
        """扫描patch目录下的所有PNG文件"""
        if not self.patch_dir.exists():
            raise FileNotFoundError(f"Patch目录不存在: {self.patch_dir}")

        png_files = sorted(self.patch_dir.glob("*.png"))
        print(f"[ImageQualityEDA] 找到 {len(png_files)} 个PNG文件")
        return png_files

    def compute_quality_metrics(self, img: np.ndarray) -> Dict:
        """计算单张图像的质量指标"""
        import cv2

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Laplacian方差（清晰度）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = float(laplacian.var())

        # Sobel梯度（边缘强度）
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = float(np.sqrt(sobel_x**2 + sobel_y**2).mean())

        # 亮度统计
        brightness_mean = float(gray.mean())
        brightness_std = float(gray.std())

        # RGB通道统计
        r_mean, g_mean, b_mean = img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()
        r_std, g_std, b_std = img[:,:,0].std(), img[:,:,1].std(), img[:,:,2].std()

        return {
            "laplacian_var": laplacian_var,
            "gradient_mag": gradient_mag,
            "brightness_mean": brightness_mean,
            "brightness_std": brightness_std,
            "r_mean": r_mean, "g_mean": g_mean, "b_mean": b_mean,
            "r_std": r_std, "g_std": g_std, "b_std": b_std,
            "is_blurry": laplacian_var < self.blur_threshold,
        }

    def analyze_samples(self, n_samples: int = 50) -> Dict:
        """
        抽样分析patch图像质量。

        参数:
            n_samples: 随机抽样数量

        返回:
            dict: 质量统计摘要
        """
        import cv2
        png_files = self.scan_patches()

        if len(png_files) == 0:
            return {}

        # 随机抽样
        n_analyze = min(n_samples, len(png_files))
        indices = np.random.choice(len(png_files), n_analyze, replace=False)
        sample_files = [png_files[i] for i in indices]

        print(f"[ImageQualityEDA] 抽样分析 {n_analyze} 张图像...")

        all_metrics = []
        for f in sample_files:
            try:
                img = np.array(Image.open(f).convert("RGB"))
                metrics = self.compute_quality_metrics(img)
                metrics["filename"] = f.name
                all_metrics.append(metrics)
            except Exception as e:
                print(f"  警告: 无法分析 {f.name}: {e}")

        if not all_metrics:
            return {}

        # 统计摘要
        laplacian_vars = [m["laplacian_var"] for m in all_metrics]
        brightnesses = [m["brightness_mean"] for m in all_metrics]
        n_blurry = sum(1 for m in all_metrics if m["is_blurry"])

        summary = {
            "分析图像数": len(all_metrics),
            "模糊图像数": n_blurry,
            "模糊比例": f"{n_blurry / len(all_metrics):.1%}",
            "Laplacian方差均值": f"{np.mean(laplacian_vars):.1f}",
            "Laplacian方差中位数": f"{np.median(laplacian_vars):.1f}",
            "Laplacian方差标准差": f"{np.std(laplacian_vars):.1f}",
            "亮度均值": f"{np.mean(brightnesses):.1f}",
            "亮度标准差": f"{np.std(brightnesses):.1f}",
        }

        print("\n  图像质量统计摘要:")
        for k, v in summary.items():
            print(f"    {k}: {v}")

        # 绘制质量分布图
        self._plot_quality_distribution(all_metrics)
        self._plot_color_distribution(all_metrics)

        return summary

    def _plot_quality_distribution(self, metrics: List[Dict]):
        """绘制质量指标分布图"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        laplacian_vars = [m["laplacian_var"] for m in metrics]
        brightness = [m["brightness_mean"] for m in metrics]
        gradient = [m["gradient_mag"] for m in metrics]

        # Laplacian方差
        axes[0].hist(laplacian_vars, bins=30, color='steelblue', edgecolor='white')
        axes[0].axvline(x=self.blur_threshold, color='red', linestyle='--',
                       label=f'模糊阈值 ({self.blur_threshold})')
        axes[0].set_xlabel('Laplacian方差')
        axes[0].set_ylabel('图像数')
        axes[0].set_title('清晰度分布 (Laplacian方差)')
        axes[0].legend()

        # 亮度分布
        axes[1].hist(brightness, bins=30, color='seagreen', edgecolor='white')
        axes[1].set_xlabel('平均亮度')
        axes[1].set_title('亮度分布')

        # 梯度幅值
        axes[2].hist(gradient, bins=30, color='coral', edgecolor='white')
        axes[2].set_xlabel('梯度幅值均值')
        axes[2].set_title('边缘强度分布 (Sobel梯度)')

        plt.tight_layout()
        save_path = self.output_dir / "image_quality_distribution.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  质量分布图已保存: {save_path}")

    def _plot_color_distribution(self, metrics: List[Dict]):
        """绘制颜色分布图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # RGB均值分布
        r_means = [m["r_mean"] for m in metrics]
        g_means = [m["g_mean"] for m in metrics]
        b_means = [m["b_mean"] for m in metrics]

        axes[0].boxplot([r_means, g_means, b_means], labels=['R', 'G', 'B'])
        axes[0].set_ylabel('像素均值')
        axes[0].set_title('RGB通道均值分布')

        # RGB标准差分布
        r_stds = [m["r_std"] for m in metrics]
        g_stds = [m["g_std"] for m in metrics]
        b_stds = [m["b_std"] for m in metrics]

        axes[1].boxplot([r_stds, g_stds, b_stds], labels=['R', 'G', 'B'])
        axes[1].set_ylabel('像素标准差')
        axes[1].set_title('RGB通道标准差分布')

        plt.tight_layout()
        save_path = self.output_dir / "color_distribution.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  颜色分布图已保存: {save_path}")

    def run_full_eda(self) -> Dict:
        """运行完整的图像质量EDA分析"""
        print("=" * 60)
        print("病理图像质量探索性数据分析 (EDA)")
        print("=" * 60)
        print(f"  Patch目录: {self.patch_dir}")
        print(f"  模糊阈值: {self.blur_threshold}")

        summary = self.analyze_samples(n_samples=100)
        print("\n图像质量EDA完成!")
        return summary


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    import sys

    patch_dir = sys.argv[1] if len(sys.argv) > 1 else "../wsi_patch_selection_benchmark/outputs/patches/grid"

    if Path(patch_dir).exists():
        eda = ImageQualityEDA(patch_dir)
        eda.run_full_eda()
    else:
        print(f"Patch目录不存在: {patch_dir}")
        print("请确保已运行上游WSI切块任务")
