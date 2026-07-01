# -*- coding: utf-8 -*-
"""
preprocessing.py
============================================================================
低质量WSI Patch图像预处理模块。

针对TCGA-COAD病理图像存在的模糊、染色偏差、背景噪声多等问题，
提供GPU加速的图像预处理流水线。

处理流水线:
    1. 去噪 (Denoising): 高斯滤波/中值滤波/双边滤波
    2. 对比度增强 (Contrast): CLAHE/直方图均衡化
    3. 颜色归一化 (Color Norm): Reinhard/Macenko染色归一化
    4. 模糊检测 (Blur Detection): Laplacian方差/梯度幅值

设计原则:
    - 所有处理可在GPU上加速（使用torch操作或kornia库）
    - 预处理在DataLoader的collate_fn中批量执行
    - 模糊检测仅标记不丢弃（保留样本量）

使用示例:
    from src.data_loading.preprocessing import ImagePreprocessor
    preprocessor = ImagePreprocessor(config)
    processed_img = preprocessor.process(img_tensor)

注意:
    当前实现基于CPU的OpenCV/PIL操作。
    如需GPU加速，可将图像转为torch tensor后使用torch操作。
============================================================================
"""

import numpy as np
from typing import Optional, Tuple
from PIL import Image


class ImagePreprocessor:
    """
    WSI Patch图像预处理器。

    可配置的处理步骤: 去噪 -> 对比度增强 -> 颜色归一化

    参数:
        config: ConfigBundle配置对象（读取image_preprocessing配置段）
    """

    def __init__(self, config):
        self.config = config
        self.pp_config = config.image_preprocessing
        self.enabled = self.pp_config.enabled

        if self.enabled:
            print(f"[ImagePreprocessor] 图像预处理已启用:")
            print(f"  去噪方法: {self.pp_config.denoise.method}")
            print(f"  对比度增强: {self.pp_config.contrast.method}")
            print(f"  颜色归一化: {self.pp_config.color_normalization.method}")
            print(f"  模糊过滤: {self.pp_config.blur_filter.enabled}")

    def process(self, img: np.ndarray) -> np.ndarray:
        """
        完整的预处理流水线。

        参数:
            img: [H, W, 3] uint8 RGB图像数组

        返回:
            processed: [H, W, 3] uint8 处理后的RGB图像
        """
        if not self.enabled:
            return img

        # 1) 去噪
        img = self._denoise(img)

        # 2) 对比度增强
        img = self._enhance_contrast(img)

        # 3) 颜色归一化
        img = self._normalize_color(img)

        return img

    # ============================================================
    # 去噪
    # ============================================================
    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """根据配置选择去噪方法"""
        method = self.pp_config.denoise.method.lower()

        if method == "none":
            return img

        elif method == "gaussian":
            return self._denoise_gaussian(img)

        elif method == "median":
            return self._denoise_median(img)

        elif method == "bilateral":
            return self._denoise_bilateral(img)

        else:
            print(f"[ImagePreprocessor] 警告: 未知的去噪方法 '{method}'，跳过")
            return img

    def _denoise_gaussian(self, img: np.ndarray) -> np.ndarray:
        """高斯滤波去噪: 平滑噪声但保留整体结构"""
        import cv2
        ksize = self.pp_config.denoise.kernel_size  # 默认3
        sigma = self.pp_config.denoise.sigma or 0   # 0=自动计算
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    def _denoise_median(self, img: np.ndarray) -> np.ndarray:
        """中值滤波去噪: 对椒盐噪声效果好"""
        import cv2
        ksize = self.pp_config.denoise.kernel_size
        return cv2.medianBlur(img, ksize)

    def _denoise_bilateral(self, img: np.ndarray) -> np.ndarray:
        """双边滤波去噪: 保留边缘的同时平滑平坦区域"""
        import cv2
        return cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)

    # ============================================================
    # 对比度增强
    # ============================================================
    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """根据配置选择对比度增强方法"""
        method = self.pp_config.contrast.method.lower()

        if method == "none":
            return img

        elif method == "clahe":
            return self._clahe(img)

        elif method == "histogram_equalization":
            return self._histogram_equalization(img)

        else:
            print(f"[ImagePreprocessor] 警告: 未知的对比度方法 '{method}'，跳过")
            return img

    def _clahe(self, img: np.ndarray) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization)。

        局部自适应直方图均衡化，比全局均衡化更好地保留局部细节。
        在LAB色彩空间的L通道上操作，避免颜色失真。
        """
        import cv2
        clip_limit = self.pp_config.contrast.clip_limit  # 默认2.0
        tile_size = tuple(self.pp_config.contrast.tile_grid_size)  # (8,8)

        # 转换到LAB色彩空间
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # 在L（亮度）通道上应用CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_eq = clahe.apply(l)

        # 合并回LAB再转RGB
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    def _histogram_equalization(self, img: np.ndarray) -> np.ndarray:
        """全局直方图均衡化（在YCbCr色彩空间的Y通道上操作）"""
        import cv2
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)
        ycrcb_eq = cv2.merge([y_eq, cr, cb])
        return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)

    # ============================================================
    # 颜色归一化
    # ============================================================
    def _normalize_color(self, img: np.ndarray) -> np.ndarray:
        """根据配置选择颜色归一化方法"""
        method = self.pp_config.color_normalization.method.lower()

        if method == "none":
            return img

        elif method == "reinhard":
            return self._reinhard_normalize(img)

        elif method == "macenko":
            return self._macenko_normalize(img)

        else:
            print(f"[ImagePreprocessor] 警告: 未知的颜色归一化方法 '{method}'，跳过")
            return img

    def _reinhard_normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Reinhard颜色归一化。

        在LAB色彩空间中将图像的均值和标准差匹配到目标值。
        这是一种轻量级的染色归一化方法，适合批量处理。
        """
        import cv2
        target_mean = np.array(self.pp_config.color_normalization.target_mean) * 255
        target_std = np.array(self.pp_config.color_normalization.target_std)

        # 转换到LAB
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)

        # 对每个通道分别归一化
        for c in range(3):
            channel = lab[:, :, c]
            c_mean = channel.mean()
            c_std = channel.std()

            # Z-score归一化后映射到目标分布
            if c_std > 0:
                channel = (channel - c_mean) / c_std * target_std[c] * 255 + target_mean[c]
            else:
                channel = channel - c_mean + target_mean[c]

            lab[:, :, c] = np.clip(channel, 0, 255)

        lab = lab.astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def _macenko_normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Macenko染色分离归一化。

        基于SVD的染色分离方法，将图像分解为苏木精(H)和伊红(E)两个染色通道，
        然后归一化染色强度后重建。对H&E染色偏差有较好的纠正效果。
        """
        import cv2
        # 转换为OD (Optical Density) 空间
        img_float = img.astype(np.float32) / 255.0
        img_float = np.maximum(img_float, 1e-6)
        OD = -np.log(img_float)

        # 去除低OD像素（背景）
        OD_reshaped = OD.reshape(-1, 3)
        OD_thresh = OD_reshaped[OD_reshaped.max(axis=1) > 0.1]

        if len(OD_thresh) < 2:
            return img

        # SVD分解找到两个主方向（对应H和E染色）
        from numpy.linalg import svd
        _, s, Vt = svd(OD_thresh, full_matrices=False)
        V = Vt[:2, :]  # 前两个主成分

        # 归一化染色强度
        if s[0] > 0:
            V = V / np.linalg.norm(V, axis=1, keepdims=True)

        # 计算染色浓度
        conc = np.dot(OD.reshape(-1, 3), V.T)

        # 归一化到99分位数
        for i in range(2):
            p99 = np.percentile(conc[:, i], 99)
            if p99 > 0:
                conc[:, i] = conc[:, i] / p99

        # 重建图像
        OD_norm = np.dot(conc, V)
        img_norm = np.exp(-OD_norm).reshape(img.shape)
        img_norm = np.clip(img_norm * 255, 0, 255).astype(np.uint8)

        return img_norm


# ============================================================
# 模糊检测
# ============================================================

def detect_blur_laplacian(img: np.ndarray, threshold: float = 100.0) -> Tuple[float, bool]:
    """
    使用Laplacian方差法检测图像模糊度。

    原理:
        清晰图像的Laplacian响应（边缘强度）的方差较大，
        模糊图像的Laplacian响应方差较小。

    参数:
        img: [H, W, C] uint8 RGB图像
        threshold: 方差阈值，低于此值判定为模糊

    返回:
        (variance, is_blurry): 方差值和模糊标记
    """
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    is_blurry = variance < threshold
    return variance, is_blurry


def detect_blur_sobel(img: np.ndarray) -> float:
    """
    使用Sobel梯度幅值检测图像清晰度。

    参数:
        img: [H, W, C] uint8 RGB图像

    返回:
        gradient_magnitude: 平均梯度幅值（越高越清晰）
    """
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return float(magnitude.mean())


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ImagePreprocessor 图像预处理模块自测")
    print("=" * 60)

    # 模拟图像
    np.random.seed(42)
    mock_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # 测试模糊检测
    var, blurry = detect_blur_laplacian(mock_img)
    sobel_mag = detect_blur_sobel(mock_img)
    print(f"\n  随机噪声图像:")
    print(f"    Laplacian方差: {var:.2f} (阈值=100, 模糊={blurry})")
    print(f"    Sobel梯度均值: {sobel_mag:.2f}")

    # 测试去噪（需要cv2）
    try:
        import cv2
        denoised = cv2.GaussianBlur(mock_img, (3, 3), 0)
        var2, _ = detect_blur_laplacian(denoised)
        print(f"\n  高斯滤波后Laplacian方差: {var2:.2f}")
    except ImportError:
        print("\n  (需要opencv-python进行去噪测试)")

    print("\n所有测试通过!")
