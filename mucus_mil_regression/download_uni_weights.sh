#!/usr/bin/env bash
# ============================================================
# 下载 UNI 病理自监督预训练权重 (ViT-L/16)
# 权重来源: MahmoodLab/uni (HuggingFace)
# 保存位置: weights/uni/pytorch_model.bin
# ============================================================

set -e

WEIGHT_DIR="weights/uni"
WEIGHT_PATH="${WEIGHT_DIR}/pytorch_model.bin"
HF_URL="https://huggingface.co/MahmoodLab/uni/resolve/main/pytorch_model.bin"

mkdir -p "${WEIGHT_DIR}"

if [ -f "${WEIGHT_PATH}" ]; then
    echo "[UNI] 权重已存在: ${WEIGHT_PATH}"
    exit 0
fi

echo "[UNI] 下载 UNI 权重 (约1GB)..."
if command -v wget >/dev/null 2>&1; then
    wget -O "${WEIGHT_PATH}" "${HF_URL}"
elif command -v curl >/dev/null 2>&1; then
    curl -L -o "${WEIGHT_PATH}" "${HF_URL}"
else
    echo "错误: 需要 wget 或 curl"
    exit 1
fi

echo "[UNI] 下载完成: ${WEIGHT_PATH}"
echo "[UNI] 使用方式: 在 config/default_config.yaml 中设置"
echo "       model.backbone: uni"
echo "       model.backbone_pretrained: weights/uni/pytorch_model.bin"
