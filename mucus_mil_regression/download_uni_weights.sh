#!/usr/bin/env bash
# ============================================================
# 下载 UNI 病理自监督预训练权重 (ViT-L/16)
# 权重来源: MahmoodLab/UNI (HuggingFace, gated 需先申请授权)
#   - 模型页: https://huggingface.co/MahmoodLab/UNI
#   - 需先在该页面点击 "Access repository" 申请访问, 并准备 HF Token:
#     https://huggingface.co/settings/tokens
# 保存位置: weights/uni/pytorch_model.bin (下载一次, 后续训练复用)
#
# 用法:
#   export HF_TOKEN=hf_xxxxxx
#   bash download_uni_weights.sh
# ============================================================

set -e

WEIGHT_DIR="weights/uni"
WEIGHT_PATH="${WEIGHT_DIR}/pytorch_model.bin"
# 注意: 仓库名区分大小写 (MahmoodLab/UNI)
HF_URL="https://huggingface.co/MahmoodLab/UNI/resolve/main/pytorch_model.bin"

mkdir -p "${WEIGHT_DIR}"

if [ -f "${WEIGHT_PATH}" ]; then
    echo "[UNI] 权重已存在, 跳过下载: ${WEIGHT_PATH}"
    exit 0
fi

if [ -z "${HF_TOKEN}" ]; then
    echo "[UNI] 错误: 需要 HF_TOKEN (该仓库为 gated, 需先申请授权并生成token)"
    echo "  1) 访问 https://huggingface.co/MahmoodLab/UNI 点击 Access repository 申请"
    echo "  2) 在 https://huggingface.co/settings/tokens 生成 Read token"
    echo "  3) export HF_TOKEN=hf_xxxxxx 后重新运行本脚本"
    exit 1
fi

echo "[UNI] 下载 UNI 权重 (约1.1GB)..."
if command -v wget >/dev/null 2>&1; then
    wget --header="Authorization: Bearer ${HF_TOKEN}" -O "${WEIGHT_PATH}" "${HF_URL}"
elif command -v curl >/dev/null 2>&1; then
    curl -L -H "Authorization: Bearer ${HF_TOKEN}" -o "${WEIGHT_PATH}" "${HF_URL}"
else
    echo "错误: 需要 wget 或 curl"
    exit 1
fi

echo "[UNI] 下载完成: ${WEIGHT_PATH}"
echo "[UNI] 使用方式: config/default_config.yaml 中设置"
echo "       model.backbone: uni"
echo "       model.backbone_pretrained: weights/uni/pytorch_model.bin"
