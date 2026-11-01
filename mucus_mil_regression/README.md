# MucusMIL Regression - 弱监督多实例学习回归

> **下游任务**：输入 WSI 切分 patch 集合，回归 PCA 得到的一维「粘液分化复合分子得分」(Histomolecular 表征学习)。

---

## 任务定义

- **包级弱监督回归**：一个病人 = 一张 WSI = 一个 bag，patch = 实例；只有 bag 级回归标签，patch 无局部标注。
- **标签来源**：4 个目标基因表达 (REG1A / REG4 / SPINK4 / ITLN1) 做 PCA，取第一主成分 PC1 作为粘液分化得分 y（EDA 阶段一次性生成）。
- **研究重心**：patch 采样策略、特征聚合（ABMIL 注意力）、多模态融合策略的对比；偏计算机视觉方向，生物学内容仅作为标签来源，弱化生物机制讨论。

---

## 三组对照实验（创新点：多模态融合）

| 实验 | 模型 | 输入 | 目的 |
|------|------|------|------|
| (1) 临床基线 | `clinical_mlp` | 仅临床表格特征 | 评估图像带来的增量价值 |
| (2) 纯图像 | `image_mil` | patch (ResNet/UNI + ABMIL) | 图像形态表征能力 |
| (3) 多模态 | `multimodal` | 图像全局表征 + 临床特征 | 融合策略消融 |

多模态融合两种方式：
- `late_concat`：晚拼接融合（基线）
- `adaptive_gate`：自适应门控融合（本文提出的改进策略）

---

## 项目结构

```
mucus_mil_regression/
├── README.md
├── main.py                          # 统一入口 (eda/train/test/ablation/visualize)
├── download_uni_weights.sh          # UNI权重下载
├── config/
│   ├── default_config.yaml
│   ├── config_loader.py
│   └── ablation/
├── src/
│   ├── data_loading/                # patch bag / 临床编码 / 拆分
│   ├── eda/                         # PCA标签构建
│   ├── feature_engineering/         # 临床特征编码
│   ├── networks/                    # ResNet/UNI + ABMIL + 融合
│   ├── losses/                      # MSE
│   ├── training/                    # 训练/CV/断点/GPU
│   ├── evaluation/                  # Spearman/Pearson/MAE/RMSE/R²
│   ├── visualization/               # 注意力热力图/散点图/消融图
│   └── utils/
├── weights/                         # (运行时下载)
└── experiments/                     # (运行时生成)
```

---

## 运行流程

### 0. 前置：上游切块（不动）+ 生成标签

```bash
# 上游已跑完 patches/{algorithm}/*.png，无需重跑

# 1) 生成 PCA PC1 标签 (一次性, EDA阶段)
python main.py --mode eda
# 输出 data/mucus_pc1_labels.csv + .loadings.csv
```

### 1. 训练

```bash
# 纯图像 ABMIL (ResNet)
python main.py --mode train --model_type image_mil --backbone resnet

# 纯图像 ABMIL (UNI, 需先下载权重)
bash download_uni_weights.sh
python main.py --mode train --model_type image_mil --backbone uni

# 临床基线 MLP
python main.py --mode train --model_type clinical_mlp

# 多模态融合 (自适应门控, 本文改进)
python main.py --mode train --model_type multimodal --fusion_type adaptive_gate

# 切换上游采样算法
python main.py --mode train --model_type image_mil --algorithm_name kmeans
```

### 2. 测试

```bash
python main.py --mode test --model_type image_mil \
  --checkpoint experiments/default/ckpt/fold1_best.pth
```

### 3. 消融

```bash
# 三组对照 (clinical_mlp / image_mil / multimodal×2融合)
python main.py --mode ablation --axis models

# 上游9种采样算法对比
python main.py --mode ablation --axis algorithms

# 融合方式对比
python main.py --mode ablation --axis fusions
```

### 4. 可视化

```bash
python main.py --mode visualize
```

---

## 评价指标（无生存指标）

Spearman 相关系数、Pearson 相关系数、MAE、RMSE、R²。

> 注：后续可利用模型预测得到的粘液分化得分做生存相关性附属分析，但不作为模型训练目标，本模块不包含任何生存分析训练代码。

---

## 环境

复用项目根目录 `environment.yml`（`conda activate dealwith`），不新增额外依赖；适配 RTX 4080 Super 16G、64G 内存。
