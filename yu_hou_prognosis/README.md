# YuHou - 结直肠癌(COAD)多模态预后预测

> **YuHou (预后)**：基于WSI病理图像与基因组数据的多模态深度学习框架，用于结直肠癌患者的生存预后预测。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![GPU](https://img.shields.io/badge/GPU-RTX%204080%20Super%2016GB-orange.svg)](https://www.nvidia.com/)

---

## 📋 目录

- [项目概览](#项目概览)
- [上下游调用关系](#上下游调用关系)
- [核心创新](#核心创新)
- [项目结构](#项目结构)
- [环境部署](#环境部署)
- [数据集说明](#数据集说明)
- [快速开始](#快速开始)
- [实验方案](#实验方案)
- [模型选型依据](#模型选型依据)
- [评估指标](#评估指标)
- [GPU运行优化](#gpu运行优化)
- [常见问题](#常见问题)

---

## 项目概览

本项目基于TCGA-COAD（结肠腺癌）数据集，利用**全切片病理图像(WSI)切块**与**基因组临床数据**构建多模态深度学习模型，预测结直肠癌患者的**总体生存期(Overall Survival)**。

### 研究背景

结直肠癌是全球第三大常见恶性肿瘤，准确的预后预测对于制定个体化治疗方案至关重要。本项目结合：

1. **病理形态学信息**：从H&E染色的WSI图像中提取肿瘤微环境特征
2. **基因组分子信息**：RNA-seq基因表达、基因突变等分子特征

通过多模态融合技术，实现比单一模态更准确的生存预后预测。

---

## 上下游调用关系

```
┌─────────────────────────────────┐
│  wsi_patch_selection_benchmark  │  ← 上游项目
│  (WSI切块 + Patch筛选算法对比)  │
│                                  │
│  输入: TCGA-COAD .svs WSI文件    │
│  输出: outputs/patches/{算法}/   │
│         TCGA-XX-XXXX_1.png       │
│         TCGA-XX-XXXX_2.png       │
│         ...                      │
└────────────┬────────────────────┘
             │  patch图像文件夹
             ▼
┌─────────────────────────────────┐
│     yu_hou_prognosis (本项目)    │  ← 下游项目
│  (多模态预后生存预测)            │
│                                  │
│  输入1: 上游patch图像             │
│  输入2: COAD_all_dataset.csv     │
│  输出: 生存风险预测分数           │
│         C-index/AUC评估报告       │
│         KM生存曲线                │
└─────────────────────────────────┘
```

**固定执行顺序**:
1. 先运行 `wsi_patch_selection_benchmark` 生成病理切块图像
2. 再运行本项目 `yu_hou_prognosis` 进行预后预测

配置文件 `config/default_config.yaml` 中的 `upstream.patch_root` 指向上游输出目录，支持自定义算法名称。

---

## 核心创新

### 1. VectorCAUGF 三流自适应融合模块

本项目核心融合模块 **VectorCAUGF** 采用三流并行架构：

- **病理特征流 (Path Stream)**：从ResNet50提取的WSI patch特征
- **基因组特征流 (Omic Stream)**：从MLP编码器提取的基因特征
- **关系流 (Relation Stream)**：显式建模两个模态之间的乘积、差异、余弦相似度关系

通过**流级softmax自适应加权** + **sigmoid门控校准** + **残差保底连接**，在表达性与稳定性之间取得平衡。

### 2. 9种多模态融合策略横向对比

| 融合策略 | 类型 | 说明 |
|---------|------|------|
| `caugf` | VectorCAUGF | 三流自适应加权融合 ★核心创新 |
| `pofusion` | BilinearFusion | 门控双线性外积融合 |
| `lmf` | LMF | 低秩多模态融合（参数高效） |
| `concat` | ConcatFusion | 简单拼接+MLP（基线） |
| `gmu` | GMUFusion | 门控多模态单元 |
| `film` | FiLMFusion | 特征线性调制（基因组调节病理） |
| `attention_weighted` | AttentionWeightedFusion | 注意力加权融合 ★新增 |
| `cross_attention` | CrossAttentionFusion | 交叉注意力多模态融合 ★新增 |
| `tensor_concat` | TensorConcatFusion | 多层级特征图拼接融合 ★新增 |

### 3. 低质量WSI图像鲁棒处理

- **自适应去噪**：高斯/中值/双边滤波
- **对比度增强**：CLAHE局部自适应直方图均衡化
- **颜色归一化**：Reinhard/Macenko染色归一化
- **模糊图像检测**：Laplacian方差法，标记低质量patch

---

## 项目结构

```
yu_hou_prognosis/
├── README.md                          # 本文件
├── requirements.txt                   # Python依赖
├── main.py                            # 统一入口(eda/train/test/ablation)
├── config/
│   ├── default_config.yaml            # 全局默认配置
│   ├── config_loader.py               # 配置加载器
│   └── ablation/                      # 消融实验配置
├── data/
│   └── TCGA_COAD/
│       └── COAD_all_dataset.csv       # 基因组临床数据
├── weights/                           # 预训练权重
│   ├── resnet50-0676ba61.pth          # ImageNet预训练ResNet50
│   └── lmf_test_model.pth             # LMF测试模型
├── src/
│   ├── data_loading/                  # 数据加载层
│   │   ├── datasets.py                # PyTorch Dataset定义
│   │   ├── upstream_reader.py         # 上游WSI patch读取
│   │   ├── preprocessing.py           # 图像去噪/增强/归一化
│   │   └── split_builder.py           # 5折CV数据集拆分
│   ├── eda/                           # 数据探索性分析
│   │   ├── genomic_eda.py             # 基因组特征EDA
│   │   ├── image_quality_eda.py       # 图像质量EDA
│   │   └── task_justification.py      # 任务合理性与基线论证
│   ├── feature_engineering/           # 特征工程
│   │   ├── feature_selection.py       # 基因组特征筛选/降维
│   │   └── image_quality_filter.py    # 模糊图像过滤
│   ├── networks/                      # 模型结构层
│   │   ├── resnet_backbone.py         # ResNet骨干(CBAM注意力)
│   │   ├── path_net.py                # 病理图像编码器
│   │   ├── omic_net.py                # 基因组特征编码器
│   │   ├── pathomic_net.py            # 多模态主网络
│   │   ├── caugf.py                   # VectorCAUGF融合(核心)
│   │   ├── fusion_bilinear.py         # 门控双线性融合
│   │   ├── fusion_lmf.py              # 低秩多模态融合
│   │   ├── fusion_concat.py           # 拼接融合基线
│   │   ├── fusion_gmu.py              # 门控多模态单元
│   │   ├── fusion_film.py             # 特征线性调制
│   │   ├── fusion_attention_weighted.py  # 注意力加权融合
│   │   ├── fusion_cross_attention.py  # 交叉注意力融合
│   │   ├── fusion_tensor_concat.py    # 多层级特征图拼接
│   │   └── fusion_factory.py          # 融合策略工厂
│   ├── losses/                        # 损失函数层
│   │   ├── cox_loss.py                # Cox比例风险损失
│   │   ├── classification_loss.py     # 分类损失(CE/BCE/Focal)
│   │   └── regularization.py          # 正则化
│   ├── training/                      # 训练验证推理层
│   │   ├── trainer.py                 # 训练器(AMP/梯度累积)
│   │   ├── tester.py                  # 测试/推理
│   │   ├── cv_runner.py               # 交叉验证主控
│   │   ├── checkpoint.py              # 断点续训管理
│   │   ├── gpu_manager.py             # GPU显存监控/OOM保护
│   │   ├── optimizer.py               # 优化器工厂
│   │   └── scheduler.py               # 学习率调度器工厂
│   ├── evaluation/                    # 指标评估层
│   │   ├── survival_metrics.py        # C-index/tdAUC/logrank/HR
│   │   ├── classification_metrics.py  # 分类指标
│   │   ├── calibration.py             # 校准曲线+Brier Score
│   │   └── metric_formatter.py        # 中文指标格式化
│   ├── visualization/                 # 可视化层
│   │   ├── km_curves.py               # KM生存曲线
│   │   ├── risk_histograms.py         # 风险直方图
│   │   ├── roc_curves.py              # 时间依赖ROC曲线
│   │   ├── calibration_plots.py       # 校准曲线图
│   │   └── boxplots.py                # 箱线图
│   └── utils/                         # 工具层
│       ├── logger.py                  # 中文日志
│       ├── seed.py                    # 随机种子
│       └── io_utils.py                # IO工具
└── experiments/                       # 实验输出(运行时创建)
    └── {exp_name}/
        ├── ckpt/                      # 模型检查点
        ├── logs/                      # 训练日志
        ├── preds/                     # 预测结果
        ├── results/                   # 评估结果
        └── figures/                   # 可视化图表
```

---

## 环境部署

### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | NVIDIA GPU 8GB+ | RTX 4080 Super 16GB |
| CPU | 8核 | 16核+ |
| RAM | 32GB | 64GB+ |
| 存储 | 50GB | 200GB+ SSD |

### 软件环境

```bash
# 1. 创建虚拟环境 (推荐Python 3.10)
conda create -n yuhou python=3.10
conda activate yuhou

# 2. 安装PyTorch (CUDA 11.8版本)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 安装项目依赖
cd yu_hou_prognosis
pip install -r requirements.txt

# 4. 验证GPU可用性
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU型号: {torch.cuda.get_device_name(0)}')"
```

### 4080 Super环境特别说明

- CUDA版本建议 11.8 或 12.1+
- 推荐PyTorch 2.0+ (支持`torch.compile()`加速)
- 混合精度使用float16 (4080 Super不支持bfloat16硬件加速)
- 驱动版本 >= 535.xx

---

## 数据集说明

### COAD_all_dataset.csv

TCGA-COAD（结肠腺癌）患者数据集，包含约**350列**：

| 列类型 | 说明 | 示例 |
|--------|------|------|
| 患者ID | TCGA样本编号 | TCGA-A6-2686 |
| 生存时间 | OS.time (天) | 1123 |
| 删失标记 | OS (1=死亡, 0=存活/失访) | 1 |
| 临床特征 | 年龄、性别、TNM分期、Grade等 | T3, N0, M0 |
| 基因突变 | BRCA1, TP53, KRAS等~300个基因 | 0/1 (野生型/突变型) |

### WSI Patch图像

来自上游 `wsi_patch_selection_benchmark` 项目输出：
- 格式：PNG (RGB)
- 尺寸：1024×1024像素
- 每病人：6个代表性patch
- 来源：TCGA-COAD DX1诊断切片

---

## 快速开始

### 1. 配置

编辑 `config/default_config.yaml` 或创建自定义配置：

```bash
# 使用默认配置（自动检测路径）
python main.py --mode train

# 使用自定义配置
python main.py --mode train --config config/default_config.yaml

# 命令行覆盖参数
python main.py --mode train --fusion_type cross_attention --batch_size 2 --exp_name exp_cross_attn
```

### 2. 数据探索性分析 (EDA)

```bash
# 运行完整EDA（基因组 + 图像质量 + 任务论证）
python main.py --mode eda
```

### 3. 训练

```bash
# 默认CAUGF融合策略训练
python main.py --mode train

# 交叉注意力融合训练
python main.py --mode train --fusion_type cross_attention --exp_name ca_experiment

# 注意力加权融合训练
python main.py --mode train --fusion_type attention_weighted --exp_name aw_experiment
```

### 4. 消融实验

```bash
# 融合策略对比实验（所有9种策略）
python main.py --mode ablation --config config/ablation/fusion_comparison.yaml

# 损失函数对比实验
python main.py --mode ablation --config config/ablation/loss_comparison.yaml

# 图像预处理消融实验
python main.py --mode ablation --config config/ablation/preprocessing.yaml
```

### 5. 推理

```bash
# 单模型推理
python main.py --mode test --checkpoint experiments/default/ckpt/best.pt
```

---

## 实验方案

### 主要实验

| 实验编号 | 实验名称 | 目的 | 对比维度 |
|---------|---------|------|---------|
| Exp-1 | 融合策略对比 | 比较9种融合策略的预后预测性能 | C-index, tdAUC |
| Exp-2 | 损失函数对比 | CoxLoss vs CoxPHLoss vs 组合损失 | 收敛速度, 预测精度 |
| Exp-3 | 图像预处理消融 | 评估去噪/增强/归一化的影响 | 有/无预处理 |
| Exp-4 | 特征选择消融 | 评估降维对过拟合的缓解效果 | 原始356维 vs 100维 |
| Exp-5 | 单模态基线 | Path-only vs Omic-only | 多模态增益 |
| Exp-6 | CAUGF优化验证 | 验证7项CAUGF优化的改进效果 | 每项改动独立对比 |

---

## 模型选型依据

### 为什么不用传统机器学习模型（LASSO/RF/XGBoost）？

1. **多模态非线性融合**：病理图像与基因组数据具有本质不同的数据结构，深度学习可以端到端学习跨模态非线性交互
2. **高维特征自动提取**：WSI patch的1024×1024像素需要CNN提取层次化视觉特征
3. **生存分析专项优化**：Cox比例风险损失函数天然适合生存数据，传统ML模型难以直接优化生存排序
4. **端到端训练**：深度学习可以同时优化特征提取和预测，避免分阶段误差累积

### 为什么是生存分析而非普通分类/回归？

1. **删失数据处理**：临床随访中大量患者未发生终点事件（删失），普通分类/回归无法处理
2. **时间维度信息**：生存分析可以输出随时间变化的风险函数，临床意义更丰富
3. **临床标准指标**：C-index、KM曲线是肿瘤预后领域的金标准
4. **个体化预测**：生存分析可输出任意时间点的存活概率，而非简单的二分类判断

---

## 评估指标

### 主要指标

| 指标 | 说明 | 范围 | 方向 |
|------|------|------|------|
| **C-index** | 一致性指数，衡量风险预测的排序能力 | [0, 1] | ↑越高越好 |
| **tdAUC** | 时间依赖AUC，特定时间点的判别能力 | [0, 1] | ↑越高越好 |
| **Log-rank p** | 高/低风险组生存差异显著性 | [0, 1] | ↓越低越好 |
| **Hazard Ratio** | 风险比，高/低风险组的风险比例 | (0, ∞) | 远离1越好 |
| **Brier Score** | 预测概率校准度 | [0, 1] | ↓越低越好 |

### 可视化评估

- **KM生存曲线**：高中低风险组的Kaplan-Meier生存曲线 + log-rank检验
- **时间依赖ROC曲线**：不同时间点的ROC曲线
- **校准曲线**：预测风险 vs 实际事件发生率
- **风险直方图**：预测风险分数的分布

---

## GPU运行优化

### RTX 4080 Super (16GB) 优化策略

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `batch_size` | 2-4 | 1024×1024×6 patches每样本约1.5GB |
| `gradient_accumulation_steps` | 2-4 | 等效增大batch_size到8-16 |
| `amp.dtype` | float16 | 4080 Super推荐float16 |
| `num_workers` | 4 | DataLoader进程数 |
| `pin_memory` | true | 锁页内存加速CPU→GPU传输 |
| `gradient_clip_norm` | 1.0 | 防止梯度爆炸 |

### 显存监控

项目内置GPU显存监控模块，自动：
- 记录每10个batch的显存使用
- 在85%显存使用时告警
- 捕获CUDA OOM异常并尝试自动降batch重试

---

## 常见问题

### Q: 上游patch图像不存在怎么办？
A: 请确保已运行 `wsi_patch_selection_benchmark` 项目，并检查 `config/default_config.yaml` 中 `upstream.patch_root` 路径正确。

### Q: CUDA内存不足怎么办？
A: 尝试以下方案：(1) 减小 `batch_size`；(2) 增大 `gradient_accumulation_steps`；(3) 减少 `num_patches_per_patient`。

### Q: 如何使用自己的数据集？
A: (1) 准备类似格式的CSV文件；(2) 修改 `data.genomic_csv` 路径；(3) 重新生成5折CV拆分文件。

### Q: 断点续训如何工作？
A: 训练中断后，使用相同配置重新运行即可自动从最近的断点恢复。断点保存在 `experiments/{exp_name}/resume/` 目录下。

---

## 引用

本项目参考了以下开源工作：
- **Pathomic Fusion**: Chen et al., "Pathomic Fusion: An Integrated Framework for Fusing Histopathology and Genomic Features for Cancer Diagnosis and Prognosis", IEEE TMI, 2020
- **LMF**: Liu et al., "Efficient Low-rank Multimodal Fusion with Modality-Specific Factors", ACL 2018
- **GMU**: Arevalo et al., "Gated Multimodal Units for Information Fusion", ICLR 2017
- **FiLM**: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018
