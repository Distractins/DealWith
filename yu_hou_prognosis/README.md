# YuHou → Histomolecular 基因表达预测

> **下游任务 (重构后)**：用上游输出的 WSI patch 图像，预测肠分化/粘液相关 RNA-seq 基因表达（如 REG4、OLFM4、MUC2、DEFA5）。

---

## 任务说明

原「WSI+基因组多模态 Cox 生存预后预测」已替换为 **Histomolecular 任务**：

- **主任务**：输入 WSI patch 图像 → 预测目标基因表达。
  - 默认 **二分类**：用表达量中位数分组，高表达 / 低表达（`CrossEntropyLoss`）。
  - 可选 **回归**：预测连续表达量（`MSE`）。
- **标签**：表格文件（CSV）包含 `TCGA ID`、临床信息、多个目标基因 `*_rnaseq` 表达值；程序自动匹配 TCGA ID、自动做中位数划分。
- **多模态用途改变**：图像 patch 特征用于预测基因标签；基因组/临床信息作为**可选协变量**参与融合（`mode=pathomic`）。
- **附属分析（非主目标）**：用模型预测出的分子表型分组做生存 KM 曲线 + logrank 检验，仅作补充验证，**不再**以 C-index 作为主评价指标。

上游 `wsi_patch_selection_benchmark` **完全不动**：所有切块逻辑、9 种 patch 采样算法、输出路径保持不变，其输出 patch 仍作为下游输入。

---

## 项目结构

```
yu_hou_prognosis/
├── README.md
├── main.py                          # 统一入口 (eda/train/test/ablation)
├── make_split.py                    # 按病人ID生成5折拆分 (训练前运行一次)
├── build_data_splits.py             # (可选) 委托 cv_runner 构建完整数据
├── config/
│   ├── default_config.yaml          # 全局配置
│   ├── config_loader.py             # 配置加载器
│   └── ablation/                    # 消融实验配置
│       ├── patch_sampling_comparison.yaml   # 上游9种采样算法对比
│       ├── fusion_comparison.yaml
│       ├── loss_comparison.yaml
│       ├── preprocessing.yaml
│       └── feature_selection.yaml
├── data/
│   └── COAD_all_dataset.csv         # TCGA ID + 临床 + 目标基因rnaseq表达
├── weights/                         # 预训练权重 (resnet50)
└── src/
    ├── data_loading/
    │   ├── datasets.py              # 病人级Dataset (patch袋 + 基因标签)
    │   ├── gene_labels.py           # ★ 基因表达列解析 + 中位数分组
    │   ├── upstream_reader.py       # 上游patch读取 (不变)
    │   ├── split_builder.py         # 按病人ID分层拆分 (防泄露)
    │   └── preprocessing.py         # 图像预处理 (保留)
    ├── eda/
    │   └── gene_eda.py              # ★ 基因表达分布 + 高低表达组临床差异 + KM
    ├── networks/                    # 网络结构 (保留ResNet/PathNet/OmicNet/9种融合)
    │   └── pathomic_net.py          # ★ 主网络 (输出基因标签)
    ├── losses/
    │   ├── classification_loss.py   # CE/BCE/Focal + MSE回归
    │   ├── cox_loss.py              # (保留, 未用于主任务)
    │   └── regularization.py
    ├── training/
    │   ├── trainer.py               # ★ 训练/评估 (ACC/AUC/F1)
    │   ├── cv_runner.py             # ★ 交叉验证主控 + 结果CSV
    │   ├── checkpoint.py / gpu_manager.py / optimizer.py / scheduler.py
    │   └── tester.py                # (保留, 已被 trainer.test 取代)
    ├── evaluation/
    │   ├── classification_metrics.py
    │   ├── regression_metrics.py    # ★ MSE/MAE/RMSE/Pearson/Spearman
    │   ├── survival_metrics.py      # (保留, 附属KM用)
    │   ├── calibration.py / metric_formatter.py
    ├── visualization/
    │   ├── roc_curves.py            # ROC
    │   ├── confusion_matrix.py      # ★ 混淆矩阵
    │   ├── km_curves.py             # KM (含按分子表型分组)
    │   └── ...
    └── utils/                       # logger/seed/io
└── experiments/                     # 运行时生成 (每基因独立目录)
```

---

## 环境部署

与旧版一致（PyTorch 2.0+、CUDA 11.8/12.1、RTX 4080 Super 16GB 推荐）：

```bash
conda create -n dealwith python=3.10
conda activate dealwith
cd yu_hou_prognosis
pip install -r requirements.txt
```

---

## 数据准备

1. 上游切块（不动）：先运行 `wsi_patch_selection_benchmark` 生成 `outputs/patches/{algorithm}/*.png`。
2. 表格文件：`data/COAD_all_dataset.csv`，需包含：
   - `TCGA ID` 列（病人 ID）
   - 临床/生存列（可选，用于附属 KM，如 `event`、`Survival months`）
   - 目标基因表达列（如 `REG4_rnaseq`、`OLFM4_rnaseq`、`MUC2_rnaseq`、`DEFA5_rnaseq`）
3. 生成病人级拆分（同一病人 6 张切片不跨折）：

```bash
python make_split.py
```

---

## 运行示例命令

### 1. EDA（基因表达分布 + 高低表达组临床差异 + 附属 KM）

```bash
python main.py --mode eda --target_gene REG4
```

### 2. 训练（分类，默认高/低表达二分类）

```bash
# 单基因
python main.py --mode train --target_gene REG4

# 指定上游patch采样算法 (grid/random/kmeans/yottixel/stratified/sentinel/splice/sdm/largest_tissue)
python main.py --mode train --target_gene REG4 --algorithm_name kmeans

# 指定融合策略 (path-only时融合被旁路, 需配置协变量并 mode=pathomic)
python main.py --mode train --target_gene REG4 --fusion_type caugf

# 批量多基因 (每个基因独立实验目录 + 各自CSV)
python main.py --mode train --target_genes REG4 OLFM4 MUC2 DEFA5
```

### 3. 训练（回归，可选）

```bash
python main.py --mode train --target_gene REG4 --task_type regression
```

### 4. 测试（加载 checkpoint 评估）

```bash
python main.py --mode test --target_gene REG4 \
  --checkpoint experiments/default/ckpt/fold1_best_model.pth
```

### 5. 消融（上游 9 种 patch 采样算法对比）

```bash
python main.py --mode ablation --target_gene REG4
# 或
python main.py --mode ablation --config config/ablation/patch_sampling_comparison.yaml
```

---

## 关键配置项（`config/default_config.yaml`）

| 配置 | 说明 |
|------|------|
| `model.task_type` | `classification`（二分类）/ `regression`（回归） |
| `model.target_gene` | 指定目标基因，如 `REG4` |
| `data.genomic.target_genes` | 目标基因列表（批量训练） |
| `model.group_threshold` | 分组阈值：`median`（默认）/`mean`/`quantile`/数值 |
| `model.mode` | `path`（纯图像，默认）/ `pathomic`（图像+协变量融合）/ `omic` |
| `data.genomic.covariate_cols` | 可选协变量列（临床/其他分子特征；勿放目标基因自身） |
| `upstream.algorithm_name` | 上游采样算法（消融实验切换此项） |
| `training.early_stopping.monitor` | 早停监控指标：`auc`/`accuracy`/`f1`/`loss`（回归：`pearson`） |

---

## 实验设计要点

- **Patient-level 聚合**：每个患者 6 张 WSI 切片，训练时 PathNet 逐 patch 提取特征后**均值池化**为病人级表示；训练与评估均为 patient-level，不按 patch 评估。
- **防数据泄露**：5 折拆分以 `TCGA ID` 为最小单元，同一病人的所有 patch 始终在同一折；二分类中位数阈值**基于训练集**计算后应用到 train/test。
- **标签构建**：`data_loading/gene_labels.py` 自动解析 `REG4` → `REG4_rnaseq`，中位数划分高/低表达。

## 评估指标

- **主指标（分类）**：ACC、AUC、F1（同时输出 balanced ACC、precision、recall、AP、MCC）。
- **主指标（回归）**：MSE、MAE、RMSE、Pearson、Spearman。
- **附属（非主目标）**：预测分子表型分组的 KM 曲线 + logrank p 值，仅作补充验证。

## 输出

每个基因在 `experiments/{exp_name}/` 下输出：

- `results/{gene}_results.csv` — 每折 + 均值±标准差实验日志
- `results/cv_results.pkl` — 完整 CV 结果
- `ckpt/fold{id}_best_model.pth` — 模型权重
- `figures/` — ROC、混淆矩阵、Loss 曲线、附属 KM 图
- `logs/cv_train.log` — 训练日志
