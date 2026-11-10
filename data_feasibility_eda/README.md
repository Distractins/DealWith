# data_feasibility_eda

前置可行性论证流水线: 在投入深度学习训练前, 从数据源头评估 COAD / READ 两个 TCGA 队列在 WSI-MIL 弱监督任务上的学习信号。

与 `wsi_patch_selection_benchmark` / `mucus_mil_regression` **完全解耦**, 只读它们的 CSV, 不改动任何已有代码, 不含 MIL/Backbone/训练代码。

## 运行

```bash
# 环境需含: pandas scipy scikit-learn matplotlib seaborn lifelines PyYAML
conda activate <你的环境>
python scripts/run_eda.py
```

## 产物

- `feasibility_report.md` — 最终论证报告
- `output/coad/{figures,tables}/` — COAD 图表与表格
- `output/read/{figures,tables}/` — READ 图表与表格

## 分析清单

1. 样本基础统计 (规模 / 粘液占比 / TNM-Stage / 生存事件 / 删失率)
2. 粘液基因集 EDA (分布直方图 / 箱线图 / 相关热力图 / Mann-Whitney 秩和 / 队列内独立 PCA)
3. 标签可行性评估 (A: PC1 回归 / B: 粘液二分类 / C: 生存 Cox)
4. 生存附属分析 (KM 曲线 + log-rank, 仅作事后验证)
