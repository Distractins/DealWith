# DealWith - 结直肠癌WSI病理分析全流程

> **上游**: WSI全切片图像切块 + Patch筛选算法对比  
> **下游**: 病理图像+基因组多模态预后生存预测

---

## 项目结构

```
DealWith/
├── README.md
├── .gitignore
├── environment.yml                     # 统一Conda环境(上下游共用)
├── requirements.txt                    # 统一pip依赖
│
├── wsi_patch_selection_benchmark/      # 上游: WSI切块 + 9种采样算法
│   ├── run_patch_selection.py          #   主入口
│   ├── configs/default_config.yaml     #   配置文件
│   ├── core/                           #   核心: mask/候选池/特征/评分
│   ├── samplers/                       #   9种patch采样算法
│   ├── metrics/                        #   4种评估指标
│   ├── evaluation/                     #   批量评估+CSV输出
│   ├── visualization/                  #   图表+论文图
│   └── outputs/                        #   (运行时生成, 29GB+)
│       ├── patches/{algorithm}/        #   patch图像
│       ├── csv/                        #   评估CSV
│       ├── figures/                    #   图表
│       └── reports/                    #   报告
│
└── yu_hou_prognosis/                   # 下游: 多模态预后预测
    ├── main.py                         #   统一入口 (eda/train/test/ablation)
    ├── config/default_config.yaml      #   配置文件
    ├── download_weights.sh             #   权重下载脚本
    ├── src/
    │   ├── data_loading/               #   数据加载+预处理
    │   ├── eda/                        #   数据探索性分析
    │   ├── feature_engineering/        #   特征选择+质量过滤
    │   ├── networks/                   #   9种融合策略+CAUGF
    │   ├── losses/                     #   Cox损失+分类损失
    │   ├── training/                   #   训练/验证/断点/GPU管理
    │   ├── evaluation/                 #   生存分析+分类指标
    │   └── visualization/              #   KM曲线/ROC/校准图
    ├── weights/                        #   (运行时下载, 700MB+)
    └── experiments/                    #   (运行时生成)
```

## 执行流程

```
[1] wsi_patch_selection_benchmark
    输入: TCGA-COAD .svs WSI文件
    运行: python run_patch_selection.py
    输出: outputs/patches/{algorithm}/*.png

        ↓ patch图像

[2] yu_hou_prognosis
    输入1: 上游patch图像
    输入2: COAD_all_dataset.csv (基因组数据)
    运行: python main.py --mode train --fusion_type caugf
    输出: 生存风险预测 + C-index/tdAUC/KM曲线
```

## 服务器部署

```bash
# 1. 克隆代码
git clone https://github.com/你的用户名/DealWith.git
cd DealWith

# 2. 安装环境
conda env create -f environment.yml
conda activate dealwith

# 3. 下载权重 (仅下游需要)
cd yu_hou_prognosis && bash download_weights.sh && cd ..

# 4. 准备数据文件 (scp上传)
#    - WSI .svs文件 → 配置 wsi_root
#    - COAD_all_dataset.csv → yu_hou_prognosis/data/TCGA_COAD/

# 5. 先跑上游切块
cd wsi_patch_selection_benchmark
# 修改 configs/default_config.yaml 中的 wsi_root
python run_patch_selection.py

# 6. 再跑下游预测
cd ../yu_hou_prognosis
# 修改 config/default_config.yaml 中的 upstream.patch_root
python main.py --mode train --fusion_type caugf
```

## 硬件要求

- GPU: NVIDIA RTX 4080 Super 16GB (推荐)
- RAM: 64GB+
- 存储: 200GB+ SSD (WSI文件和patch图像占用较大)
