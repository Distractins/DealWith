# -*- coding: utf-8 -*-
"""
main.py
============================================================================
DealWith Histomolecular - 统一入口脚本。

任务: 用WSI patch图像预测肠分化/粘液相关RNA-seq基因表达 (高/低表达二分类,
      可选连续表达量回归)。附属: 预测分子表型分组的KM生存分析 (仅补充验证)。

运行模式:
    eda       - 目标基因表达EDA (分布 + 高低表达组临床差异 + KM)
    train     - 模型训练 (单基因或批量多基因)
    test      - 加载checkpoint在测试集评估
    ablation  - 消融实验 (对比不同上游patch采样算法)

使用示例:
    # EDA (单个基因)
    python main.py --mode eda --target_gene REG4

    # 训练单基因 (分类)
    python main.py --mode train --target_gene REG4 --task_type classification

    # 训练单基因 (回归)
    python main.py --mode train --target_gene REG4 --task_type regression

    # 批量多基因训练
    python main.py --mode train --target_genes REG4 OLFM4 MUC2 DEFA5

    # 指定上游patch采样算法
    python main.py --mode train --target_gene REG4 --algorithm_name kmeans

    # 消融: 对比9种patch采样算法
    python main.py --mode ablation --target_gene REG4

    # 测试 (加载checkpoint)
    python main.py --mode test --target_gene REG4 --checkpoint experiments/default/ckpt/fold1_best_model.pth
============================================================================
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="DealWith Histomolecular - WSI预测肠分化/粘液基因表达",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --mode eda --target_gene REG4
  python main.py --mode train --target_gene REG4
  python main.py --mode train --target_genes REG4 OLFM4 MUC2 DEFA5
  python main.py --mode ablation --target_gene REG4
        """,
    )

    parser.add_argument("--config", "-c", type=str, default="config/default_config.yaml")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["eda", "train", "test", "ablation"])
    parser.add_argument("--target_gene", type=str, default=None, help="目标基因 (覆盖配置)")
    parser.add_argument("--target_genes", type=str, nargs="+", default=None,
                        help="批量目标基因列表")
    parser.add_argument("--task_type", type=str, default=None,
                        choices=["classification", "regression"])
    parser.add_argument("--fusion_type", type=str, default=None)
    parser.add_argument("--algorithm_name", type=str, default=None,
                        help="上游patch采样算法 (覆盖配置)")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="test模式: 模型权重路径")

    args = parser.parse_args()

    from config.config_loader import load_config

    config = load_config(args.config)

    # 命令行覆盖
    if args.fusion_type:
        config.model.fusion.type = args.fusion_type
    if args.exp_name:
        config.experiment.name = args.exp_name
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.optimizer.lr = args.lr
    if args.n_epochs is not None:
        config.training.scheduler.n_epochs = args.n_epochs
    if args.gpu_id is not None:
        config.training.gpu_id = args.gpu_id
    if args.seed is not None:
        config.model.seed = args.seed
    if args.target_gene is not None:
        config.model.target_gene = args.target_gene
    if args.task_type is not None:
        config.model.task_type = args.task_type
    if args.algorithm_name is not None:
        config.upstream.algorithm_name = args.algorithm_name

    print("=" * 70)
    print("  DealWith Histomolecular - WSI病理预测基因表达")
    print("=" * 70)

    if args.mode == "eda":
        _run_eda(config, args.target_genes or [config.model.target_gene])
        return

    if args.mode == "test":
        _run_test(config, args.checkpoint)
        return

    if args.mode == "ablation":
        _run_ablation(config)
        return

    # train
    if args.target_genes:
        _run_train_multi_gene(config, args.target_genes)
    else:
        _run_train(config)


def _setup(config):
    from config.config_loader import create_experiment_dirs
    from src.utils.logger import setup_logger
    from src.utils.seed import set_seed
    create_experiment_dirs(config)
    log_file = str(config.get_subdir("logs") / "main.log")
    logger = setup_logger(log_file=log_file, level=config.logging.level,
                          language=config.logging.language)
    set_seed(config.model.seed, logger=logger)
    return logger


def _run_eda(config, genes):
    """目标基因表达EDA。"""
    logger = _setup(config)
    from src.eda.gene_eda import GeneExpressionEDA

    csv_path = config.resolve_path(config.data.genomic_csv)
    if not csv_path.exists():
        logger.error(f"表格CSV不存在: {csv_path}")
        return

    eda = GeneExpressionEDA(
        str(csv_path),
        output_dir=str(config.get_subdir("figures")),
        patient_id_col=config.data.genomic.patient_id_col,
        survtime_col=config.data.genomic.survtime_col,
        censor_col=config.data.genomic.censor_col,
    )
    for gene in genes:
        logger.info(f"EDA目标基因: {gene}")
        eda.run(gene, suffix=config.data.genomic.rnaseq_suffix,
                threshold_method=config.model.group_threshold)


def _run_train(config):
    """单基因训练 (完整5折交叉验证)。"""
    logger = _setup(config)
    from src.training.cv_runner import run_cross_validation
    logger.info(f"训练目标基因: {config.model.target_gene}, 任务: {config.model.task_type}, "
                f"算法: {config.upstream.algorithm_name}")
    run_cross_validation(config)


def _run_train_multi_gene(config, genes):
    """批量多基因训练: 每个基因独立训练并保存各自CSV日志。"""
    logger = _setup(config)
    from src.training.cv_runner import run_cross_validation

    base_exp = config.experiment.name
    for gene in genes:
        config.model.target_gene = gene
        # 每个基因独立实验目录
        config.experiment.name = f"{base_exp}_{gene}"
        logger.info("=" * 70)
        logger.info(f"  >>> 开始训练基因: {gene} <<<")
        logger.info("=" * 70)
        run_cross_validation(config)


def _run_test(config, checkpoint):
    """加载checkpoint在测试集评估。"""
    logger = _setup(config)
    import torch
    from src.training.cv_runner import _load_data_splits
    from src.networks.pathomic_net import PathomicNet
    from src.training.trainer import test

    if not checkpoint:
        logger.error("test模式需要 --checkpoint 指定模型权重路径")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_splits = _load_data_splits(config, logger)
    fold_name = sorted(data_splits.keys())[0]

    model = PathomicNet(config).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    result = test(config, model, data_splits[fold_name], split="test", device=device)
    logger.info(f"测试完成 (fold={fold_name})")


def _run_ablation(config):
    """消融实验: 对比不同上游patch采样算法对分子表型预测的影响。"""
    logger = _setup(config)
    from src.training.cv_runner import run_cross_validation
    import pandas as pd

    algorithms = config.ablation.algorithms
    genes = config.ablation.target_genes or [config.model.target_gene]

    base_exp = config.experiment.name
    all_summaries = []

    for gene in genes:
        config.model.target_gene = gene
        for alg in algorithms:
            config.upstream.algorithm_name = alg
            config.experiment.name = f"{base_exp}_ablation_{alg}_{gene}"
            logger.info("=" * 70)
            logger.info(f"  >>> 消融: 算法={alg}, 基因={gene} <<<")
            logger.info("=" * 70)
            try:
                cv_package = run_cross_validation(config)
                s = cv_package.get("cv_summary", {})
                s["algorithm"] = alg
                s["target_gene"] = gene
                all_summaries.append(s)
            except Exception as e:
                logger.error(f"  消融 {alg}/{gene} 失败: {e}")

    # 汇总CSV
    if all_summaries:
        results_dir = config.get_subdir("results")
        rows = []
        for s in all_summaries:
            row = {"algorithm": s.get("algorithm"), "target_gene": s.get("target_gene")}
            task_type = s.get("task_type", config.model.task_type)
            if task_type == "regression":
                for k in ["mse", "mae", "rmse", "pearson", "spearman"]:
                    row[k] = s.get(f"{k}_mean", float("nan"))
            else:
                for k in ["acc", "balanced_acc", "f1", "auc"]:
                    row[k] = s.get(f"{k}_mean", float("nan"))
            row["km_logrank_significant"] = s.get("km_logrank_significant_folds", 0)
            rows.append(row)

        summary_path = results_dir / "ablation_patch_sampling_summary.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False, float_format="%.4f")
        logger.info(f"  消融汇总CSV已保存: {summary_path}")


if __name__ == "__main__":
    main()
