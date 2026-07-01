# -*- coding: utf-8 -*-
"""
main.py
============================================================================
YuHou多模态预后预测 - 统一入口脚本。

运行模式:
    eda       - 数据探索性分析 (基因组+图像质量+任务论证)
    train     - 模型训练 (单融合策略)
    test      - 模型推理/评估
    ablation  - 消融实验 (批量运行多个配置)

使用示例:
    # EDA
    python main.py --mode eda

    # 训练 (默认CAUGF融合)
    python main.py --mode train

    # 训练 (指定融合策略)
    python main.py --mode train --fusion_type cross_attention --exp_name ca_exp

    # 消融实验
    python main.py --mode ablation --config config/ablation/fusion_comparison.yaml

    # 推理
    python main.py --mode test --checkpoint experiments/default/ckpt/best.pt

命令行参数:
    --config / -c     配置文件路径 (默认: config/default_config.yaml)
    --mode            运行模式 (eda / train / test / ablation)
    --fusion_type     融合策略类型 (覆盖配置文件)
    --exp_name        实验名称 (覆盖配置文件)
    --batch_size      批次大小 (覆盖配置文件)
    --lr              学习率 (覆盖配置文件)
    --gpu_id          GPU卡号
============================================================================
"""

import sys
import os
from pathlib import Path

# 将项目根目录加入Python路径
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    """主入口函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="YuHou - 结直肠癌COAD多模态预后预测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --mode eda                          # 数据探索性分析
  python main.py --mode train                        # 默认配置训练
  python main.py --mode train --fusion_type caugf    # CAUGF融合训练
  python main.py --mode ablation --config config/ablation/fusion_comparison.yaml
        """,
    )

    parser.add_argument("--config", "-c", type=str, default="config/default_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--mode", type=str, default="train",
                       choices=["eda", "train", "test", "ablation"],
                       help="运行模式")
    parser.add_argument("--fusion_type", type=str, default=None,
                       help="融合策略类型 (覆盖配置文件)")
    parser.add_argument("--exp_name", type=str, default=None,
                       help="实验名称 (覆盖配置文件)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="批次大小 (覆盖配置文件)")
    parser.add_argument("--lr", type=float, default=None,
                       help="学习率 (覆盖配置文件)")
    parser.add_argument("--gpu_id", type=int, default=None,
                       help="GPU卡号 (覆盖配置文件)")
    parser.add_argument("--n_epochs", type=int, default=None,
                       help="训练轮数 (覆盖配置文件)")
    parser.add_argument("--seed", type=int, default=None,
                       help="随机种子")

    args = parser.parse_args()

    # 加载配置
    from config.config_loader import load_config, print_config, create_experiment_dirs
    from src.utils.logger import setup_logger, log_section
    from src.utils.seed import set_seed

    print("=" * 70)
    print("  YuHou - 结直肠癌COAD多模态预后预测系统")
    print("  YuHou (预后) Multi-modal Prognosis Prediction")
    print("=" * 70)

    # 加载配置
    config = load_config(args.config)

    # 命令行参数覆盖
    if args.fusion_type:
        config.model.fusion.type = args.fusion_type
    if args.exp_name:
        config.experiment.name = args.exp_name
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.optimizer.lr = args.lr
    if args.gpu_id is not None:
        config.training.gpu_id = args.gpu_id
    if args.n_epochs:
        config.training.scheduler.n_epochs = args.n_epochs
    if args.seed:
        config.model.seed = args.seed

    # 创建实验目录
    exp_dirs = create_experiment_dirs(config)

    # 设置日志
    log_file = str(exp_dirs["logs"] / "main.log")
    logger = setup_logger(
        log_file=log_file,
        level=config.logging.level,
        language=config.logging.language,
    )

    # 设置随机种子
    set_seed(config.model.seed, logger=logger)

    # 打印配置
    print_config(config, logger=logger)

    # 根据模式执行
    mode = args.mode

    if mode == "eda":
        _run_eda(config, logger)
    elif mode == "train":
        _run_train(config, logger, exp_dirs)
    elif mode == "test":
        _run_test(config, logger, exp_dirs)
    elif mode == "ablation":
        _run_ablation(config, logger, exp_dirs)
    else:
        logger.error(f"未知的运行模式: {mode}")


def _run_eda(config, logger):
    """运行数据探索性分析"""
    log_section(logger, "数据探索性分析 (EDA)")

    # 基因组EDA
    try:
        from src.eda.genomic_eda import GenomicEDA
        genomic_csv = config.resolve_path(config.data.genomic_csv)
        if genomic_csv.exists():
            logger.info("开始基因组特征EDA...")
            eda = GenomicEDA(str(genomic_csv))
            eda.run_full_eda()
        else:
            logger.warning(f"基因组数据文件不存在: {genomic_csv}")
    except Exception as e:
        logger.error(f"基因组EDA失败: {e}")

    # 图像质量EDA
    try:
        from src.eda.image_quality_eda import ImageQualityEDA
        patch_dir = config.resolve_path(config.upstream.patch_root) / config.upstream.algorithm_name
        if patch_dir.exists():
            logger.info("开始图像质量EDA...")
            img_eda = ImageQualityEDA(str(patch_dir))
            img_eda.run_full_eda()
        else:
            logger.warning(f"上游patch目录不存在: {patch_dir}")
    except Exception as e:
        logger.error(f"图像质量EDA失败: {e}")

    # 任务论证
    try:
        from src.eda.task_justification import TaskJustification
        genomic_csv = config.resolve_path(config.data.genomic_csv)
        logger.info("生成任务合理性论证报告...")
        justifier = TaskJustification(str(genomic_csv) if genomic_csv.exists() else ".")
        justifier.load_data()
        justifier.plot_n_stage_km()
        report = justifier.generate_justification_report()
        logger.info(report)
    except Exception as e:
        logger.error(f"任务论证失败: {e}")

    logger.info("EDA完成!")


def _run_train(config, logger, exp_dirs):
    """运行模型训练"""
    log_section(logger, "模型训练")

    # 设置GPU
    from src.training.gpu_manager import setup_gpu, GPUMonitor
    device = setup_gpu(config)

    logger.info(f"训练设备: {device}")
    logger.info(f"融合策略: {config.model.fusion.type}")
    logger.info(f"批次大小: {config.training.batch_size}")
    logger.info(f"实验名称: {config.experiment.name}")

    # 构建模型
    from src.networks.pathomic_net import create_pathomic_net
    logger.info("构建PathomicNet多模态预测模型...")
    model = create_pathomic_net(config)
    model = model.to(device)

    logger.info(f"模型总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟训练（完整训练需要在cv_runner中执行）
    logger.info("训练就绪。请使用cv_runner执行完整的交叉验证训练。")
    logger.info("提示: 完整的5折CV训练流程将在后续版本中自动触发。")


def _run_test(config, logger, exp_dirs):
    """运行模型推理/评估"""
    log_section(logger, "模型推理与评估")
    logger.info("推理模式已就绪。")
    logger.info("请通过 --checkpoint 参数指定模型权重路径。")


def _run_ablation(config, logger, exp_dirs):
    """运行消融实验"""
    log_section(logger, "消融实验")
    logger.info("消融实验模式已就绪。")
    logger.info(f"请使用: --config config/ablation/fusion_comparison.yaml")


if __name__ == "__main__":
    main()
