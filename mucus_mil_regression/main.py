# -*- coding: utf-8 -*-
"""
main.py - MucusMIL Regression 统一入口。

mode:
    eda        - 生成PCA PC1粘液分化得分标签 (一次性)
    train      - 训练 (五折交叉验证)
    test       - 加载checkpoint在测试集评估
    ablation   - 三组对照/融合方式/采样算法消融
    visualize  - 从已保存预测CSV画散点图/消融图

示例:
    python main.py --mode eda
    python main.py --mode train --model_type image_mil --backbone resnet
    python main.py --mode train --model_type clinical_mlp
    python main.py --mode train --model_type multimodal --fusion_type adaptive_gate
    python main.py --mode ablation --axis models
    python main.py --mode ablation --axis algorithms
    python main.py --mode ablation --axis fusions
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MucusMIL Regression")
    parser.add_argument("--config", "-c", type=str, default="config/default_config.yaml")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["eda", "train", "test", "ablation", "visualize"])
    parser.add_argument("--model_type", type=str, default=None,
                        choices=["clinical_mlp", "image_mil", "multimodal"])
    parser.add_argument("--backbone", type=str, default=None, choices=["resnet", "uni"])
    parser.add_argument("--fusion_type", type=str, default=None,
                        choices=["late_concat", "adaptive_gate"])
    parser.add_argument("--algorithm_name", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="test模式权重路径")
    parser.add_argument("--axis", type=str, default="models",
                        choices=["models", "algorithms", "fusions"])
    args = parser.parse_args()

    from config.config_loader import load_config, print_config, create_experiment_dirs
    # cli_args=[] 让 load_config 不再重复解析命令行(main.py已自行解析并应用),
    # 避免 --mode 被误匹配为 --model_type 之类的前缀冲突
    config = load_config(args.config, cli_args=[])

    if args.model_type:
        config.model.model_type = args.model_type
    if args.backbone:
        config.model.backbone = args.backbone
    if args.fusion_type:
        config.model.fusion_type = args.fusion_type
    if args.algorithm_name:
        config.upstream.algorithm_name = args.algorithm_name
    if args.exp_name:
        config.experiment.name = args.exp_name
    if args.gpu_id is not None:
        config.training.gpu_id = args.gpu_id

    create_experiment_dirs(config)
    print_config(config)

    if args.mode == "eda":
        _run_eda(config)
    elif args.mode == "train":
        _run_train(config)
    elif args.mode == "test":
        _run_test(config, args.checkpoint)
    elif args.mode == "ablation":
        _run_ablation(config, args.axis)
    elif args.mode == "visualize":
        _run_visualize(config)


def _run_eda(config):
    from src.eda.pca_label_builder import build_pc1_labels
    build_pc1_labels(config)


def _run_train(config):
    from src.training.cv_runner import run_cross_validation
    run_cross_validation(config)


def _run_test(config, checkpoint):
    import torch
    from src.training.cv_runner import _build_all_data
    from src.training.trainer import build_model, test
    from src.utils.logger import setup_logger

    logger = setup_logger(level=config.logging.level, language=config.logging.language)
    if not checkpoint:
        logger.error("test模式需要 --checkpoint 指定权重路径")
        return

    data = _build_all_data(config, logger)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()

    fold_name = sorted(data.keys())[0]
    result = test(config, model, data[fold_name], "test", device)
    from src.evaluation.regression_metrics import regression_metrics
    m = regression_metrics(result["preds"], result["targets"])
    logger.info(f"  [TEST] fold={fold_name}")
    for k, v in m.items():
        if isinstance(v, float):
            logger.info(f"    {k}: {v:.4f}")


def _run_ablation(config, axis):
    from src.training.cv_runner import run_cross_validation
    import pandas as pd
    import numpy as np

    base_name = config.experiment.name
    summary_rows = []

    def _collect():
        cv = run_cross_validation(config)
        # 取均值行 (最后两行为mean/std)
        df = cv["fold_metrics_df"]
        mean = df[df["fold"] == "mean"]
        if len(mean):
            row = {"spearman": float(mean["spearman"].iloc[0]),
                   "pearson": float(mean["pearson"].iloc[0]),
                   "mae": float(mean["mae"].iloc[0]),
                   "rmse": float(mean["rmse"].iloc[0]),
                   "r2": float(mean["r2"].iloc[0])}
        else:
            row = {}
        return row

    if axis == "models":
        configs = [
            ("clinical_mlp", "clinical_mlp", None, None, None),
            ("image_mil", "image_mil", "resnet", None, None),
            ("multimodal_lateconcat", "multimodal", "resnet", "late_concat", None),
            ("multimodal_adaptivegate", "multimodal", "resnet", "adaptive_gate", None),
        ]
        for tag, mt, bb, ft, alg in configs:
            config.model.model_type = mt
            if bb:
                config.model.backbone = bb
            if ft:
                config.model.fusion_type = ft
            config.experiment.name = f"{base_name}_{tag}"
            print(f"\n===== 消融: {tag} =====")
            row = {"config": tag, **_collect()}
            summary_rows.append(row)

    elif axis == "algorithms":
        config.model.model_type = "image_mil"
        for alg in config.ablation.algorithms:
            config.upstream.algorithm_name = alg
            config.experiment.name = f"{base_name}_alg_{alg}"
            print(f"\n===== 消融算法: {alg} =====")
            row = {"config": alg, **_collect()}
            summary_rows.append(row)

    elif axis == "fusions":
        config.model.model_type = "multimodal"
        for ft in config.ablation.fusions:
            config.model.fusion_type = ft
            config.experiment.name = f"{base_name}_fusion_{ft}"
            print(f"\n===== 消融融合: {ft} =====")
            row = {"config": ft, **_collect()}
            summary_rows.append(row)

    if summary_rows:
        out = config.get_subdir("results") / f"{base_name}_ablation_{axis}_summary.csv"
        pd.DataFrame(summary_rows).to_csv(out, index=False, float_format="%.4f")
        print(f"\n消融汇总已保存: {out}")
        # 画柱状图
        try:
            from src.visualization.ablation_plots import plot_ablation_bar
            df = pd.DataFrame(summary_rows)
            fig_dir = config.get_subdir("figures")
            plot_ablation_bar(df, "config", "spearman",
                              save_path=str(fig_dir / f"{base_name}_ablation_{axis}.png"))
        except Exception as e:
            print(f"柱状图绘制失败: {e}")


def _run_visualize(config):
    from src.visualization.scatter import plot_pred_true_scatter
    import pandas as pd
    results_dir = config.get_subdir("results")
    fig_dir = config.get_subdir("figures")
    csv_path = results_dir / f"{config.experiment.name}_patient_predictions.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        plot_pred_true_scatter(df["y_true"].values, df["y_pred"].values,
                               save_path=str(fig_dir / f"{config.experiment.name}_scatter.png"),
                               title=f"{config.experiment.name} 预测vs真实")
        print(f"散点图已保存: {fig_dir / f'{config.experiment.name}_scatter.png'}")
    else:
        print(f"未找到预测CSV: {csv_path}, 请先运行 train")


if __name__ == "__main__":
    main()
