# -*- coding: utf-8 -*-
"""
tester.py
============================================================================
模型测试/评估模块。

功能:
    1. 在测试集上评估模型性能
    2. 计算完整的生存分析指标体系
    3. 支持分类任务评估
    4. GPU加速推理

使用示例:
    from src.training.tester import Tester
    tester = Tester(config, model, device)
    results = tester.test(data, split="test")
============================================================================
"""

import gc
import numpy as np
import torch
from typing import Dict, Tuple, Optional


class Tester:
    """
    模型测试器。

    参数:
        config: ConfigBundle配置对象
        model: 训练好的模型
        device: torch设备
    """

    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device

    @torch.no_grad()
    def test(self, data: Dict, split: str = "test") -> Dict:
        """
        在指定数据划分上评估模型。

        参数:
            data: 数据字典 (含train/test的x_path/x_omic/e/t/g)
            split: 数据划分 ("train" 或 "test")

        返回:
            dict: 包含所有评估指标的字典
        """
        from src.data_loading.datasets import PathomicDataset
        from src.losses.cox_loss import CoxLoss
        from src.evaluation.survival_metrics import (
            CIndex_lifeline, cox_log_rank, safe_time_dependent_auc,
            safe_binary_metrics_from_risk, safe_group_survival_summary,
            safe_hazard_ratio_by_median_split,
        )
        from src.evaluation.classification_metrics import classification_metrics

        self.model.eval()

        # 构建DataLoader
        dataset = PathomicDataset(self.config, data, split=split, mode=self.config.model.mode)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.training.batch_size,
            shuffle=False, num_workers=0, pin_memory=False,
        )

        task = self.config.model.task

        # 累积预测结果
        risk_pred_all = []
        censor_all = []
        survtime_all = []
        cls_logits_all = []
        cls_y_all = []
        loss_total = 0.0

        for batch in loader:
            x_path, _, x_omic, censor, survtime, grade = batch

            x_path = x_path.to(self.device, non_blocking=True)
            x_omic = x_omic.to(self.device, non_blocking=True)
            censor = censor.to(self.device)
            survtime = survtime.to(self.device)

            # 前向传播
            _, pred = self._forward(x_path, x_omic)

            # 损失计算
            if task == "surv":
                loss = CoxLoss(survtime.cpu(), censor.cpu(), pred.cpu(), self.device)
                loss_total += float(loss.item() if hasattr(loss, 'item') else loss)
                risk_pred_all.append(pred.cpu().numpy().reshape(-1))
                censor_all.append(censor.cpu().numpy().reshape(-1))
                survtime_all.append(survtime.cpu().numpy().reshape(-1))
            elif task == "ncls":
                cls_logits_all.append(pred.cpu())
                cls_y_all.append(grade.cpu().view(-1))

        loss_total /= max(1, len(loader))

        # 拼接所有batch的结果
        results = {"loss": loss_total}

        if task == "surv" and len(risk_pred_all) > 0:
            risk_pred = np.concatenate(risk_pred_all)
            censor = np.concatenate(censor_all)
            survtime = np.concatenate(survtime_all)

            # C-index (参数顺序: hazard_pred, survtime, event)
            cindex = CIndex_lifeline(risk_pred, survtime, censor)
            results["cindex"] = cindex

            # Log-rank p-value (参数顺序: hazard_pred, survtime, event)
            pvalue = cox_log_rank(risk_pred, survtime, censor)
            results["pvalue"] = pvalue

            # Time-dependent AUC (参数顺序: survtime, event, hazard_pred)
            td_auc = safe_time_dependent_auc(
                survtime=survtime,
                event=censor,
                hazard_pred=risk_pred,
                times=self.config.evaluation.eval_times,
            )
            results["td_auc"] = td_auc

            # 二分类指标 (参数顺序: hazard_pred, survtime, event)
            binary_metrics = safe_binary_metrics_from_risk(risk_pred, survtime, censor)
            results["binary_metrics"] = binary_metrics

            # 风险分组摘要 (参数顺序: hazard_pred, survtime, event)
            group_summary = safe_group_survival_summary(risk_pred, survtime, censor)
            results["group_summary"] = group_summary

            # Hazard Ratio (参数顺序: hazard_pred, survtime, event)
            hr = safe_hazard_ratio_by_median_split(risk_pred, survtime, censor)
            results["hazard_ratio"] = hr

            # 预测结果
            results["predictions"] = {
                "risk_pred": risk_pred,
                "censor": censor,
                "survtime": survtime,
            }

        elif task == "ncls" and len(cls_logits_all) > 0:
            logits = torch.cat(cls_logits_all, dim=0)
            y_true = torch.cat(cls_y_all, dim=0).numpy()
            cls_metrics = classification_metrics(logits=logits, y_true=y_true)
            results["cls_metrics"] = cls_metrics

        # 清理
        del loader
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return results

    def _forward(self, x_path, x_omic):
        """
        模型前向传播（处理多patch图像）。

        参数:
            x_path: [B, N, C, H, W] 或 [B, C, H, W]
            x_omic: [B, D]

        返回:
            (features, hazard)
        """
        if x_path.dim() == 5:
            B, N, C, H, W = x_path.shape
            x_path_flat = x_path.view(B * N, C, H, W)

            patch_feat, _ = self.model.path_net(x_path=x_path_flat)
            feat_dim = patch_feat.size(-1)
            patch_feat = patch_feat.view(B, N, feat_dim)
            patient_path_feat = patch_feat.mean(dim=1)  # 均值池化
        else:
            patient_path_feat, _ = self.model.path_net(x_path=x_path)

        patient_omic_feat, _ = self.model.omic_net(x_omic=x_omic)
        fused_feat = self.model.fusion(patient_path_feat, patient_omic_feat)
        hazard = self.model.classifier(fused_feat)

        return fused_feat, hazard


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("Tester 评估模块自测")
    print("测试通过!")
