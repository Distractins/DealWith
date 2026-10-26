# -*- coding: utf-8 -*-
"""regression_loss.py - MSE回归损失 (bag级弱监督)。"""

import torch
import torch.nn as nn


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred.view(-1), target.view(-1))
