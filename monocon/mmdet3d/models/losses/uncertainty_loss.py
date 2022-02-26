import torch
from torch import nn as nn

from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class LaplacianAleatoricUncertaintyLoss(nn.Module):
    """

    Args:
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """
    def __init__(self, loss_weight=1.0):
        super(LaplacianAleatoricUncertaintyLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self,
                input,
                target,
                log_variance):

        log_variance = log_variance.flatten()
        input = input.flatten()
        target = target.flatten()

        loss = 1.4142 * torch.exp(-log_variance) * torch.abs(input - target) + log_variance

        return loss.mean() * self.loss_weight


@LOSSES.register_module()
class GaussianAleatoricUncertaintyLoss(nn.Module):
    """

    Args:
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """
    def __init__(self, loss_weight=1.0):
        super(GaussianAleatoricUncertaintyLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self,
                input,
                target,
                log_variance):

        log_variance = log_variance.flatten()
        input = input.flatten()
        target = target.flatten()

        loss = 0.5 * torch.exp(-log_variance) * torch.abs(input - target)**2 + 0.5 * log_variance

        return loss.mean() * self.loss_weight
