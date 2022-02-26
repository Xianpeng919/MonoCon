import torch
from torch import nn as nn

from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class CenterNetGaussianFocalLoss(nn.Module):
    """Calculate the IoU loss (1-IoU) of axis aligned bounding boxes.

    Args:
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, loss_weight=1.0, gamma=2.0, beta=4.0, alpha=-1):
        super(CenterNetGaussianFocalLoss, self).__init__()
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha

    def forward(self,
                input,
                target,
                **kwargs):
        """Forward function of loss calculation.

        Args:
            pred (torch.Tensor): Bbox predictions with shape [..., 3].
            target (torch.Tensor): Bbox targets (gt) with shape [..., 3].
            weight (torch.Tensor|float, optional): Weight of loss. \
                Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): Method to reduce losses.
                The valid reduction method are 'none', 'sum' or 'mean'.
                Defaults to None.

        Returns:
            torch.Tensor: IoU loss between predictions and targets.
        """
        eps = 1e-12

        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        num_pos = pos_inds.sum()

        neg_weights = torch.pow(1 - target, self.beta)

        loss = 0

        pos_loss = torch.log(input + eps) * torch.pow(1 - input, self.gamma) * pos_inds
        neg_loss = torch.log(1 - input + eps) * torch.pow(input, self.gamma) * neg_weights * neg_inds

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if self.alpha >= 0:
            pos_loss = self.alpha * pos_loss
            neg_loss = (1 - self.alpha) * neg_loss

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss.mean() * self.loss_weight
