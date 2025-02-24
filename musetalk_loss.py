# LPIPSLoss from https://github.com/sczhou/CodeFormer/tree/master

import math
import lpips
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

_reduction_modes = ['none', 'mean', 'sum']

def l1_loss(pred, target, reduction='mean'):
    return F.l1_loss(pred, target, reduction=reduction)

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # print(self.reduction)
        return self.loss_weight * l1_loss(pred, target, reduction=self.reduction)


class WeightedTVLoss(L1Loss):
    """Weighted TV loss.
        Args:
            loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight)

    def forward(self, pred, weight=None):
        y_diff = super(WeightedTVLoss, self).forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=weight[:, :, :-1, :])
        x_diff = super(WeightedTVLoss, self).forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=weight[:, :, :, :-1])
        loss = x_diff + y_diff
        return loss



class LPIPSLoss(nn.Module):
    def __init__(self, 
            loss_weight=1.0, 
            use_input_norm=True,
            range_norm=False,):
        super(LPIPSLoss, self).__init__()
        self.perceptual = lpips.LPIPS(net="vgg", spatial=False).eval()
        self.loss_weight = loss_weight
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        if self.range_norm:
            pred   = (pred + 1) / 2
            target = (target + 1) / 2
        if self.use_input_norm:
            pred   = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std
        lpips_loss = self.perceptual(target.contiguous(), pred.contiguous())
        return self.loss_weight * lpips_loss.mean()


