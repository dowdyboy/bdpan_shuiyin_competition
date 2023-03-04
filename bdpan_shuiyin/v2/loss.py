import paddle
import paddle.nn as nn
import numpy as np


class PSNRLoss(nn.Layer):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)

    def forward(self, pred, target):
        assert len(pred.shape) == 4

        return self.loss_weight * self.scale * paddle.log(((pred - target) ** 2).mean(axis=(1, 2, 3)) + 1e-8).mean()


class ShuiyinLoss(nn.Layer):

    def __init__(self, mode='l1', weight=[0.4, 0.5, 0.1]):
        super(ShuiyinLoss, self).__init__()
        assert mode in ['l1', 'psnr', ]
        self.weight = weight
        if mode == 'l1':
            self.main_loss = nn.L1Loss()
        if mode == 'psnr':
            self.main_loss = PSNRLoss()
        self.mask_loss = nn.BCELoss()

    def forward(self, pred, pred_mask, gt, gt_mask):
        loss_shuiyin = self.main_loss(pred * gt_mask, gt * gt_mask)
        loss_empty = self.main_loss(pred * (1. - gt_mask), gt * (1. - gt_mask))
        loss_mask = self.mask_loss(pred_mask, gt_mask)
        loss = loss_shuiyin * self.weight[0] + loss_empty * self.weight[1] + loss_mask * self.weight[2]
        # loss = loss_shuiyin * self.weight[0] + loss_empty * self.weight[1]
        return loss
