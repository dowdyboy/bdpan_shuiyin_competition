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

    def __init__(self,
                 mode='l1', ):
        super(ShuiyinLoss, self).__init__()
        assert mode in ['l1', 'psnr', ]
        if mode == 'l1':
            self.main_loss = nn.L1Loss()
        if mode == 'psnr':
            self.main_loss = PSNRLoss()

    def forward(self, pred, gt):
        loss = self.main_loss(pred, gt)
        return loss
