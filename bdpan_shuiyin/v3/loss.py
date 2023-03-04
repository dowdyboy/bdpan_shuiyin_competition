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
                 mode='l1',
                 main_weight=0.5,
                 mm_weight=0.1,
                 aux_o4_weight=0.1,
                 aux_o2_weight=0.1, ):
        super(ShuiyinLoss, self).__init__()
        assert mode in ['l1', 'psnr', ]
        self.main_weight = main_weight
        self.mm_weight = mm_weight
        self.aux_o4_weight = aux_o4_weight
        self.aux_o2_weight = aux_o2_weight
        if mode == 'l1':
            self.main_loss = nn.L1Loss()
        if mode == 'psnr':
            self.main_loss = PSNRLoss()
        self.mask_loss = nn.BCELoss()

    def forward(self, pred, pred_mask, aux_o4, aux_o2, gt, gt_mask):
        loss_shuiyin = self.main_loss(pred * gt_mask, gt * gt_mask)
        loss_empty = self.main_loss(pred * (1. - gt_mask), gt * (1. - gt_mask))
        loss_o4 = self.main_loss(aux_o4, gt)
        loss_o2 = self.main_loss(aux_o2, gt)
        loss_mask = self.mask_loss(pred_mask, gt_mask)
        loss = (loss_shuiyin + loss_empty) * self.main_weight + loss_mask * self.mm_weight + \
            loss_o4 * self.aux_o4_weight + loss_o2 * self.aux_o2_weight
        return loss
