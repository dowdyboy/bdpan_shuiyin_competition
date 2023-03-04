import paddle
from paddle.io import DataLoader
import os
import argparse
from PIL import Image
import numpy as np
import random

from dowdyboy_lib.paddle.trainer import Trainer, TrainerConfig

from bdpan_shuiyin.v6.loss import ShuiyinLoss
from bdpan_shuiyin.v6.optim import CosineAnnealingRestartLR
from bdpan_shuiyin.v6.dataset2 import ShuiyinTrainDataset
from bdpan_shuiyin.v6.psnr_ssim import calculate_psnr, calculate_ssim

import paddle.nn.functional as F


parser = argparse.ArgumentParser(description='train shuiyin multi scale image')
# model config
parser.add_argument('--model-type', type=str, default='baseline', help='model type')
# data config
parser.add_argument('--train-data-dir', type=str, required=True, help='train data dir')
parser.add_argument('--val-data-dir', type=str, required=True, help='val data dir')
# parser.add_argument('--data-dir', type=str, required=True, help='train data dir')
parser.add_argument('--num-workers', type=int, default=4, help='num workers')
parser.add_argument('--img-size', type=int, default=1024, help='input img size')
parser.add_argument('--ms', type=int, nargs='+', default=[256, 384, 512, 640, 768, 896, 1024, 1152, 1280], help='multi scale crop')
# optimizer config
parser.add_argument('--lr', type=float, default=1e-5, help='lr')
parser.add_argument('--use-scheduler', default=False, action='store_true', help='use schedule')
parser.add_argument('--use-warmup', default=False, action='store_true', help='use warmup')
parser.add_argument('--weight-decay', type=float, default=0., help='model weight decay')
# train config
parser.add_argument('--epoch', type=int, default=1024, help='epoch num')
parser.add_argument('--batch-size', type=int, default=8, help='batch size')
parser.add_argument('--out-dir', type=str, default='./output_multi_scale', help='out dir')
parser.add_argument('--resume', type=str, default=None, help='resume checkpoint')
parser.add_argument('--last-epoch', type=int, default=-1, help='last epoch')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--log-interval', type=int, default=500, help='log process')
parser.add_argument('--save-val-count', type=int, default=50, help='log process')
parser.add_argument('--sync-bn', default=False, action='store_true', help='sync_bn')
parser.add_argument('--device', default=None, type=str, help='device')
parser.add_argument('--multi-gpu', default=False, action='store_true', help='whether to use multi gpu')
parser.add_argument('--amp', default=False, action='store_true', help='use amp to train')
args = parser.parse_args()


def to_img_arr(x, un_norm=None):
    if un_norm is not None:
        y = un_norm((x, x, x))[0]
        y = y.numpy().transpose(1, 2, 0)
        y = np.clip(y, 0., 255.).astype(np.uint8)
    else:
        y = x.numpy().transpose(1, 2, 0)
        y = np.clip(y, 0., 1.)
        y = (y * 255).astype(np.uint8)
    return y


def do_random_crop(inputs, x_patch_size):
    # x_patch_size = args.ms[random.randint(0, len(args.ms) - 1)]
    in_x = inputs[0]
    in_y = inputs[1]
    in_mask = inputs[2]
    in_x = paddle.unsqueeze(in_x, axis=0)
    in_y = paddle.unsqueeze(in_y, axis=0)
    in_mask = paddle.unsqueeze(in_mask, axis=0)

    _, _, ori_h, ori_w = in_x[0].shape if isinstance(in_x, list) else in_x.shape
    if ori_h < x_patch_size or ori_w < x_patch_size:
        pre_pad_right = x_patch_size - ori_w if ori_w < x_patch_size else 0
        pre_pad_bottom = x_patch_size - ori_h if ori_h < x_patch_size else 0
        # in_x = paddle.vision.transforms.pad(in_x, (0, 0, pre_pad_right, pre_pad_bottom), padding_mode='reflect')
        # in_y = paddle.vision.transforms.pad(in_y, (0, 0, pre_pad_right, pre_pad_bottom), padding_mode='reflect')
        # in_mask = paddle.vision.transforms.pad(in_mask, (0, 0, pre_pad_right, pre_pad_bottom), padding_mode='reflect')
        in_x = F.pad(in_x, (0, pre_pad_right, 0, pre_pad_bottom), value=1.)
        in_y = F.pad(in_y, (0, pre_pad_right, 0, pre_pad_bottom), value=1.)
        in_mask = F.pad(in_mask, (0, pre_pad_right, 0, pre_pad_bottom), value=0.)

    if isinstance(in_x, list):
        # h_in_x, w_in_x, _ = in_x[0].shape
        # h_in_y, w_in_y, _ = in_y[0].shape
        # h_in_mask, w_in_mask, _ = in_mask[0].shape
        raise NotImplementedError()
    else:
        _, _, h_in_x, w_in_x = in_x.shape
        _, _, h_in_y, w_in_y = in_y.shape
        _, _, h_in_mask, w_in_mask = in_mask.shape

    if h_in_y != h_in_x or w_in_y != w_in_x:
        raise ValueError('x y size not match')
    if h_in_x < x_patch_size or w_in_x < x_patch_size:
        raise ValueError('too small size error')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_in_x - x_patch_size)
    left = random.randint(0, w_in_x - x_patch_size)

    if isinstance(in_x, list):
        # in_x = [
        #     v[top:top + x_patch_size, left:left + x_patch_size, ...]
        #     for v in in_x
        # ]
        # in_y = [
        #     v[top:top + x_patch_size, left:left + x_patch_size, ...]
        #     for v in in_y
        # ]
        raise NotImplementedError()
    else:
        in_x = in_x[..., top:top + x_patch_size, left:left + x_patch_size]
        in_y = in_y[..., top:top + x_patch_size, left:left + x_patch_size]
        in_mask = in_mask[..., top:top + x_patch_size, left:left + x_patch_size]

    return in_x, in_y, in_mask


def merge_and_crop_bat(bat):
    x_patch_size = args.ms[random.randint(0, len(args.ms) - 1)]
    bat_x, bat_y, bat_mask, bat_filename = [], [], [], []
    for item in bat:
        x, y, m = do_random_crop([item[0], item[1], item[2]], x_patch_size)
        bat_x.append(x)
        bat_y.append(y)
        bat_mask.append(m)
        bat_filename.append(item[3])
    bat_x = paddle.concat(bat_x, axis=0)
    bat_y = paddle.concat(bat_y, axis=0)
    bat_mask = paddle.concat(bat_mask, axis=0)
    return bat_x, bat_y, bat_mask, bat_filename


def build_data():
    train_dataset = ShuiyinTrainDataset(
        root_dir=args.train_data_dir,
        img_size=args.img_size,
        hflip_p=0.5, vflip_p=0.5, rotate_p=0.5, color_jit_p=0.5,
        cutout_big_p=0.0, cutout_small_p=0.0, salt_p=0.0,
        real_p=0.5, logo_p=0.75,
        is_to_tensor=True, use_mem_cache=False, cache_dir='cache_dataset_train', cache_live=[1, 3],
        is_val=True, val_ratio=1.0,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True, collate_fn=merge_and_crop_bat)
    val_dataset = ShuiyinTrainDataset(
        root_dir=args.val_data_dir,
        img_size=args.img_size,
        hflip_p=0.3, vflip_p=0.0, rotate_p=0.3, color_jit_p=0.2,
        cutout_big_p=0.1, cutout_small_p=0.1, salt_p=0.1,
        real_p=0.2, logo_p=0.5,
        is_to_tensor=True, use_mem_cache=False, cache_dir='cache_dataset_val', cache_live=[99999, 999999],
        is_val=True, val_ratio=1.0,
    )
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=args.num_workers, drop_last=False)
    return train_loader, train_dataset, val_loader, val_dataset


def build_model():
    assert args.model_type in ['baseline', 'v6']
    if args.model_type == 'baseline':
        from bdpan_shuiyin.v1.sa_aidr import STRAIDR
        model = STRAIDR(num_c=96, )
        pre_weight = paddle.load('checkpoints/model_best_v1.pdparams')
        model.load_dict(pre_weight)
        print('successful load pretrain : checkpoints/model_best_v1.pdparams')
    else:
        from bdpan_shuiyin.v6.sa_aidr import STRAIDR
        from bdpan_shuiyin.v6.init import init_model
        model = STRAIDR(
            unet_num_c=[48, 96, 128, 256, 512],
            fine_num_c=[48, 96],
        )
        init_model(model)
    # pre_weight = paddle.load('checkpoints/model_best_v1.pdparams')
    # model.load_dict(pre_weight)
    # print('successful load pretrain : checkpoints/model_best_v1.pdparams')
    return model


# def pad_bat(bat_x, bat_y, bat_mask):
#     div_by = 32
#     _, _, h, w = bat_x.shape
#     pad_h = 0 if h % div_by == 0 else div_by - (h % div_by)
#     pad_w = 0 if w % div_by == 0 else div_by - (w % div_by)
#     bat_x = F.pad(bat_x, (0, pad_w, 0, pad_h), mode='reflect', )
#     bat_y = F.pad(bat_y, (0, pad_w, 0, pad_h), mode='reflect', )
#     bat_mask = F.pad(bat_mask, (0, pad_w, 0, pad_h), mode='reflect', )
#     return bat_x, bat_y, bat_mask


def do_model(model, bat_x, ):
    # if is_pad:
    #     div_by = 32
    #     _, _, h, w = bat_x.shape
    #     pad_h = 0 if h % div_by == 0 else div_by - (h % div_by)
    #     pad_w = 0 if w % div_by == 0 else div_by - (w % div_by)
    #     bat_x = F.pad(bat_x, (0, pad_w, 0, pad_h), value=1.)
    if args.model_type == 'v6':
        pred_y, pred_mask, _, _ = model(bat_x)
    elif args.model_type == 'baseline':
        _, _, _, pred_y, pred_mask = model(bat_x)
    return pred_y, pred_mask, 0, 0


def build_optimizer(model):
    # interval = 250000
    interval = 90000
    lr = args.lr
    lr_scheduler = None
    if args.use_scheduler:
        # lr = paddle.optimizer.lr.CosineAnnealingDecay(lr, args.epoch, last_epoch=args.last_epoch, verbose=True)
        lr = CosineAnnealingRestartLR(
            lr,
            periods=[interval, interval, interval, interval],
            restart_weights=[1, 1, 1, 1],
            eta_min=args.lr * 0.01,
            last_epoch=args.last_epoch,
        )
        lr_scheduler = lr
    if args.use_warmup:
        lr = paddle.optimizer.lr.LinearWarmup(lr, 10, args.lr * 0.1, args.lr, last_epoch=args.last_epoch, verbose=True)
        lr_scheduler = lr
    optimizer = paddle.optimizer.Adam(
        lr,
        parameters=[{
            'params': m.parameters()
        } for m in model] if isinstance(model, list) else model.parameters(),
        weight_decay=args.weight_decay,
        beta1=0.9,
        beta2=0.99,
    )
    return optimizer, lr_scheduler


def train_step(trainer: Trainer, bat, bat_idx, global_step):
    [model] = trainer.get_models()
    [loss_func] = trainer.get_components()
    _, [lr_scheduler] = trainer.get_optimizers()

    bat_x, bat_y, bat_mask, bat_filename = bat
    # bat_x, bat_y, bat_mask = pad_bat(bat_x, bat_y, bat_mask)
    # pred_y, pred_mask, _, _ = model(bat_x)
    pred_y, pred_mask, _, _ = do_model(model, bat_x)
    loss = loss_func(pred_y, pred_mask, bat_y, bat_mask)

    trainer.log({
        'train_loss': loss.item(),
    }, global_step)
    trainer.set_records({
        'train_loss': loss.item(),
    })
    if global_step % args.log_interval == 0:
        trainer.print(f'global step: {global_step}, loss: {loss.item()}')

    trainer.step(lr_scheduler=lr_scheduler)
    return loss


def val_step(trainer: Trainer, bat, bat_idx, global_step):
    [model] = trainer.get_models()
    [loss_func] = trainer.get_components()

    bat_x, bat_y, bat_mask, bat_filename = bat
    _, _, h, w = bat_x.shape
    _, _, h_y, w_y = bat_y.shape
    rh_y, rw_y = h_y, w_y
    step = args.img_size
    pad_h = step - h if h < step else 0
    pad_w = step - w if w < step else 0
    m = paddle.nn.Pad2D((0, pad_w, 0, pad_h))
    bat_x = m(bat_x)
    bat_y = m(bat_y)
    bat_mask = m(bat_mask)
    _, _, h, w = bat_x.shape
    _, _, h_y, w_y = bat_y.shape
    res_y = paddle.zeros_like(bat_y)
    loss_list = []
    for i in range(0, h, step):
        for j in range(0, w, step):
            if h - i < step:
                i = h - step
            if w - j < step:
                j = w - step
            clip_x = bat_x[:, :, i:i+step, j:j+step]
            clip_y = bat_y[:, :, i:i+step, j:j+step]
            clip_mask = bat_mask[:, :, i:i+step, j:j+step]
            # pred_y, pred_mask, _, _ = model(clip_x)
            pred_y, pred_mask, _, _ = do_model(model, clip_x)
            loss = loss_func(pred_y, pred_mask, clip_y, clip_mask)
            loss_list.append(loss.item())
            res_y[:, :, i:i+step, j:j+step] = pred_y
    loss = sum(loss_list) / len(loss_list)
    res_y = res_y[:, :, :rh_y, :rw_y]
    bat_y = bat_y[:, :, :rh_y, :rw_y]

    pred_im = to_img_arr(res_y[0])
    gt_im = to_img_arr(bat_y[0])
    psnr = float(calculate_psnr(pred_im, gt_im, crop_border=4, test_y_channel=True, ))
    ssim = float(calculate_ssim(pred_im, gt_im, crop_border=4, test_y_channel=True, ))
    # psnr_ssim = psnr / 15. + ssim
    psnr_ssim = psnr + ssim

    trainer.log({
        'val_loss': loss,
        'val_psnr': psnr,
        'val_ssim': ssim,
        'val_psnr_ssim': psnr_ssim,
    }, global_step)
    trainer.set_bar_state({
        'val_psnr': psnr,
        'val_ssim': ssim,
    })
    trainer.set_records({
        'psnr_ssim': psnr_ssim,
        'mean_psnr': psnr,
        'mean_ssim': ssim,
        'val_loss': loss,
    })
    Image.fromarray(pred_im).save(os.path.join(args.out_dir, f'{global_step % args.save_val_count}_pred.png'))
    Image.fromarray(gt_im).save(os.path.join(args.out_dir, f'{global_step % args.save_val_count}_gt.png'))
    return loss


def on_epoch_end(trainer: Trainer, ep):
    [optimizer], [lr_scheduler] = trainer.get_optimizers()
    rec = trainer.get_records()
    psnr_ssim = paddle.mean(rec['psnr_ssim']).item()
    mean_psnr = paddle.mean(rec['mean_psnr']).item()
    mean_ssim = paddle.mean(rec['mean_ssim']).item()
    val_loss = paddle.mean(rec['val_loss']).item()
    trainer.log({
        'ep_psnr_ssim': psnr_ssim,
        'ep_mean_psnr': mean_psnr,
        'ep_mean_ssim': mean_ssim,
        'ep_val_loss': val_loss,
        'ep_lr': optimizer.get_lr(),
    }, ep)
    trainer.print(f'loss : {val_loss}, mean_psnr : {mean_psnr}, mean_ssim : {mean_ssim}, '
                  f'psnr_ssim : {psnr_ssim}, lr : {optimizer.get_lr()}')


def main():
    cfg = TrainerConfig(
        epoch=args.epoch,
        out_dir=args.out_dir,
        mixed_precision='fp16' if args.amp else 'no',
        multi_gpu=args.multi_gpu,
        device=args.device,
        save_interval=5,
        save_best=True,
        save_best_type='max',
        save_best_rec='psnr_ssim',
        seed=args.seed,
        auto_optimize=True,
        auto_schedule=False,
        auto_free=False,
        sync_bn=args.sync_bn,
    )
    trainer = Trainer(cfg)
    trainer.print(args)

    train_loader, train_dataset, val_loader, val_dataset = build_data()
    trainer.print(f'train size: {len(train_dataset)}, val size: {len(val_dataset)}')

    model = build_model()
    loss_func = ShuiyinLoss(mode='l1', weight=[0.5, 0.5, 0.0])
    optimizer, lr_scheduler = build_optimizer(model)

    trainer.set_train_dataloader(train_loader)
    trainer.set_val_dataloader(val_loader)
    trainer.set_model(model)
    trainer.set_components([loss_func])
    trainer.set_optimizer(optimizer, lr_scheduler=lr_scheduler)

    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
        trainer.print(f'load checkpoint from {args.resume}')

    trainer.fit(
        train_step=train_step,
        val_step=val_step,
        on_epoch_end=on_epoch_end,
    )


if __name__ == '__main__':
    main()
