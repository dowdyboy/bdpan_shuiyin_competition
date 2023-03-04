import os
import sys
import glob
import cv2
from PIL import Image
import numpy as np
from paddle.io import DataLoader
import paddle
import argparse

from bdpan_shuiyin.v6.sa_aidr import STRAIDR
from bdpan_shuiyin.v6.dataset2 import ShuiyinTestDataset

assert len(sys.argv) == 3
src_image_dir = sys.argv[1]
save_dir = sys.argv[2]
chk_path = 'checkpoints/v6/v6s1ep812.pd'
step = 700
pad = 34
h_flip_aug = True
v_flip_aug = False
hw_flip_aug = False
multi_scale_aug = False
steps = [500, 1000]
pads = [6, 12]


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


def save_img_arr(x, save_path):
    # Image.fromarray(x).save(
    #     save_path
    # )
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, x)


def save_mask_arr(mask, save_path):
    mask = np.squeeze(mask, axis=-1)
    cv2.imwrite(save_path, mask)


def build_data():
    test_dataset = ShuiyinTestDataset(src_image_dir, is_to_tensor=True)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, num_workers=0, drop_last=False)
    return test_loader, test_dataset


def build_model():
    model = STRAIDR(
        unet_num_c=[48, 96, 128, 256, 512],
        fine_num_c=[48, 96],
    )
    pre_weight = paddle.load(chk_path)
    model.load_dict(pre_weight)
    print(f'successful load pretrain : {chk_path}')
    model.eval()
    return model


def test_step_multi_scale(model, bat, dataset):
    res_list = []
    for step_i in range(len(steps)):
        step = steps[step_i]
        pad = pads[step_i]
        bat_im, bat_filename = bat
        bat_im = paddle.to_tensor(bat_im.numpy())
        _, _, ori_h, ori_w = bat_im.shape
        pre_m = None
        pre_pad_right = 0
        pre_pad_bottom = 0
        if ori_h < step + 2 * pad or ori_w < step + 2 * pad:
            if ori_h < step + 2 * pad:
                pre_pad_bottom = step + 2 * pad - ori_h
            if ori_w < step + 2 * pad:
                pre_pad_right = step + 2 * pad - ori_w
            pre_m = paddle.nn.Pad2D((0, pre_pad_right, 0, pre_pad_bottom), mode='reflect', )
            bat_im = pre_m(bat_im)
        m = paddle.nn.Pad2D(pad, mode='reflect')
        bat_im = m(bat_im)
        _, _, h, w = bat_im.shape
        res = paddle.zeros((bat_im.shape[0], bat_im.shape[1], h, w))

        all_count = 0

        for i in range(0, h, step):
            for j in range(0, w, step):
                all_count += 1
                if h - i < step + 2 * pad:
                    i = h - (step + 2 * pad)
                if w - j < step + 2 * pad:
                    j = w - (step + 2 * pad)
                multi_count = 1
                clip = bat_im[:, :, i:i + step + 2 * pad, j:j + step + 2 * pad]
                pred_y, pred_mask, _, _ = model(clip)
                if h_flip_aug:
                    pred_y_h_flip, pred_mask_h_flip, _, _ = model(paddle.flip(clip, axis=[3]))
                    pred_y_h_flip = paddle.flip(pred_y_h_flip, axis=[3])
                    pred_y += pred_y_h_flip
                    multi_count += 1
                if v_flip_aug:
                    pred_y_w_flip, pred_mask_w_flip, _, _ = model(paddle.flip(clip, axis=[2]))
                    pred_y_w_flip = paddle.flip(pred_y_w_flip, axis=[2])
                    pred_y += pred_y_w_flip
                    multi_count += 1
                if hw_flip_aug:
                    pred_y_hw_flip, pred_mask_hw_flip, _, _ = model(paddle.flip(clip, axis=[2, 3]))
                    pred_y_hw_flip = paddle.flip(pred_y_hw_flip, axis=[2, 3])
                    pred_y += pred_y_hw_flip
                    multi_count += 1
                pred_y = pred_y / multi_count

                res[:, :, i + pad:i + step + pad, j + pad:j + step + pad] = pred_y[:, :, pad:-pad, pad:-pad]

        # print(f'forward count : {all_count}')
        res = res[:, :, pad:-pad, pad:-pad]
        if pre_m is not None:
            res = res[:, :, :ori_h, :ori_w]
        res_list.append(res)
    if len(res_list) > 1:
        res = res_list[0]
        for i in range(1, len(res_list)):
            res += res_list[i]
        res /= len(res_list)
        res = to_img_arr(res[0])
        save_img_arr(res, os.path.join(save_dir, bat_filename[0]))
    else:
        res = res_list[0]
        res = to_img_arr(res[0])
        save_img_arr(res, os.path.join(save_dir, bat_filename[0]))


def test_step(model, bat, dataset):
    bat_im, bat_filename = bat
    _, _, ori_h, ori_w = bat_im.shape
    pre_m = None
    pre_pad_right = 0
    pre_pad_bottom = 0
    if ori_h < step + 2 * pad or ori_w < step + 2 * pad:
        if ori_h < step + 2 * pad:
            pre_pad_bottom = step + 2 * pad - ori_h
        if ori_w < step + 2 * pad:
            pre_pad_right = step + 2 * pad - ori_w
        pre_m = paddle.nn.Pad2D((0, pre_pad_right, 0, pre_pad_bottom), mode='reflect', )
        bat_im = pre_m(bat_im)
    m = paddle.nn.Pad2D(pad, mode='reflect')
    bat_im = m(bat_im)
    _, _, h, w = bat_im.shape
    res = paddle.zeros((bat_im.shape[0], bat_im.shape[1], h, w))
    res_mask = paddle.zeros((bat_im.shape[0], 1, h, w))

    all_count = 0

    for i in range(0, h, step):
        for j in range(0, w, step):
            all_count += 1
            if h - i < step + 2 * pad:
                i = h - (step + 2 * pad)
            if w - j < step + 2 * pad:
                j = w - (step + 2 * pad)
            multi_count = 1
            clip = bat_im[:, :, i:i + step + 2 * pad, j:j + step + 2 * pad]
            pred_y, pred_mask, _, _ = model(clip)
            if h_flip_aug:
                pred_y_h_flip, pred_mask_h_flip, _, _ = model(paddle.flip(clip, axis=[3]))
                pred_y_h_flip = paddle.flip(pred_y_h_flip, axis=[3])
                pred_mask_h_flip = paddle.flip(pred_mask_h_flip, axis=[3])
                pred_y += pred_y_h_flip
                pred_mask += pred_mask_h_flip
                multi_count += 1
            if v_flip_aug:
                pred_y_w_flip, pred_mask_w_flip, _, _ = model(paddle.flip(clip, axis=[2]))
                pred_y_w_flip = paddle.flip(pred_y_w_flip, axis=[2])
                pred_mask_w_flip = paddle.flip(pred_mask_w_flip, axis=[2])
                pred_y += pred_y_w_flip
                pred_mask += pred_mask_w_flip
                multi_count += 1
            if hw_flip_aug:
                pred_y_hw_flip, pred_mask_hw_flip, _, _ = model(paddle.flip(clip, axis=[2, 3]))
                pred_y_hw_flip = paddle.flip(pred_y_hw_flip, axis=[2, 3])
                pred_mask_hw_flip = paddle.flip(pred_mask_hw_flip, axis=[2, 3])
                pred_y += pred_y_hw_flip
                pred_mask += pred_mask_hw_flip
                multi_count += 1
            pred_y = pred_y / multi_count
            pred_mask = pred_mask / multi_count

            res[:, :, i + pad:i + step + pad, j + pad:j + step + pad] = pred_y[:, :, pad:-pad, pad:-pad]
            res_mask[:, :, i + pad:i + step + pad, j + pad:j + step + pad] = pred_mask[:, :, pad:-pad, pad:-pad]

    # print(f'forward count : {all_count}')
    res = res[:, :, pad:-pad, pad:-pad]
    res_mask = res_mask[:, :, pad:-pad, pad:-pad]
    if pre_m is not None:
        res = res[:, :, :ori_h, :ori_w]
        res_mask = res_mask[:, :, :ori_h, :ori_w]
    res = to_img_arr(res[0])
    res_mask = to_img_arr(res_mask[0])

    save_mask_arr(res_mask, os.path.join(save_dir, f'mask_{bat_filename[0]}'))
    save_img_arr(res, os.path.join(save_dir, bat_filename[0]))


def main():
    test_loader, test_dataset = build_data()
    model = build_model()

    for bat in test_loader:
        with paddle.no_grad():
            if multi_scale_aug:
                test_step_multi_scale(model, bat, test_dataset)
            else:
                test_step(model, bat, test_dataset)


if __name__ == '__main__':
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    main()


