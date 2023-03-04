import paddle
from paddle.io import Dataset, DataLoader
import paddle.vision.transforms as T
import paddle.vision.transforms.functional as F
import random
import cv2
import numbers
import numpy as np
import os
from PIL import Image
from .watermark_generator import ShuiyinGenerator, ShuiyinRealGenerator


def _check_input(value,
                 name,
                 center=1,
                 bound=(0, float('inf')),
                 clip_first_on_zero=True):
    if isinstance(value, numbers.Number):
        if value < 0:
            raise ValueError(
                "If {} is a single number, it must be non negative.".format(
                    name))
        value = [center - value, center + value]
        if clip_first_on_zero:
            value[0] = max(value[0], 0)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError("{} values should be between {}".format(name,
                                                                     bound))
    else:
        raise TypeError(
            "{} should be a single number or a list/tuple with lenght 2.".
                format(name))

    if value[0] == value[1] == center:
        value = None
    return value


def load_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(x, save_path):
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, x)


class PairedRandomCrop(T.BaseTransform):

    def __init__(self, patch_size, keys=None):
        self.patch_size = patch_size
        self.keys = keys

    def __call__(self, inputs):
        """inputs must be (lq_img or list[lq_img], gt_img or list[gt_img])"""
        x_patch_size = self.patch_size

        in_x = inputs[0]
        in_y = inputs[1]

        ori_h, ori_w, _ = in_x[0].shape if isinstance(in_x, list) else in_x.shape
        if ori_h < x_patch_size or ori_w < x_patch_size:
            pre_pad_right = x_patch_size - ori_w if ori_w < x_patch_size else 0
            pre_pad_bottom = x_patch_size - ori_h if ori_h < x_patch_size else 0
            in_x = paddle.vision.transforms.pad(in_x, (0, 0, pre_pad_right, pre_pad_bottom), padding_mode='reflect')
            in_y = paddle.vision.transforms.pad(in_y, (0, 0, pre_pad_right, pre_pad_bottom), padding_mode='reflect')

        if isinstance(in_x, list):
            h_in_x, w_in_x, _ = in_x[0].shape
            h_in_y, w_in_y, _ = in_y[0].shape
        else:
            h_in_x, w_in_x, _ = in_x.shape
            h_in_y, w_in_y, _ = in_y.shape

        if h_in_y != h_in_x or w_in_y != w_in_x:
            raise ValueError('x y size not match')
        if h_in_x < x_patch_size or w_in_x < x_patch_size:
            raise ValueError('too small size error')

        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h_in_x - x_patch_size)
        left = random.randint(0, w_in_x - x_patch_size)

        if isinstance(in_x, list):
            in_x = [
                v[top:top + x_patch_size, left:left + x_patch_size, ...]
                for v in in_x
            ]
            in_y = [
                v[top:top + x_patch_size, left:left + x_patch_size, ...]
                for v in in_y
            ]
        else:
            in_x = in_x[top:top + x_patch_size, left:left + x_patch_size, ...]
            in_y = in_y[top:top + x_patch_size, left:left + x_patch_size, ...]

        outputs = (in_x, in_y)
        return outputs


class PairedRandomHorizontalFlip(T.RandomHorizontalFlip):

    def __init__(self, prob=0.5, keys=None):
        super().__init__(prob, keys=keys)

    def _get_params(self, inputs):
        params = {}
        params['flip'] = random.random() < self.prob
        return params

    def _apply_image(self, image):
        if self.params['flip']:
            if isinstance(image, list):
                image = [F.hflip(v) for v in image]
            else:
                return F.hflip(image)
        return image


class PairedRandomVerticalFlip(T.RandomVerticalFlip):

    def __init__(self, prob=0.5, keys=None):
        super().__init__(prob, keys=keys)

    def _get_params(self, inputs):
        params = {}
        params['flip'] = random.random() < self.prob
        return params

    def _apply_image(self, image):
        if self.params['flip']:
            if isinstance(image, list):
                image = [F.vflip(v) for v in image]
            else:
                return F.vflip(image)
        return image


class PairedRandomRotation(T.RandomRotation):

    def __init__(self, p, degrees, interpolation='nearest', expand=False,
                 center=None, fill=255, keys=None):
        super(PairedRandomRotation, self).__init__(degrees, interpolation=interpolation, expand=expand,
                                                   center=center, fill=fill, keys=keys)
        self.p = p

    def _get_params(self, inputs):
        params = {}
        if random.random() < self.p:
            angle = random.uniform(self.degrees[0], self.degrees[1])
        else:
            angle = None
        params['angle'] = angle
        return params

    def _apply_image(self, img):
        angle = self.params['angle']
        if angle is None:
            return img
        if isinstance(img, list):
            img = [F.rotate(v, angle, self.interpolation, self.expand,
                            self.center, [self.fill, self.fill, self.fill]) for v in img]
        else:
            img = F.rotate(img, angle, self.interpolation, self.expand,
                           self.center, [self.fill, self.fill, self.fill])
        return img


class PairedBrightnessTransform(T.BaseTransform):

    def __init__(self, value, keys=None):
        super(PairedBrightnessTransform, self).__init__(keys)
        self.value = _check_input(value, 'brightness')

    def _get_params(self, inputs):
        params = {}
        brightness_factor = random.uniform(self.value[0], self.value[1])
        params['value'] = brightness_factor
        return params

    def _apply_image(self, img):
        if self.params['value']:
            brightness_factor = self.params['value']
            if isinstance(img, list):
                return [F.adjust_brightness(v, brightness_factor) for v in img]
            else:
                return F.adjust_brightness(img, brightness_factor)
        return img


class PairedContrastTransform(T.BaseTransform):

    def __init__(self, value, keys=None):
        super(PairedContrastTransform, self).__init__(keys)
        if value < 0:
            raise ValueError("contrast value should be non-negative")
        self.value = _check_input(value, 'contrast')

    def _get_params(self, inputs):
        params = {}
        contrast_factor = random.uniform(self.value[0], self.value[1])
        params['value'] = contrast_factor
        return params

    def _apply_image(self, img):
        if self.params['value']:
            contrast_factor = self.params['value']
            if isinstance(img, list):
                return [F.adjust_contrast(v, contrast_factor) for v in img]
            else:
                return F.adjust_contrast(img, contrast_factor)
        return img


class PairedSaturationTransform(T.BaseTransform):

    def __init__(self, value, keys=None):
        super(PairedSaturationTransform, self).__init__(keys)
        self.value = _check_input(value, 'saturation')

    def _get_params(self, inputs):
        params = {}
        saturation_factor = random.uniform(self.value[0], self.value[1])
        params['value'] = saturation_factor
        return params

    def _apply_image(self, img):
        if self.params['value']:
            saturation_factor = self.params['value']
            if isinstance(img, list):
                return [F.adjust_saturation(v, saturation_factor) for v in img]
            else:
                return F.adjust_saturation(img, saturation_factor)
        return img


class PairedHueTransform(T.BaseTransform):

    def __init__(self, value, keys=None):
        super(PairedHueTransform, self).__init__(keys)
        self.value = _check_input(
            value, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def _get_params(self, inputs):
        params = {}
        hue_factor = random.uniform(self.value[0], self.value[1])
        params['value'] = hue_factor
        return params

    def _apply_image(self, img):
        if self.params['value']:
            hue_factor = self.params['value']
            if isinstance(img, list):
                return [F.adjust_hue(v, hue_factor) for v in img]
            else:
                return F.adjust_hue(img, hue_factor)
        return img


class PairedColorJitter(T.BaseTransform):

    def __init__(self, p, brightness=0, contrast=0, saturation=0, hue=0,
                 keys=None):
        super(PairedColorJitter, self).__init__(keys)
        self.p = p
        self.brightness = _check_input(brightness, 'brightness')
        self.contrast = _check_input(contrast, 'contrast')
        self.saturation = _check_input(saturation, 'saturation')
        self.hue = _check_input(
            hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        # self.brightness = brightness
        # self.contrast = contrast
        # self.saturation = saturation
        # self.hue = hue

    def _get_params(self, inputs):
        params = {}
        if random.random() < self.p:
            transforms = []
            brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            transforms.append(['brightness', brightness_factor])
            contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            transforms.append(['contrast', contrast_factor])
            saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            transforms.append(['saturation', saturation_factor])
            hue_factor = random.uniform(self.hue[0], self.hue[1])
            transforms.append(['hue', hue_factor])
            random.shuffle(transforms)
            params['transform'] = transforms

            # transforms = []
            # transforms.append(PairedBrightnessTransform(self.brightness, self.keys))
            # transforms.append(PairedContrastTransform(self.contrast, self.keys))
            # transforms.append(PairedSaturationTransform(self.saturation, self.keys))
            # transforms.append(PairedHueTransform(self.hue, self.keys))
            # random.shuffle(transforms)
            # # transform = T.Compose(transforms)
            # params['transform'] = transforms
        else:
            params['transform'] = None
        return params

    def _apply_image(self, img):
        transform = self.params['transform']
        if transform is None:
            return img
        for k, v in transform:
            if k == 'brightness':
                img = F.adjust_brightness(img, v)
            elif k == 'contrast':
                img = F.adjust_contrast(img, v)
            elif k == 'saturation':
                img = F.adjust_saturation(img, v)
            elif k == 'hue':
                img = F.adjust_hue(img, v)
        return img
        # return transform(img)


class PairedCutout(T.BaseTransform):

    def __init__(self, p=0.5, n_holes=[1, 3], length=[12, 32], fill=0, keys=None):
        super(PairedCutout, self).__init__(keys=keys)
        self.p = p
        self.n_holes = n_holes
        self.length = length
        self.fill = fill

    def _get_params(self, inputs):
        params = {}
        if random.random() < self.p:
            params['cutout'] = True
            params['holes'] = []
            n_hole = random.randint(self.n_holes[0], self.n_holes[1])
            h, w = inputs[0].shape[0], inputs[0].shape[1]
            for i in range(n_hole):
                hole_length = random.randint(self.length[0], self.length[1])
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(y - hole_length // 2, 0, h)
                y2 = np.clip(y + hole_length // 2, 0, h)
                x1 = np.clip(x - hole_length // 2, 0, w)
                x2 = np.clip(x + hole_length // 2, 0, w)
                params['holes'].append([x1, y1, x2, y2])
        else:
            params['cutout'] = False
        return params

    def _apply_image(self, image):
        is_cutout = self.params['cutout']
        if is_cutout:
            holes = self.params['holes']
            for x1, y1, x2, y2 in holes:
                if isinstance(image, list):
                    for i in range(len(image)):
                        image[i][y1:y2, x1:x2, :] = self.fill
                else:
                    image[y1:y2, x1:x2, :] = self.fill
        return image


class PairedSalt(T.BaseTransform):

    def __init__(self, p, rate=[0.05, 0.1], keys=None):
        super(PairedSalt, self).__init__(keys=keys)
        self.p = p
        self.rate = rate

    def _get_params(self, inputs):
        params = {}
        if random.random() < self.p:
            params['salt'] = True
            params['points'] = []
            final_rate = self.rate[0] + random.random() * (self.rate[1] - self.rate[0])
            h, w = inputs[0].shape[0], inputs[0].shape[1]
            noise_num = int(final_rate * h * w)
            for i in range(noise_num):
                y = random.randint(0, h-1)
                x = random.randint(0, w-1)
                v = 0 if random.random() < 0.5 else 255
                params['points'].append([x, y, v])
        else:
            params['salt'] = False
        return params

    def _apply_image(self, image):
        is_salt = self.params['salt']
        if is_salt:
            points = self.params['points']
            for x, y, v in points:
                if isinstance(image, list):
                    for i in range(len(image)):
                        image[i][y, x, :] = v
                else:
                    image[y, x, :] = v
        return image


class ShuiyinTrainDataset(Dataset):

    def __init__(self, root_dir, img_size,
                 hflip_p=0.4, vflip_p=0.2, rotate_p=0.1, color_jit_p=0.2,
                 cutout_big_p=0.05, cutout_small_p=0.05, salt_p=0.05,
                 real_p=0.1, is_to_tensor=True, is_val=False, val_ratio=0.1,
                 use_mem_cache=False, cache_dir=None, cache_live=[1, 3]):
        super(ShuiyinTrainDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.cache_dir = cache_dir if cache_dir is not None else f'cache_dataset_{random.randint(99, 999999)}'
        self.real_p = real_p
        self.is_to_tensor = is_to_tensor
        self.is_val = is_val
        self.val_ratio = val_ratio
        self.use_mem_cache = use_mem_cache
        self.cache_live = cache_live
        self.cache = {}
        self.bg_file_list = []
        self.generator = ShuiyinGenerator(
            font_dir=os.path.join(self.root_dir, 'watermark_scripts', 'fonts'),
            word_path=os.path.join(self.root_dir, 'watermark_scripts', 'data.txt'),
        )
        self.generator2 = ShuiyinRealGenerator(
            mask_dir=os.path.join(self.root_dir, 'mask_real'),
        )
        # data aug
        self.random_crop = PairedRandomCrop(img_size, keys=['image', 'image'])
        self.random_hflip = PairedRandomHorizontalFlip(prob=hflip_p, keys=['image', 'image'])
        self.random_vflip = PairedRandomVerticalFlip(prob=vflip_p, keys=['image', 'image'])
        self.random_rotate = PairedRandomRotation(p=rotate_p, degrees=15, keys=['image', 'image'])
        self.random_color_jit = PairedColorJitter(p=color_jit_p,
                                                  brightness=0.25, contrast=0.2, saturation=0.15, hue=0.1,
                                                  keys=['image', 'image'])
        self.cutout_big = PairedCutout(p=cutout_big_p, n_holes=[2, 3], length=[128, 256], keys=['image', 'image'])
        self.cutout_small = PairedCutout(p=cutout_small_p, n_holes=[800, 1200], length=[2, 6], keys=['image', 'image'])
        self.salt = PairedSalt(p=salt_p, keys=['image', 'image'])
        # self.channel_transpose = T.Transpose(keys=['image', 'image', ])
        self.to_tensor = T.ToTensor(keys=['image', 'image', ])
        self._init_dataset()

    def _init_dataset(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        self.bg_file_list = list(map(lambda x: os.path.join(self.root_dir, 'bg_pics', x),
                                     os.listdir(os.path.join(self.root_dir, 'bg_pics'))))

    def _apply_aug(self, x, y):
        # self.cutout_big((x, y))
        # self.cutout_small((x, y))
        # self.salt((x, y))
        # self.random_rotate((x, y))
        # self.random_hflip((x, y))
        # self.random_vflip((x, y))
        # self.random_color_jit((x, y))
        x, y = self.random_hflip((x, y))
        x, y = self.random_vflip((x, y))
        x, y = self.salt((x, y))
        x, y = self.random_color_jit((x, y))
        x, y = self.cutout_big((x, y))
        x, y = self.cutout_small((x, y))
        x, y = self.random_rotate((x, y))
        return x, y

    def _get_cache(self, idx):
        if self.use_mem_cache:
            x = np.copy(self.cache[idx]['x_img'])
            y = np.copy(self.cache[idx]['y_img'])
        else:
            x_path = self.cache[idx]['x_path']
            y_path = self.cache[idx]['y_path']
            x = load_image(x_path)
            y = load_image(y_path)
        self.cache[idx]['live'] -= 1
        return x, y

    def _put_cache(self, idx):
        im_bg = load_image(self.bg_file_list[idx])
        if random.random() < self.real_p:
            im_gen = self.generator2.generate_shuiyin_image(Image.fromarray(im_bg))
        else:
            im_gen = self.generator.generate_shuiyin_image(Image.fromarray(im_bg))
        im_gen = np.array(im_gen)
        if self.use_mem_cache:
            self.cache[idx] = {}
            self.cache[idx]['x_img'] = im_bg
            self.cache[idx]['y_img'] = im_gen
        else:
            filename = os.path.basename(self.bg_file_list[idx])
            filename = filename.split('.')
            x_path = os.path.join(self.cache_dir, f'{filename[0]}_x.{filename[1]}')
            y_path = os.path.join(self.cache_dir, f'{filename[0]}_y.{filename[1]}')
            save_image(im_bg, x_path)
            save_image(im_gen, y_path)
            self.cache[idx] = {}
            self.cache[idx]['x_path'] = x_path
            self.cache[idx]['y_path'] = y_path
        live = random.randint(self.cache_live[0], self.cache_live[1])
        self.cache[idx]['live'] = live

    def _get_filename(self, idx):
        return os.path.basename(self.bg_file_list[idx])

    def _get_mask(self, x, y):
        kernel = np.ones((2, 2), np.uint8)
        threshold = 10
        diff_image = np.abs(x.astype(np.float32) - y.astype(np.float32))
        mean_image = np.mean(diff_image, axis=-1)
        mask = np.greater(mean_image, threshold).astype(np.uint8)
        mask = (1 - mask) * 255
        mask = cv2.erode(np.uint8(mask),  kernel, iterations=1)
        mask = np.uint8(mask)
        mask = 255 - mask
        return mask

    def __getitem__(self, idx):
        if idx not in self.cache.keys() or self.cache[idx]['live'] <= 0:
            self._put_cache(idx)
        filename = self._get_filename(idx)
        x, y = self._get_cache(idx)
        x, y = self._apply_aug(x, y)
        if not self.is_val:
            x, y = self.random_crop((x, y))
        mask = self._get_mask(x, y)
        if self.is_to_tensor:
            x = self.to_tensor(x)
            y = self.to_tensor(y)
            mask = self.to_tensor(mask)
        # return x, y, mask, filename
        return y, x, mask, filename

    def __len__(self):
        if self.is_val:
            return int(len(self.bg_file_list) * self.val_ratio)
        else:
            return len(self.bg_file_list)


class ShuiyinTestDataset(Dataset):

    def __init__(self, data_dir, is_to_tensor=True):
        super(ShuiyinTestDataset, self).__init__()
        self.data_dir = data_dir
        self.is_to_tensor = is_to_tensor
        self.filepath_list = self._get_filepath_list()
        self.to_tensor = T.ToTensor()

    def _get_filepath_list(self):
        ret = []
        for filename in os.listdir(self.data_dir):
            x_path = os.path.join(self.data_dir, filename)
            ret.append(x_path)
        return ret

    def __getitem__(self, idx):
        x_path = self.filepath_list[idx]
        img = load_image(x_path)
        if self.is_to_tensor:
            img = self.to_tensor(img)
        return img, os.path.basename(x_path)

    def __len__(self):
        return len(self.filepath_list)
