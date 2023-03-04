import argparse
import os
import sys
import math
import textwrap
import random
import glob
import hashlib

from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageChops
from numpy import full
import numpy as np


class WatermarkGenerator(object):

    def __init__(self,
                 text,
                 font_path,
                 size,
                 color,
                 font_height_crop,
                 space,
                 angle,
                 full,
                 alpha):
        self.text = text
        self.font_path = font_path
        self.size = size
        self.color = color
        self.font_height_crop = font_height_crop
        self.space = space
        self.angle = angle
        self.full = full
        self.alpha = alpha

    def generate(self, back_image):
        mark = self._gen_font()
        if self.full:
            return self._mark_im_full(back_image, mark)
        
        return self._mark_im_single(back_image, mark)
    
    def _gen_font(self):
        width = len(self.text) * self.size
        height = round(self.size * float(self.font_height_crop))

        # 创建水印图片(宽度、高度)
        mark = Image.new(mode='RGBA', size=(width, height))

        # 生成文字
        draw_table = ImageDraw.Draw(im=mark)
        draw_table.text(xy=(0, 0),
                        text=self.text,
                        fill=self.color,
                        font=ImageFont.truetype(self.font_path,
                                                size=self.size))

        mark = self._crop_image(mark)

        mark = self._set_opacity(mark, self.alpha)

        return mark

    def _crop_image(self, im):
        """
        裁剪图片边缘空白
        """
        
        bg = Image.new(mode='RGBA', size=im.size)
        diff = ImageChops.difference(im, bg)
        del bg
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)
        
        return im

    def _set_opacity(self, im, opacity):
        """
        设置水印透明度
        """
        
        assert opacity >= 0 and opacity <= 1

        alpha = im.split()[3]
        alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
        im.putalpha(alpha)
        
        return im

    def _mark_im_full(self, im, mark):
        """
        在im图片上添加水印 im为打开的原图
        """

        # 计算斜边长度
        c = int(math.sqrt(im.size[0] * im.size[0] + im.size[1] * im.size[1]))

        # 以斜边长度为宽高创建大图（旋转后大图才足以覆盖原图）
        mark2 = Image.new(mode='RGBA', size=(c, c))

        # 在大图上生成水印文字，此处mark为上面生成的水印图片
        y, idx = 0, 0
        while y < c:
            # 制造x坐标错位
            x = -int((mark.size[0] + self.space) * 0.5 * idx)
            idx = (idx + 1) % 2

            while x < c:
                # 在该位置粘贴mark水印图片
                mark2.paste(mark, (x, y))
                x = x + mark.size[0] + self.space
            y = y + mark.size[1] + self.space

        # 将大图旋转一定角度
        mark2 = mark2.rotate(self.angle)

        # 在原图上添加大图水印
        if im.mode != 'RGBA':
            im = im.convert('RGBA')
        im.paste(mark2,  # 大图
                 (int((im.size[0] - c) / 2), int((im.size[1] - c) / 2)),  # 坐标
                 mask=mark2.split()[3])
        del mark2

        return im

    def _mark_im_single(self, im, mark):
        """
        在im图片上添加mark水印图片, 只生成单条水印
        """

        # 计算斜边长度
        c = int(math.sqrt(im.size[0] * im.size[0] + im.size[1] * im.size[1]))

        # 以斜边长度为宽高创建大图（旋转后大图才足以覆盖原图）
        mark2 = Image.new(mode='RGBA', size=(c, c))
        # if (c > 2000):
        #     x_offset = int(mark2.size[0] / 4)
        #     y_offset = int(mark2.size[1] / 4)
        # else:
        #     x_offset = 0
        #     y_offset = 0
        # x = random.randint(x_offset, mark2.size[0] - x_offset - mark.size[0])
        # y = random.randint(y_offset, mark2.size[1] - y_offset - mark.size[1])
        x = int((mark2.size[0] - mark.size[0]) / 2)
        y = int((mark2.size[1] - mark.size[1]) / 2)
        mark2.paste(mark, (x, y))

        mark2 = mark2.rotate(self.angle)

        # 在原图上添加大图水印
        if im.mode != 'RGBA':
            im = im.convert('RGBA')
        im.paste(mark2,  # 大图
                 (int((im.size[0] - c) / 2), int((im.size[1] - c) / 2)),  # 坐标
                 mask=mark2.split()[3])
        del mark2

        return im


def random_generate_text(words, length=(2, 10)):
    text_num = random.randint(length[0], length[1])
    rand_words = random.choices(words, k=text_num)

    return "".join(rand_words)


def random_generate(bg_image_dir, font_dir, word_path, save_dir):
    tmp_bg_image_paths = glob.glob(os.path.join(bg_image_dir, "*"))
    bg_image_paths = []
    for image_path in tmp_bg_image_paths:
        ext = os.path.splitext(os.path.basename(image_path))[1]
        if ext.lower() in [".jpg", ".png"]:
            bg_image_paths.append(image_path)
    
    tmp_font_paths = glob.glob(os.path.join(font_dir, "*"))
    font_paths = []
    for font_path in tmp_font_paths:
        ext = os.path.splitext(os.path.basename(font_path))[1]
        if ext.lower() in [".ttf", ".otf"]:
            font_paths.append(font_path)
    
    words = []
    with open(word_path, "r") as fid:
        for line in fid:
            words.append(line.strip())
    
    for bg_image_path in bg_image_paths:
        print(bg_image_path)
        for i in range(10):
            try:
                text = random_generate_text(words)
                font_path = random.choice(font_paths)
                font_size = random.randint(30, 70)
                alpha = random.randint(10, 50) / 100.
                angle = random.randint(10, 70)
                space = random.randint(20, 90)
                font_height_crop = 1.2
                color = [0] * 3
                color[0] = random.randint(0, 200)
                color[1] = random.randint(0, 200)
                color[2] = random.randint(0, 200)

                full_flag = random.random()
                if full_flag >= 0.4:
                    full = True
                else:
                    full = False

                bg_image = Image.open(bg_image_path)
                bg_image.thumbnail((2056, 2056), Image.ANTIALIAS)
                min_side = min(bg_image.size)
                ratio = int(min_side / 500.)
                ratio = max(1, ratio)
                font_size = font_size * ratio
                space = space * ratio

                generator = WatermarkGenerator(text=text, font_path=font_path, size=font_size, 
                                            color=tuple(color), font_height_crop=font_height_crop,
                                            space=space, angle=angle, full=full, alpha=alpha)
            
                name = os.path.splitext(os.path.basename(bg_image_path))[0]
                gen_image = generator.generate(bg_image)
                gen_image = gen_image.convert('RGB')

                save_path = os.path.join(save_dir, "{}_{}.jpg".format(name, i))

                gen_image.save(save_path)
            except Exception as e:
                print("Get exception: {}, {}, {}".format(bg_image_path, text, e))
                continue


class ShuiyinGenerator():

    def __init__(self, font_dir, word_path,
                 font_size=[30, 70], alpha=[10, 50],
                 angle=[10, 70], space=[20, 90], font_height_crop=1.2,
                 colors=[[0, 200], [0, 200], [0, 200]], full_ratio=0.4):
        tmp_font_paths = glob.glob(os.path.join(font_dir, "*"))
        font_paths = []
        for font_path in tmp_font_paths:
            ext = os.path.splitext(os.path.basename(font_path))[1]
            if ext.lower() in [".ttf", ".otf"]:
                font_paths.append(font_path)
        self.font_paths = font_paths
        words = []
        with open(word_path, "r", encoding='utf-8') as fid:
            for line in fid:
                words.append(line.strip())
        self.words = words

        self.font_size = font_size
        self.alpha = alpha
        self.angle = angle
        self.space = space
        self.font_height_crop = font_height_crop
        self.colors = colors
        self.full_ratio = full_ratio

    def generate_shuiyin_image(self, bg_image):
        text = random_generate_text(self.words)
        font_path = random.choice(self.font_paths)
        font_size = random.randint(self.font_size[0], self.font_size[1])
        alpha = random.randint(self.alpha[0], self.alpha[1]) / 100.
        angle = random.randint(self.angle[0], self.angle[1])
        space = random.randint(self.space[0], self.space[1])
        font_height_crop = self.font_height_crop
        color = [0] * 3
        color[0] = random.randint(self.colors[0][0], self.colors[0][1])
        color[1] = random.randint(self.colors[1][0], self.colors[1][1])
        color[2] = random.randint(self.colors[2][0], self.colors[2][1])
        full_flag = random.random()
        if full_flag <= self.full_ratio:
            full = True
        else:
            full = False
        bg_image.thumbnail((2056, 2056), Image.ANTIALIAS)
        min_side = min(bg_image.size)
        ratio = int(min_side / 500.)
        ratio = max(1, ratio)
        font_size = font_size * ratio
        space = space * ratio
        generator = WatermarkGenerator(text=text, font_path=font_path, size=font_size,
                                       color=tuple(color), font_height_crop=font_height_crop,
                                       space=space, angle=angle, full=full, alpha=alpha)
        gen_image = generator.generate(bg_image)
        gen_image = gen_image.convert('RGB')
        return gen_image


class ShuiyinRealGenerator():

    def __init__(self, mask_dir,
                 h_flip=0.25, v_flip=0.25,
                 alpha=[10, 50], angle=[10, 70], colors=[[0, 200], [0, 200], [0, 200]]):
        super(ShuiyinRealGenerator, self).__init__()
        self.mask_dir = mask_dir
        self.alpha = alpha
        self.angle = angle
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.colors = colors
        self.mask_imgs = []
        self._init_generator()

    def _init_generator(self):
        for mask_filename in os.listdir(self.mask_dir):
            mask_path = os.path.join(self.mask_dir, mask_filename)
            im_mask = np.array(Image.open(mask_path))
            im_mask[im_mask < 32] = 0
            im_mask[im_mask >= 32] = 1
            im_mask = im_mask[:, :, 0]
            self.mask_imgs.append(im_mask)

    def generate_shuiyin_image(self, bg_image):
        alpha = random.randint(self.alpha[0], self.alpha[1]) / 100.
        angle = random.randint(self.angle[0], self.angle[1])
        color = [0] * 3
        color[0] = random.randint(self.colors[0][0], self.colors[0][1])
        color[1] = random.randint(self.colors[1][0], self.colors[1][1])
        color[2] = random.randint(self.colors[2][0], self.colors[2][1])
        idx = random.randint(0, len(self.mask_imgs) - 1)
        im_mask = np.copy(self.mask_imgs[idx])
        ori_im_mask = np.copy(im_mask)
        im_mask = im_mask.reshape(im_mask.shape + (1, ))
        im_mask = np.concatenate([im_mask, im_mask, im_mask, im_mask], axis=2)
        im_mask[:, :, 0] *= color[0]
        im_mask[:, :, 1] *= color[1]
        im_mask[:, :, 2] *= color[2]
        im_mask[:, :, 3] *= int(255 * alpha)
        if random.random() < self.v_flip:
            im_mask = im_mask[::-1, :, :]
        if random.random() < self.h_flip:
            im_mask = im_mask[:, ::-1, :]
        mask_image = Image.fromarray(im_mask)
        mask_image = mask_image.rotate(angle)
        if bg_image.mode != 'RGBA':
            bg_image = bg_image.convert('RGBA')
        mask_image = mask_image.resize(bg_image.size)
        bg_image.paste(mask_image,  # 大图
                       (0, 0),  # 坐标
                       mask=mask_image.split()[3])
        bg_image = bg_image.convert('RGB')
        return bg_image


def main():
    bg_image_dir = "background_images_testA"
    font_dir = "fonts"
    word_path = "data.txt"
    save_dir = "generate_images_with_watermark_for_testA"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    random_generate(bg_image_dir=bg_image_dir, font_dir=font_dir, word_path=word_path, save_dir=save_dir)

