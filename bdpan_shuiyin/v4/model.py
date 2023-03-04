import paddle
import paddle.nn as nn

from .sa_aidr import STRAIDR, NonLocalBlock
from dowdyboy_lib.paddle.model_util import frozen_layer


class NonLocalUpsample(nn.Layer):

    def __init__(self, in_channel, out_channel, num_c):
        super(NonLocalUpsample, self).__init__()
        self.coarse_conv = nn.Conv2D(in_channel, num_c, 3, 1, 1)
        self.non_local = NonLocalBlock(num_c)
        self.finetune_conv = nn.Conv2D(num_c, num_c, 3, 1, 1)
        self.merge_conv = nn.Conv2D(num_c * 2, out_channel, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.lrelu(self.coarse_conv(x))
        o1 = self.lrelu(self.non_local(x))
        o2 = self.lrelu(x + self.finetune_conv(x))
        out = self.lrelu(self.merge_conv(paddle.concat([o1, o2], axis=1)))
        out = self.upsample(out)
        return out


class ShuiyinModel(nn.Layer):
    
    def __init__(self, unet_pretrain=None, frozen_unet=False):
        super(ShuiyinModel, self).__init__()
        self.unet = STRAIDR(num_c=96)
        if unet_pretrain is not None:
            self.unet.load_dict(paddle.load(unet_pretrain))
            if frozen_unet:
                frozen_layer(self.unet)
            print(f'success load pretrain : {unet_pretrain}')
        self.upconv_o4 = NonLocalUpsample(256 * 2, 128, 256)
        self.upconv_o3 = NonLocalUpsample(128 * 3, 64, 128)
        self.upconv_o2 = NonLocalUpsample(64 * 3, 32, 64)
        self.upconv_o1 = NonLocalUpsample(32 * 3, 16, 32)
        self.out_conv_a = nn.Sequential(
            nn.Conv2D(16, 3, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.out_conv_b = nn.Conv2D(3, 3, 3, 1, 1)
        self._init_layers()

    def _init_layers(self):
        from bdpan_shuiyin.v4.init import init_model
        init_model(self.upconv_o4)
        init_model(self.upconv_o3)
        init_model(self.upconv_o2)
        init_model(self.upconv_o1)
        init_model(self.out_conv_a)
        init_model(self.out_conv_b)

    def forward(self, x):
        x_ori = x
        x_o_unet, xo1, xo2, xo3, xo4, con_x1, con_x2, con_x3, con_x4 = self.unet.forward_unet(x)
        u_o3 = self.upconv_o4(paddle.concat([xo4, con_x4], axis=1))
        u_o2 = self.upconv_o3(paddle.concat([xo3, con_x3, u_o3], axis=1))
        u_o1 = self.upconv_o2(paddle.concat([xo2, con_x2, u_o2], axis=1))
        u_o0 = self.upconv_o1(paddle.concat([xo1, con_x1, u_o1], axis=1))
        out = self.out_conv_b(self.out_conv_a(u_o0) + x_o_unet) + x_ori
        return out

