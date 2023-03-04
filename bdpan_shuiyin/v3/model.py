import paddle
import paddle.nn as nn
from .sa_aidr import STRAIDR, DeConvWithActivation


class AuxHead(nn.Layer):

    def __init__(self, channels=[], out_channel=3):
        super(AuxHead, self).__init__()
        assert len(channels) > 1
        self.layers = nn.LayerList()
        for i in range(len(channels) - 1):
            self.layers.append(
                DeConvWithActivation(channels[i], channels[i+1], kernel_size=3, padding=1, stride=2)
            )
        self.last = nn.Conv2D(channels[-1], out_channel, 3, 1, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.last(x)
        return x


class ShuiyinModel(nn.Layer):

    def __init__(self):
        super(ShuiyinModel, self).__init__()
        self.aidr = STRAIDR()
        self.aux_head_o4 = AuxHead(channels=[64, 32, 16], out_channel=3)
        self.aux_head_o2 = AuxHead(channels=[32, 16], out_channel=3)

    def forward(self, x):
        out, mm, xo4, xo2 = self.aidr(x)
        out_o4 = self.aux_head_o4(xo4)
        out_o2 = self.aux_head_o2(xo2)
        return out, mm, out_o4, out_o2
