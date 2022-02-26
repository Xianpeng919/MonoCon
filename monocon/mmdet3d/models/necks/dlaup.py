import os, sys
import math
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from ..builder import NECKS


class Conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernal_szie=3, stride=1, bias=True,
                 norm_cfg=dict(type='BN')):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernal_szie,
                              stride=stride,
                              padding=kernal_szie//2,
                              bias=bias)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, out_planes, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm1(x)
        x = self.relu(x)
        return x


class IDAUp(nn.Module):
    '''
    input: features map of different layers
    output: up-sampled features
    '''
    def __init__(self, in_channels_list, up_factors_list, out_channels,
                 norm_cfg=dict(type='BN')):
        super(IDAUp, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        for i in range(1, len(in_channels_list)):
            in_channels = in_channels_list[i]
            up_factors = int(up_factors_list[i])

            proj = Conv2d(in_channels, out_channels, kernal_szie=3, stride=1, bias=False,
                          norm_cfg=norm_cfg)
            node = Conv2d(out_channels*2, out_channels, kernal_szie=3, stride=1, bias=False,
                          norm_cfg=norm_cfg)
            up = nn.ConvTranspose2d(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=up_factors * 2,
                                    stride=up_factors,
                                    padding=up_factors // 2,
                                    output_padding=0,
                                    groups=out_channels,
                                    bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, layers):
        assert len(self.in_channels_list) == len(layers), \
            '{} vs {} layers'.format(len(self.in_channels_list), len(layers))

        for i in range(1, len(layers)):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            node = getattr(self, 'node_' + str(i))

            layers[i] = upsample(project(layers[i]))
            layers[i] = node(torch.cat([layers[i-1], layers[i]], 1))

        return layers


@NECKS.register_module()
class DLAUp(nn.Module):
    def __init__(self, in_channels_list=[64, 128, 256, 512], scales_list=(1, 2, 4, 8),
                 start_level=2,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(DLAUp, self).__init__()
        scales_list = np.array(scales_list, dtype=int)
        self.in_channels_list = in_channels_list
        self.start_level = start_level

        for i in range(len(in_channels_list) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i), IDAUp(in_channels_list=in_channels_list[j:],
                                                    up_factors_list=scales_list[j:] // scales_list[j],
                                                    out_channels=in_channels_list[j],
                                                    norm_cfg=norm_cfg))
            scales_list[j + 1:] = scales_list[j]
            in_channels_list[j + 1:] = [in_channels_list[j] for _ in in_channels_list[j + 1:]]

    def init_weights(self):
        for i in range(len(self.in_channels_list) - 1):
            getattr(self, 'ida_{}'.format(i)).init_weights

    def forward(self, layers):
        layers = layers[self.start_level:]
        layers = list(layers)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            layers[-i - 2:] = ida(layers[-i - 2:])
        return [layers[-1]]


# weight init for up-sample layers [tranposed conv2d]
def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]
