import warnings

import os
import math

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import build_norm_layer
import torch.utils.model_zoo as model_zoo

from ..builder import BACKBONES


BatchNorm = nn.BatchNorm2d

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return os.path.join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 norm_cfg=dict(type='BN')):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 norm_cfg=dict(type='BN')):
        super(Bottleneck, self).__init__()
        self.norm_cfg = norm_cfg
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion

        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual,
                 norm_cfg=dict(type='BN'),):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False,
                 norm_cfg=dict(type='BN')):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation, norm_cfg=norm_cfg)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation, norm_cfg=norm_cfg)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual,
                              norm_cfg=norm_cfg)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual,
                              norm_cfg=norm_cfg)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual, norm_cfg=norm_cfg)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels

        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            _, norm = build_norm_layer(norm_cfg, out_channels, postfix=1)
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                norm
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


@BACKBONES.register_module()
class DLA(nn.Module):

    arch_settings = {
        34: (BasicBlock, (1, 1, 1, 2, 2, 1), (16, 32, 64, 128, 256, 512), False,
             'dla34', 'ba72cf86'),
        46: (Bottleneck, (1, 1, 1, 2, 2, 1), (16, 32, 64, 64, 128, 256), False,
             'dla46_c', '2bfd52c3'),
        60: (Bottleneck, (1, 1, 1, 2, 3, 1), (16, 32, 128, 256, 512, 1024), False,
             'dla60', '24839fc4'),
        102: (Bottleneck, (1, 1, 1, 3, 4, 1), (16, 32, 128, 256, 512, 1024), True,
              'dla102', 'd94d9790'),
    }

    def __init__(self, depth, in_channels=3,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 pretrained=None,):
        super(DLA, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth}')
        block, levels, channels, residual_root, model_name, download_hash = self.arch_settings[depth]
        self.model_name = model_name
        self.download_hash = download_hash

        self.channels = channels
        self.norm_eval = norm_eval
        _, norm1 = build_norm_layer(norm_cfg, channels[0], postfix=1)
        self.base_layer = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            norm1,
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0], norm_cfg=norm_cfg)
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2, norm_cfg=norm_cfg)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root,
                           norm_cfg=norm_cfg)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root,
                           norm_cfg=norm_cfg)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root,
                           norm_cfg=norm_cfg)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root,
                           norm_cfg=norm_cfg)

    def init_weights(self, pretrained=False):
        if pretrained:
            self.load_pretrained_model(name=self.model_name,
                                       hash=self.download_hash)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, _BatchNorm) or isinstance(m, nn.GroupNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        self.load_state_dict(model_weights, strict=False)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1,
                         norm_cfg=dict(type='BN')):
        modules = []
        _, norm = build_norm_layer(norm_cfg, planes, postfix=1)
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                norm,
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return tuple(y)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(DLA, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
