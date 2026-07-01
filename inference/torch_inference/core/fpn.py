"""Pure-PyTorch FPN (ports mmdet FPN; matches checkpoint keys neck.*.conv.*)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    """Conv2d wrapper exposing ``.conv`` to match mmdet FPN param names."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)


class FPN(nn.Module):
    def __init__(self, in_channels=(1024,) * 4, out_channels=256, num_outs=5,
                 start_level=0, add_extra_convs='on_output', relu_before_extra_convs=True):
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.num_ins = len(in_channels)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.lateral_convs.append(ConvModule(in_channels[i], out_channels, 1))
            self.fpn_convs.append(ConvModule(out_channels, out_channels, 3, padding=1))
        # extra convs for levels beyond the input pyramid
        self.extra_convs = []
        n_extra = num_outs - self.num_ins
        for i in range(n_extra):
            self.fpn_convs.append(ConvModule(out_channels, out_channels, 3, padding=1))
        self.add_extra_convs = add_extra_convs
        self.relu_before_extra_convs = relu_before_extra_convs

    def forward(self, inputs):
        assert len(inputs) == self.num_ins
        laterals = [self.lateral_convs[i](inputs[i]) for i in range(self.num_ins)]
        for i in range(self.num_ins - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode='nearest')
        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]
        # one extra level (stride 64) on the last output
        if self.num_outs > self.num_ins:
            extra = outs[-1]
            if self.relu_before_extra_convs:
                extra = F.relu(extra)
            outs.append(self.fpn_convs[self.num_ins](extra))
        return tuple(outs)
