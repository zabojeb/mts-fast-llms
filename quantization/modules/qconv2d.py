# Module done on 17.07 in 14:33

from typing import Optional, Tuple
import torch.nn.functional as F
from torch import Tensor
from torch import nn
import torch


class Conv2dAffine8bit(nn.Module):
    def __init__(
        self,
        weight: nn.Parameter,
        bias: Optional[nn.Parameter] = None,
        stride: Optional[Tuple[int, int] | int] = None,
        padding: Optional[int | str] = None,
        dilation: Optional[int] = None,
        groups: Optional[int] = None,
    ):

        super().__init__()

        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = weight.shape[1]
        self.out_channels = weight.shape[0]

        with torch.no_grad():
            min_val, max_val = torch.aminmax(weight)

            self.scale = ((max_val - min_val) / 255).to(weight.dtype)
            self.zero_point = 255 - max_val / self.scale

            if self.zero_point < 0:
                self.zero_point = torch.tensor([0], dtype=weight.dtype).to(
                    min_val.device
                )
            elif self.zero_point > 255:
                self.zero_point = torch.tensor([255], dtype=weight.dtype).to(
                    max_val.device
                )

            self.zero_point.round_()

            self.weight = self.zero_point + weight / self.scale
            self.weight.clamp_(0, 255).round_()
            self.weight = self.weight.to(torch.uint8)

    def forward(self, x: Tensor):
        return F.conv2d(
            input=x,
            weight=self.scale * (self.weight - self.zero_point),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def __repr__(self):
        return "Conv2dAffine8bit(in_channels={}, out_channels={}, stride={}, padding={}, dilation={}, groups={}, scale={:.6f}, zero_point={})".format(
            self.in_channels,
            self.out_channels,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.scale,
            self.zero_point,
        )
