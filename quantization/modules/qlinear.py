# Module done on 16.07 in 11:30

import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from torch import nn
import torch


class LinearAffine8bit(nn.Module):
    def __init__(
        self,
        weight: nn.Parameter,
        bias: Optional[nn.Parameter] = None,
    ):

        super().__init__()

        self.bias = bias
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

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
        return F.linear(
            x,
            self.scale * (self.weight - self.zero_point),
            self.bias,
        )

    def __repr__(self):
        return "LinearAffine8bit(in_features={}, out_features={}, scale={:.6f}, zero_point={})".format(
            self.in_features,
            self.out_features,
            self.scale,
            self.zero_point,
        )
