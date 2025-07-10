import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from torch import nn
import torch


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        weight: nn.Parameter,
        bias: Optional[nn.Parameter] = None,
    ):
        super().__init__()

        self.bias = bias

        with torch.no_grad():
            min_val = weight.detach().min()
            max_val = weight.detach().max()

            self.scale = ((max_val - min_val) / 255).to(weight.dtype)
            self.zero_point = 127 - max_val / self.scale

            if self.zero_point < -128:
                self.zero_point = torch.tensor([-128], dtype=torch.int8).to(
                    min_val.device
                )
            elif self.zero_point > 127:
                self.zero_point = torch.tensor([127], dtype=torch.int8).to(
                    max_val.device
                )

            self.zero_point.round_()
            self.zero_point = self.zero_point.to(weight.dtype)

            self.weight = self.zero_point + weight / self.scale
            self.weight.clamp_(-128, 127).round_()
            self.weight = self.weight.to(torch.int8)

    def forward(self, x: Tensor):
        return F.linear(
            # x, (self.scale * (self.weight.float() - self.zero_point)), self.bias  Time coeff: ~2.14 x; Memory coeff: 2.53 x
            x,
            (self.scale * (self.weight - self.zero_point)),
            self.bias,  #                                                           Time coeff: ~1.49 x; Memory coeff: 2.53 x
        )

    def __repr__(self):
        return "QuantizedLinear(in_features={}, out_features={}, scale={:.6f}, zero_point={})".format(
            self.weight.shape[-1],
            self.weight.shape[-2],
            self.scale,
            self.zero_point,
        )
