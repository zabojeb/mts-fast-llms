# Module not implemented

from torchao.dtypes.uintx.bitpacking import pack, unpack
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from torch import nn
import torch

from ...exceptions import QuantizationError


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        weight: nn.Parameter,
        bias: Optional[nn.Parameter] = None,
        num_bits: int = 8,
    ):
        super().__init__()

        if num_bits < 1 or num_bits > 8:
            raise QuantizationError("Number of bits must be from 1 to 8")

        self.bias = bias
        self.num_bits = num_bits
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

        with torch.no_grad():
            qmax = 2 ** (num_bits) - 1

            min_val, max_val = torch.aminmax(weight)

            self.scale = ((max_val - min_val) / qmax).to(weight.dtype)
            self.zero_point = qmax - max_val / self.scale

            if self.zero_point < 0:
                self.zero_point = torch.tensor([0], dtype=weight.dtype).to(
                    min_val.device
                )
            elif self.zero_point > qmax:
                self.zero_point = torch.tensor([qmax], dtype=weight.dtype).to(
                    max_val.device
                )

            self.zero_point.round_()

            self.weight = self.zero_point + weight / self.scale
            self.weight.clamp_(0, qmax).round_()
            self.weight = self.weight.to(torch.uint8)

            try:
                self.weight = (
                    pack(self.weight, num_bits, -2) if num_bits != 8 else self.weight
                )
            except AssertionError:
                # TODO: Implement it
                raise QuantizationError(
                    "Input and output features must be divisble by scale (8)"
                )

    def forward(self, x: Tensor):
        return F.linear(
            x,
            (
                self.scale
                * (
                    (
                        unpack(self.weight, self.num_bits, -2)
                        if self.num_bits != 8
                        else self.weight
                    )
                    - self.zero_point
                )
            ),
            self.bias,
        )

    def __repr__(self):
        return "QuantizedLinear(in_features={}, out_features={}, scale={:.6f}, zero_point={}, num_bits={})".format(
            self.in_features,
            self.out_features,
            self.scale,
            self.zero_point,
            self.num_bits,
        )
