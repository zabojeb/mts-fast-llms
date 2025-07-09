import torch.nn.functional as F
from torch import Tensor
from torch import nn
import torch

class QuantizedLinear(nn.Module):
    def __init__(self, weight: nn.Parameter, num_bits: int = 8):
        super().__init__()

        self.num_bits = num_bits
        self.original_dtype = weight.dtype

        qmin = -(2 ** (num_bits - 1))
        qmax = 2 ** (num_bits - 1) - 1

        min_val = weight.min()
        max_val = weight.max()

        self.scale = (max_val - min_val) / (qmax - qmin)
        self.zero_point = qmax - max_val / self.scale

        if self.zero_point < qmin:
            self.zero_point = torch.tensor([qmin], dtype=torch.int8).to(min_val.device)
        elif self.zero_point > qmax:
            self.zero_point = torch.tensor([qmax], dtype=torch.int8).to(max_val.device)

        self.zero_point.round_()

        self.weight = self.zero_point + weight / self.scale
        self.weight.clamp_(qmin, qmax).round_()
        self.weight = self.weight.to(torch.int8)

        del weight

    def forward(self, x: Tensor):
        return F.linear(x, (self.scale * (self.weight - self.zero_point)))

    def __repr__(self):
        return "QuantizedLinear(in_features={}, out_features={}, num_bits={})".format(
            self.weight.shape[-1], self.weight.shape[-2], self.num_bits
        )