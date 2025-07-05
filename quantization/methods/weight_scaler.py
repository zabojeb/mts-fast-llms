from torch import Tensor
import torch


def quantize_weights(weights: Tensor, num_bits: int = 8) -> tuple:
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    min_val = weights.min().item()
    max_val = weights.max().item()

    if min_val == max_val:
        min_val -= 0.001
        max_val += 0.001

    scale = (max_val - min_val) / (qmax - qmin)

    zero_point = qmin - min_val / scale
    zero_point = int(
        torch.clamp(torch.round(torch.tensor(zero_point)), qmin, qmax).item()
    )

    quantized = torch.clamp(torch.round(weights / scale + zero_point), qmin, qmax).to(
        torch.int8
    )

    return quantized, scale, zero_point
