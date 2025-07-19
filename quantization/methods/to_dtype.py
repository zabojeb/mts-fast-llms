# Module done on 16.07 in 11:28

from torch import dtype
from torch import nn
import torch


def weights_to_dtype(model: nn.Module, dtype: dtype) -> nn.Module:
    """
    Applies dtype to model and returns it

    Args:
        model: model for convertation
        dtype: torch dtype that will be applied

    Returns:
        model with certain dtype
    """

    return model.to(dtype)


if __name__ == "__main__":
    torch.set_printoptions(precision=32)
    W = torch.tensor(
        [1.123456789012345, 1.23456789012345, 1.23456789012345], dtype=torch.float32
    )

    print(W)
    print(weights_to_dtype(W, torch.float16))
