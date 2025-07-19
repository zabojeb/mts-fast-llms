# Module done on 16.07 in 12:38

from typing import List, Tuple
from torch import nn


def predict_memory(
    model: nn.Module, targets_names: List[str], num_bits: int
) -> Tuple[int, int]:
    """
    Computes size of original and compressed model and returns them

    Args:
        model: model to be computed
        targets_names: names of layers to be compressed
        num_bits: quantity of bits for compute

    Returns:
        Tuple[
            original model size,
            compressed model size,
        ]
    """

    original_size = 0
    compressed_size = 0

    for name, layer in model.named_modules():
        if not hasattr(layer, "weight"):
            continue

        if name in targets_names:
            compressed_size += num_bits * layer.weight.numel() // 8
        else:
            compressed_size += layer.weight.numel() * layer.weight.dtype.itemsize

        original_size += layer.weight.numel() * layer.weight.dtype.itemsize

    return original_size, compressed_size
