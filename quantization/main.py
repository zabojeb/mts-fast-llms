from typing import NoReturn
from copy import deepcopy
from torch import nn

from .function import QuantizationFunction
from .config import QuantizationConfig

def quantize(
        model: nn.Module,
        config: QuantizationConfig,
        quantization_func: QuantizationFunction,
        ) -> nn.Module:
    
    """
    Quantizes copy of the model and returns it

    Kwargs:
        model: input model for quantization
        config: quantization config
        quantization_func: function for quantization

    Returns:
        q_model: quantized model
    """

    q_model = deepcopy(model)

    extra_kwargs = {name: kwarg for name, kwarg in vars(config) if kwarg is not None}

    for weight in q_model.parameters():
        quantization_func(
            weight=weight,
            **extra_kwargs,
        )

    return q_model