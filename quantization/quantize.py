# Module done on 16.07 in 11:59

from random import sample
from torch import nn

from ..layers_worker import get_names_from_type
from .config import QuantizationConfig
from .methods import *


def quantize(
    model: nn.Module,
    config: QuantizationConfig,
) -> nn.Module:
    """
    Quantizes the model and returns it

    Kwargs:
        model: input model for quantization
        config: quantization config

    Returns:
        q_model: quantized model
    """

    if config.compute_dtype is not None:
        model = weights_to_dtype(model, config.compute_dtype)

    if config.layers is not None:
        for layer_type, layer in config.layers.items():
            if layer.fraction is not None:
                targets_names = get_names_from_type(model=model, layer_type=layer_type)
                to_quantize = sample(
                    targets_names, k=round(len(targets_names) * layer.fraction)
                )
            elif layer.names is not None:
                to_quantize = layer.names

            affine_transform(
                model=model, targets_names=to_quantize, num_bits=layer.qtype
            )

    return model


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2-xl",
        device_map="auto",
    )

    q_model = quantize(model, QuantizationConfig(compute_dtype=torch.float16)).to(
        "cuda"
    )

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
    inputs = tokenizer(
        "Once upon a time, there was a magical forest", return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
