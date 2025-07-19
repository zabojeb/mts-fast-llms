# Module done on 16.07 in 11:26

from bitsandbytes.nn import Linear4bit, Params4bit
from typing import List, NoReturn
from torch import nn
import gc

from ...exceptions import QuantizationError
from ..modules import *


def affine_transform(
    model: nn.Module, targets_names: List[str], num_bits: int
) -> NoReturn:
    """
    Applying affine transform to layers

    Args:
        model: model for transform
        targets_names: names of layers for transform
        num_bits: quantity of bits for quantization

    IMPORTANT: num_bits must be 8 or 4
    """

    if not (num_bits != 8 or num_bits != 4):
        raise QuantizationError(
            "Affine transform quantization supports only int8 and int4 quantization"
        )

    for name in targets_names:
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)

        layer = getattr(parent, parts[-1])

        if num_bits == 8:
            source = LinearAffine8bit(
                weight=layer.weight.detach(),
                bias=(
                    layer.bias.detach()
                    if hasattr(layer, "bias") and layer.bias is not None
                    else None
                ),
            )

        elif num_bits == 4:
            source = Linear4bit(
                input_features=layer.in_features,
                output_features=layer.out_features,
                bias=hasattr(layer, "bias") and layer.bias is not None,
            )

            source.weight = Params4bit(data=layer.weight, requires_grad=False)

            if source.bias:
                source.bias = nn.Parameter(layer.bias)

        setattr(
            parent,
            parts[-1],
            source,
        )

    model.to(model.device)
    gc.collect()


if __name__ == "__main__":
    from ...layers_worker import get_names_from_type

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch import nn
    import torch
    import time
    import gc

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2-xl", trust_remote_code=True
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        "openai-community/gpt2-xl", trust_remote_code=True
    )
    inputs = tokenizer(
        "How to write wifi sniffer using python? Answer: ", return_tensors="pt"
    ).to(device)

    torch.cuda.reset_peak_memory_stats()
    start = time()
    outputs = model.generate(**inputs, max_new_tokens=24)
    end = time()
    peak_memory = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    affine_transform(
        model=model, targets_names=get_names_from_type(model, nn.Linear), num_bits=4
    )
    torch.cuda.empty_cache()
    gc.collect()

    torch.cuda.reset_peak_memory_stats()
    q_start = time()
    q_outputs = model.generate(**inputs, max_new_tokens=24)
    q_peak_memory = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    q_end = time()

    print("Model time:", round((end - start) * 1000), "ms")
    print("QModel time:", round((q_end - q_start) * 1000), "ms")
    print("Model peak memory:", peak_memory, "MB")
    print("QModel peak memory:", q_peak_memory, "MB")
    print("--------------------------------------------")
    print(tokenizer.decode(q_outputs[0], skip_special_tokens=True))
    print("--------------------------------------------")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
