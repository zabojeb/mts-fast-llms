from torch import nn

from .config import QuantizationConfig
from .methods import *


def quantize(
    model: nn.Module,
    config: QuantizationConfig,
) -> nn.Module:
    """
    Quantizes copy of the model and returns it

    Kwargs:
        model: input model for quantization
        config: quantization config

    Returns:
        q_model: quantized model
    """

    if config.compute_dtype is not None:
        model = weights_to_dtype(model, config.compute_dtype)

    if config.quantize_int8:
        scaled_weights(model)

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
