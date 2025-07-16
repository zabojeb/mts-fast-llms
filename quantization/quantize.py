<<<<<<< HEAD
# Module done on 16.07 in 11:59

=======
>>>>>>> a523bfc160e57732f7fa65d0fad6e9cf377873d2
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

    if config.num_bits is not None:
<<<<<<< HEAD
        affine_transform(
            model=model, targets_names=config.targets_names, num_bits=config.num_bits
        )
=======
        scaled_weights(model=model, num_bits=config.num_bits)
>>>>>>> a523bfc160e57732f7fa65d0fad6e9cf377873d2

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
