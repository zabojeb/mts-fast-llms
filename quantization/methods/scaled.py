from torch import nn
import gc

from ..modules import *


def scaled_weights(model: nn.Module, num_bits: int) -> nn.Module:
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            parent = model

            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)

            setattr(
                parent,
                parts[-1],
                QuantizedLinear(
                    weight=layer.weight.detach(),
                    bias=(
                        layer.bias.detach()
                        if hasattr(layer, "bias") and layer.bias is not None
                        else None
                    ),
                    num_bits=num_bits,
                ),
            )

    gc.collect()


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch import nn
    import torch
    import time
    import gc

    device = "cuda"

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

    scaled_weights(model)
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
