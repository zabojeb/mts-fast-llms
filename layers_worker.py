from typing import List, NoReturn
from torch import nn


def get_names_from_type(model: nn.Module, layer_type: nn.Module) -> List[str]:
    """
    Searches for names of layers with certain type and returns them

    Args:
        model: model for searching
        layer_type: type of layers to be found

    Returns:
        names of layers with certain type
    """

    return [
        name for name, layer in model.named_modules() if isinstance(layer, layer_type)
    ]


def replace_layers(
    model: nn.Module, targets_names: List[str], source: nn.Module
) -> NoReturn:
    """
    Replaces layers with names of model to source layer

    IMPORTANT: source layer must have "target_layer" argument

    Args:
        model: model for replacing
        targets_names: names of layers to be replaced
        source: layer that will replace the rest
    """

    for name in targets_names:
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)

        setattr(
            parent,
            parts[-1],
            source,
        )


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    from torch import nn
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", trust_remote_code=True
    ).to(device)

    targets_names = get_names_from_type(model=model, layer_type=nn.Linear)
    print("Targets names: ")

    for name in targets_names:
        print(name)

    replace_layers(model=model, targets_names=targets_names, source=nn.Identity())
    print("Model after replacing layers to Identity: ")
    print(model)
