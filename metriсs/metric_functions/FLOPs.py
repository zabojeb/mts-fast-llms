from fvcore.nn import FlopCountAnalysis
import torch
from typing import List, Any


def compute_flops(
    *,
    model: torch.nn.Module,
    text: List[str],
    tokenizer: Any,
    device: str = "cuda",
    **kwargs
) -> float:
    """Вычисляет количество операций для обработки текста."""
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    flops = FlopCountAnalysis(model, inputs["input_ids"])
    return flops.total()