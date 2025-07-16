from fvcore.nn import FlopCountAnalysis
import torch
from typing import List, Any, Optional
import logging

def compute_flops(
    *,
    model: torch.nn.Module,
    text: Any,
    images: Optional[torch.Tensor] = None,
    processor: Any = None,
    device: str = "cuda",
    task_name: str,
    **kwargs
) -> float:
    """Вычисляет количество операций для обработки входных данных."""
    try:
        if task_name == "vision":
            if images is None or processor is None:
                return 0.0
            inputs = processor(text=text, images=images, return_tensors="pt", padding=True, truncation=True).to(device)
            flops = FlopCountAnalysis(model, (inputs["input_ids"], inputs["pixel_values"]))
        else:
            inputs = processor(text, return_tensors="pt", padding=True, truncation=True).to(device)
            flops = FlopCountAnalysis(model, inputs["input_ids"])
        return flops.total()
    except Exception as e:
        if "DynamicCache" in str(e):
            logging.warning("FLOPs не поддерживаются для моделей с DynamicCache")
            return None
        raise e
