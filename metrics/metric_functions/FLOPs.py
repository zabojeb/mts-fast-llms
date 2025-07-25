from fvcore.nn import FlopCountAnalysis
import torch
from typing import Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_flops_in_train(
        model: torch.nn.Module,
        processor: Any,
        raw_text: Any,
        images: Optional[torch.Tensor] = None,
        task_name: str = "",
        device: str = "cuda"
) -> float:
    """Вычисляет FLOPs для обработки входных данных внутри train_func."""
    try:
        if task_name == "vision":
            if images is None or processor is None:
                logger.warning("FLOPs: Отсутствуют изображения или процессор для визуальной задачи")
                return float("inf")
            inputs = processor(text=raw_text, images=images, return_tensors="pt", padding=True, truncation=True).to(device)
            flops = FlopCountAnalysis(model, (inputs["input_ids"], inputs["pixel_values"]))
        else:
            if not raw_text or not processor:
                logger.warning("FLOPs: Отсутствует текст или процессор")
                return float("inf")
            inputs = processor(raw_text, return_tensors="pt", padding=True, truncation=True).to(device)
            flops = FlopCountAnalysis(model, inputs["input_ids"])
        return flops.total() if np.isfinite(flops.total()) else float("inf")
    except Exception as e:
        logger.warning(f"FLOPs: Ошибка при вычислении: {str(e)}")
        return float("inf")