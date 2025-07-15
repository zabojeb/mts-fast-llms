from typing import Any
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_perplexity(
        *,
        loss: Any,
        task_name: str,
        **kwargs
) -> float:
    """Вычисляет перплексию на основе функции потерь.

    Args:
        loss: Значение функции потерь (float).
        task_name: Имя задачи (для проверки применимости).
        **kwargs: Дополнительные параметры (игнорируются).

    Returns:
        float: Значение перплексии или inf, если значение некорректно.
    """
    if task_name not in ["generation", "translation"]:
        logger.warning("Perplexity: Метрика применима только к задачам GENERATION и TRANSLATION")
        return float("inf")

    if not isinstance(loss, (int, float)) or not np.isfinite(loss):
        logger.warning("Perplexity: Некорректное значение loss")
        return float("inf")

    try:
        perplexity = torch.exp(torch.tensor(loss)).item()
        return perplexity if np.isfinite(perplexity) else float("inf")
    except Exception as e:
        logger.warning(f"Perplexity: Ошибка при вычислении: {str(e)}")
        return float("inf")