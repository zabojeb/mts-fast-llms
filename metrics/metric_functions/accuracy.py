from typing import List, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_accuracy(
        *,
        predictions: List[Any],
        labels: List[Any],
        **kwargs
) -> float:
    """Вычисляет точность (accuracy) для списка предсказаний и меток."""
    try:
        if not predictions or not labels or len(predictions) != len(labels):
            logger.warning("Accuracy: Пустые или несоответствующие списки предсказаний и меток")
            return float("inf")

        correct = sum(1 for p, r in zip(predictions, labels) if p == r)
        return correct / len(predictions) if predictions else float("inf")
    except Exception as e:
        logger.warning(f"Ошибка в compute_accuracy: {str(e)}")
        return float("inf")