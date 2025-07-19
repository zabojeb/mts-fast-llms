from typing import List, Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_helm(
        *,
        predictions: List[str],
        references: List[str],
        task_name: Optional[str] = None,
        **kwargs
) -> Dict[str, float]:
    """Вычисляет метрики для HELM бенчмарка."""
    try:
        if not task_name or "helm" not in task_name.lower():
            logger.warning("HELM: Отсутствует или неподдерживаемое имя задачи")
            return {"accuracy": float("inf"), "exact_match": float("inf")}

        if not predictions or not references or len(predictions) != len(references):
            logger.warning("HELM: Пустые или несоответствующие списки predictions или references")
            return {"accuracy": float("inf"), "exact_match": float("inf")}

        if not all(isinstance(p, str) for p in predictions) or not all(isinstance(r, str) for r in references):
            logger.warning("HELM: Некорректный формат predictions или references")
            return {"accuracy": float("inf"), "exact_match": float("inf")}

        accuracy = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
        exact_match = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip()) / len(predictions)
        return {
            "accuracy": accuracy if np.isfinite(accuracy) else float("inf"),
            "exact_match": exact_match if np.isfinite(exact_match) else float("inf")
        }
    except Exception as e:
        logger.warning(f"Ошибка в compute_helm: {str(e)}")
        return {"accuracy": float("inf"), "exact_match": float("inf")}