from typing import List, Dict, Optional
from evaluate import load
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_glue(
        *,
        predictions: List[str],
        references: List[str],
        task_name: Optional[str] = None,
        **kwargs
) -> Dict[str, float]:
    """Вычисляет метрики для GLUE бенчмарка."""
    try:
        if not task_name or "glue" not in task_name.lower():
            logger.warning("GLUE: Отсутствует или неподдерживаемое имя задачи")
            return {"glue_metric": float("inf")}

        if not predictions or not references or len(predictions) != len(references):
            logger.warning("GLUE: Пустые или несоответствующие списки predictions или references")
            return {"glue_metric": float("inf")}

        if not all(isinstance(p, str) for p in predictions) or not all(isinstance(r, str) for r in references):
            logger.warning("GLUE: Некорректный формат predictions или references")
            return {"glue_metric": float("inf")}

        task_metrics = {
            "cola": ("matthews_correlation", load("matthews_correlation")),
            "sst2": ("accuracy", load("accuracy")),
            "mrpc": ("f1", load("f1")),
            "qqp": ("f1", load("f1")),
            "stsb": ("pearsonr", load("pearsonr")),
            "mnli": ("accuracy", load("accuracy")),
            "qnli": ("accuracy", load("accuracy")),
            "rte": ("accuracy", load("accuracy"))
        }

        task_key = task_name.split("_")[-1]
        if task_key not in task_metrics:
            logger.warning(f"GLUE: Неподдерживаемая задача {task_key}")
            return {"glue_metric": float("inf")}

        metric_name, metric = task_metrics[task_key]
        results = metric.compute(predictions=predictions, references=references)
        return {metric_name: results[metric_name] if np.isfinite(results[metric_name]) else float("inf")}
    except Exception as e:
        logger.warning(f"Ошибка в compute_glue: {str(e)}")
        return {"glue_metric": float("inf")}