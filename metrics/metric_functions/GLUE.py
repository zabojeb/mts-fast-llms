from typing import List, Dict, Optional
from evaluate import load  # Новый правильный импорт


def compute_glue(
        *,
        predictions: List[str],
        references: List[str],
        task_name: Optional[str] = None,
        **kwargs
) -> Optional[Dict[str, float]]:
    """Вычисляет метрики для GLUE бенчмарка."""
    if not task_name or "glue" not in task_name.lower():
        return None

    # Определяем соответствие задач и метрик
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
        return None

    metric_name, metric = task_metrics[task_key]
    results = metric.compute(predictions=predictions, references=references)

    return {metric_name: results[metric_name]}