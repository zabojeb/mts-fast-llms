from typing import List, Dict, Optional


def compute_helm(
        *,
        predictions: List[str],
        references: List[str],
        task_name: Optional[str] = None,
        **kwargs
) -> Optional[Dict[str, float]]:
    """Вычисляет метрики для HELM бенчмарка."""
    if not task_name or "helm" not in task_name.lower():
        return None

    return {
        "accuracy": sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions),
        "exact_match": sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip()) / len(predictions)
    }