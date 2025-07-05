from typing import List, Dict, Optional


def compute_mmlu(
        *,
        predictions: List[str],
        references: List[str],
        task_name: Optional[str] = None,
        **kwargs
) -> Optional[float]:
    """Вычисляет точность на MMLU бенчмарке."""
    if not task_name or "mmlu" not in task_name.lower():
        return None

    correct = sum(1 for p, r in zip(predictions, references) if p.lower() == r.lower())
    return correct / len(predictions) if predictions else 0.0