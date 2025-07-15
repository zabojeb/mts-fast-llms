from typing import List, Any


def compute_accuracy(
    predictions: List[Any],
    references: List[Any],
    **kwargs
) -> float:
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / len(predictions) if predictions else 0.0