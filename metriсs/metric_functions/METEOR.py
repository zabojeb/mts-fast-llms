from nltk.translate.meteor_score import meteor_score
from typing import List

def compute_meteor(
    *,
    predictions: List[str],
    references: List[str],
    **kwargs
) -> float:
    """Вычисляет METEOR score."""
    scores = []
    for pred, ref in zip(predictions, references):
        scores.append(meteor_score([ref.split()], pred.split()))
    return sum(scores)/len(scores) if scores else 0.0