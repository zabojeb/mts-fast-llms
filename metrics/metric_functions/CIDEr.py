from pycocoevalcap.cider.cider import Cider
from typing import List

def compute_cider(
    *,
    predictions: List[str],
    references: List[List[str]],  # Для CIDEr нужно несколько референсов на каждый пример
    **kwargs
) -> float:
    """Вычисляет CIDEr score."""
    scorer = Cider()
    # Приводим к формату {id: [caption]}
    preds = {i: [p] for i, p in enumerate(predictions)}
    refs = {i: rs for i, rs in enumerate(references)}
    score, _ = scorer.compute_score(refs, preds)
    return score