from pycocoevalcap.spice.spice import Spice
from typing import List


def compute_spice(
    *,
    predictions: List[str],
    references: List[List[str]],  # Несколько референсов на пример
    **kwargs
) -> float:
    """Вычисляет SPICE score для семантического соответствия."""
    spice = Spice()
    preds = {i: [p] for i, p in enumerate(predictions)}
    refs = {i: rs for i, rs in enumerate(references)}
    score, _ = spice.compute_score(refs, preds)
    return score