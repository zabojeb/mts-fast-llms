from rouge_score import rouge_scorer
from typing import List, Dict, Optional

from rouge_score import rouge_scorer


def compute_rouge(
        *,
        predictions: List[str],
        references: List[str],
        rouge_types: Optional[List[str]] = None,
        **kwargs
) -> Dict[str, float]:
    """Вычисляет ROUGE метрики (только F1)."""
    if not rouge_types:
        rouge_types = ["rouge1", "rouge2", "rougeL"]

    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    aggregated = {rouge_type: 0.0 for rouge_type in rouge_types}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for rouge_type in rouge_types:
            aggregated[rouge_type] += scores[rouge_type].fmeasure

    return {k: v / len(predictions) for k, v in aggregated.items()}