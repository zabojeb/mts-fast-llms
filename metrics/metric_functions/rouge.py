from rouge_score import rouge_scorer
from typing import List, Dict, Optional

def compute_rouge(
        *,
        predictions: List[str],
        references: List[List[str]],
        rouge_types: Optional[List[str]] = None,
        **kwargs
) -> Dict[str, float]:
    """Вычисляет ROUGE метрики (только F1) для списка предсказаний и референсов."""
    if not rouge_types:
        rouge_types = ["rouge1", "rouge2", "rougeL"]

    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    aggregated = {rouge_type: 0.0 for rouge_type in rouge_types}

    for pred, refs in zip(predictions, references):
        # Берём первый референс из списка, если refs — список
        ref = refs[0] if isinstance(refs, list) and refs else ""
        scores = scorer.score(ref, pred if pred else "")
        for rouge_type in rouge_types:
            aggregated[rouge_type] += scores[rouge_type].fmeasure

    return {k: v / len(predictions) for k, v in aggregated.items()} if predictions else {}
