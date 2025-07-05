from bert_score import score as bert_score
from typing import List, Dict


def compute_bert_score(
    *,
    predictions: List[str],
    references: List[str],
    lang: str = "en",
    model_type: str = "bert-base-uncased",
    **kwargs
) -> Dict[str, float]:
    """Вычисляет BERTScore (precision, recall, F1)."""
    P, R, F1 = bert_score(predictions, references, lang=lang, model_type=model_type)
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item()
    }