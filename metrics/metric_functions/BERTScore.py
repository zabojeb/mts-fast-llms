from bert_score import score as bert_score
from typing import List, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_bert_score(
        *,
        predictions: List[str],
        references: List[List[str]],
        lang: str = "en",
        model_type: str = "bert-base-uncased",
        **kwargs
) -> Dict[str, float]:
    """Вычисляет BERTScore (precision, recall, F1)."""
    try:
        if not predictions or not references or len(predictions) != len(references):
            logger.warning("BERTScore: Пустые или несоответствующие списки предсказаний и референсов")
            return {"bertscore_precision": float("inf"), "bertscore_recall": float("inf"), "bertscore_f1": float("inf")}

        if not all(isinstance(p, str) for p in predictions) or not all(
                isinstance(r, list) and all(isinstance(ref, str) for ref in r) for r in references
        ):
            logger.warning("BERTScore: Некорректный формат предсказаний или референсов")
            return {"bertscore_precision": float("inf"), "bertscore_recall": float("inf"), "bertscore_f1": float("inf")}

        flat_references = [refs[0] for refs in references if refs]
        if not flat_references:
            logger.warning("BERTScore: Нет валидных референсов")
            return {"bertscore_precision": float("inf"), "bertscore_recall": float("inf"), "bertscore_f1": float("inf")}

        P, R, F1 = bert_score([p for p in predictions if p.strip()], flat_references, lang=lang, model_type=model_type)
        return {
            "bertscore_precision": P.mean().item() if np.isfinite(P.mean()) else float("inf"),
            "bertscore_recall": R.mean().item() if np.isfinite(R.mean()) else float("inf"),
            "bertscore_f1": F1.mean().item() if np.isfinite(F1.mean()) else float("inf")
        }
    except Exception as e:
        logger.warning(f"Ошибка в compute_bert_score: {str(e)}")
        return {"bertscore_precision": float("inf"), "bertscore_recall": float("inf"), "bertscore_f1": float("inf")}