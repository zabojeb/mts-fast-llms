from nltk.translate.bleu_score import corpus_bleu
from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_bleu(
        *,
        predictions: List[str],
        references: List[List[str]],
        **kwargs
) -> float:
    """Вычисляет BLEU score для списка предсказаний и референсов."""
    try:
        if not predictions or not references or len(predictions) != len(references):
            logger.warning("BLEU: Пустые или несоответствующие списки предсказаний и референсов")
            return float("inf")

        if not all(isinstance(p, str) for p in predictions) or not all(
                isinstance(r, list) and all(isinstance(ref, str) for ref in r) for r in references
        ):
            logger.warning("BLEU: Некорректный формат предсказаний или референсов")
            return float("inf")

        formatted_references = [[ref.split() for ref in refs if ref.strip()] for refs in references]
        formatted_predictions = [pred.split() for pred in predictions if pred.strip()]

        if not formatted_references or not formatted_predictions:
            logger.warning("BLEU: Нет валидных данных после фильтрации")
            return float("inf")

        score = corpus_bleu(formatted_references, formatted_predictions)
        return score if np.isfinite(score) else float("inf")
    except Exception as e:
        logger.warning(f"Ошибка в compute_bleu: {str(e)}")
        return float("inf")