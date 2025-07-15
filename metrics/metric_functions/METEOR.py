from nltk.translate.meteor_score import meteor_score
from typing import List
import nltk
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_meteor(
        *,
        predictions: List[str],
        references: List[List[str]],
        **kwargs
) -> float:
    """Вычисляет METEOR score для списка предсказаний и референсов.

    Note:
        Загружает WordNet и OMW-1.4 для семантической оценки слов.
        WordNet — лексическая база данных для английского языка.
        OMW-1.4 — многоязычная поддержка WordNet.
    """
    try:
        # Загрузка необходимых ресурсов для METEOR
        nltk.download('wordnet', quiet=True)  # Лексическая база для семантического анализа
        nltk.download('omw-1.4', quiet=True)  # Многоязычная поддержка WordNet

        if not predictions or not references or len(predictions) != len(references):
            logger.warning("METEOR: Пустые или несоответствующие списки предсказаний и референсов")
            return float("inf")

        if not all(isinstance(p, str) for p in predictions) or not all(
                isinstance(r, list) and all(isinstance(ref, str) for ref in r) for r in references
        ):
            logger.warning("METEOR: Некорректный формат предсказаний или референсов")
            return float("inf")

        scores = []
        for pred, refs in zip(predictions, references):
            if not refs or not pred.strip():
                continue
            ref_scores = [meteor_score([ref.split()], pred.split()) for ref in refs if ref.strip()]
            scores.append(max(ref_scores) if ref_scores else 0.0)

        return sum(scores) / len(scores) if scores else float("inf")

    except Exception as e:
        logger.warning(f"Ошибка в compute_meteor: {str(e)}")
        return float("inf")