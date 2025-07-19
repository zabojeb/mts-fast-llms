# CIDEr.py
from pycocoevalcap.cider.cider import Cider
from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)
CIDER_CACHE = None


def compute_cider(
        *,
        predictions: List[str],
        references: List[List[str]],
        **kwargs
) -> float:
    """Вычисляет CIDEr score с кэшированием объекта Cider."""
    global CIDER_CACHE

    try:
        if not predictions or not references or len(predictions) != len(references):
            logger.warning("CIDEr: Пустые или несоответствующие списки предсказаний и референсов")
            return float("inf")

        if not all(isinstance(p, str) for p in predictions) or not all(
                isinstance(r, list) and all(isinstance(ref, str) for ref in r) for r in references
        ):
            logger.warning("CIDEr: Некорректный формат предсказаний или референсов")
            return float("inf")

        # Инициализация кэша при первом вызове
        if CIDER_CACHE is None:
            logger.info("Инициализация CIDEr кэша...")
            CIDER_CACHE = Cider()

        preds = {i: [p] for i, p in enumerate(predictions)}
        refs = {i: rs for i, rs in enumerate(references)}
        score, _ = CIDER_CACHE.compute_score(refs, preds)
        return score if np.isfinite(score) else float("inf")
    except Exception as e:
        logger.warning(f"Ошибка в compute_cider: {str(e)}")
        return float("inf")