# SPICE.py
from pycocoevalcap.spice.spice import Spice
from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)
SPICE_CACHE = None


def compute_spice(
        *,
        predictions: List[str],
        references: List[List[str]],
        **kwargs
) -> float:
    """Вычисляет SPICE score с кэшированием объекта Spice."""
    global SPICE_CACHE

    try:
        if not predictions or not references or len(predictions) != len(references):
            logger.warning("SPICE: Пустые или несоответствующие списки предсказаний и референсов")
            return float("inf")

        if not all(isinstance(p, str) for p in predictions) or not all(
                isinstance(r, list) and all(isinstance(ref, str) for ref in r) for r in references
        ):
            logger.warning("SPICE: Некорректный формат предсказаний или референсов")
            return float("inf")

        # Инициализация кэша при первом вызове
        if SPICE_CACHE is None:
            logger.info("Инициализация SPICE кэша...")
            SPICE_CACHE = Spice()

        preds = {i: [p] for i, p in enumerate(predictions)}
        refs = {i: rs for i, rs in enumerate(references)}
        score, _ = SPICE_CACHE.compute_score(refs, preds)
        return score if np.isfinite(score) else float("inf")
    except Exception as e:
        logger.warning(f"Ошибка в compute_spice: {str(e)}")
        return float("inf")