from typing import List, Optional
import numpy as np
from sklearn.calibration import calibration_curve
import logging

logger = logging.getLogger(__name__)

def compute_mce(
        *,
        predictions: List[str],
        references: List[str],
        confidences: Optional[List[float]] = None,
        n_bins: int = 10,
        **kwargs
) -> float:
    """Вычисляет Maximum Calibration Error."""
    try:
        if confidences is None or not predictions or not references or len(predictions) != len(references):
            logger.warning("MCE: Пустые или несоответствующие списки predictions, references или confidences")
            return float("inf")

        if not all(isinstance(p, str) for p in predictions) or not all(isinstance(r, str) for r in references) or not all(isinstance(c, (int, float)) for c in confidences):
            logger.warning("MCE: Некорректный формат predictions, references или confidences")
            return float("inf")

        y_true = [1 if p == r else 0 for p, r in zip(predictions, references)]
        prob_true, prob_pred = calibration_curve(y_true, confidences, n_bins=n_bins)
        return np.max(np.abs(prob_true - prob_pred)) if np.isfinite(np.max(np.abs(prob_true - prob_pred))) else float("inf")
    except Exception as e:
        logger.warning(f"Ошибка в compute_mce: {str(e)}")
        return float("inf")