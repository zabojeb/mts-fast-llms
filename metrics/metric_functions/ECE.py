import numpy as np
from sklearn.calibration import calibration_curve
from typing import List, Optional


def compute_ece(
        *,
        predictions: List[str],
        references: List[str],
        confidences: Optional[List[float]] = None,
        n_bins: int = 10,
        **kwargs
) -> Optional[float]:
    """Вычисляет Expected Calibration Error."""
    if confidences is None:
        return None

    y_true = [1 if p == r else 0 for p, r in zip(predictions, references)]
    prob_true, prob_pred = calibration_curve(y_true, confidences, n_bins=n_bins)
    return np.mean(np.abs(prob_true - prob_pred))