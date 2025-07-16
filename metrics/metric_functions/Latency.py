from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_latency(
        *,
        timestamps: List[float],
        **kwargs
) -> float:
    """Вычисляет задержку в секундах."""
    try:
        if not timestamps or len(timestamps) < 2 or not all(isinstance(t, (int, float)) for t in timestamps):
            logger.warning("Latency: Некорректный формат timestamps")
            return float("inf")

        latency = timestamps[-1] - timestamps[0]
        return latency if np.isfinite(latency) else float("inf")
    except Exception as e:
        logger.warning(f"Ошибка в compute_latency: {str(e)}")
        return float("inf")