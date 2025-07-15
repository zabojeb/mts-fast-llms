from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_throughput(
        *,
        timestamps: List[float],
        batch_size: int,
        **kwargs
) -> float:
    """Вычисляет throughput в примерах/секунду."""
    try:
        if not timestamps or len(timestamps) < 2 or not all(isinstance(t, (int, float)) for t in timestamps):
            logger.warning("Throughput: Некорректный формат timestamps")
            return float("inf")

        if not isinstance(batch_size, int) or batch_size <= 0:
            logger.warning("Throughput: Некорректное значение batch_size")
            return float("inf")

        duration = timestamps[-1] - timestamps[0]
        return batch_size / duration if duration > 0 else float("inf")
    except Exception as e:
        logger.warning(f"Ошибка в compute_throughput: {str(e)}")
        return float("inf")