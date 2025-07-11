from typing import List


def compute_latency(
    *,
    timestamps: List[float],
    **kwargs
) -> float:
    """Вычисляет задержку в секундах."""
    return timestamps[-1] - timestamps[0] if timestamps else 0.0