from typing import List


def compute_throughput(
    *,
    timestamps: List[float],
    batch_size: int,
    **kwargs
) -> float:
    """Вычисляет throughput в примерах/секунду."""
    duration = timestamps[-1] - timestamps[0]
    return batch_size / duration if duration > 0 else 0.0