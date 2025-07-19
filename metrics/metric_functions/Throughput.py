from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_throughput(**kwargs) -> float:
    timestamps = kwargs.get('timestamps')
    total_examples = kwargs.get('total_examples')
    if timestamps and len(timestamps) >= 2 and total_examples:
        duration = timestamps[1] - timestamps[0]
        if duration > 0:
            return total_examples / duration
    return float("inf")