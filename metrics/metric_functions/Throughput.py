from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_throughput(**kwargs) -> float:
    duration = kwargs.get('duration')
    total_examples = kwargs.get('total_examples')

    if not duration or not total_examples:
        return float("inf")

    return total_examples / duration