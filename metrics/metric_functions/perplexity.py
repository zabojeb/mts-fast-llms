from typing import Any
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_perplexity(**kwargs) -> float:
    loss = kwargs.get('loss')
    if loss is None or not np.isfinite(loss):
        logger.warning("Perplexity: Некорректное значение loss")
        return float("inf")
    return np.exp(loss)