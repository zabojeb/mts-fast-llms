from typing import Any
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_perplexity(**kwargs) -> float:
    log_probs = kwargs.get('log_probs', [])
    total_tokens = kwargs.get('total_tokens', 0)
    if not log_probs or total_tokens == 0:
        logger.warning("log_probs пуст или total_tokens=0, возвращается inf")
        return float("inf")
    avg_log_prob = sum(log_probs) / total_tokens
    perplexity = np.exp(-avg_log_prob)
    logger.info(f"Perplexity: avg_log_prob={avg_log_prob:.4f}, total_tokens={total_tokens}, perplexity={perplexity:.4f}")
    return perplexity