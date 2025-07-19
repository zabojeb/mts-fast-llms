import torch
from typing import List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_clip_score_vision(
        *,
        predictions: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        text_features: Optional[torch.Tensor],
        device: str = "cuda",
        **kwargs
) -> float:
    """Вычисляет CLIPScore для vision-задач (OwlV2) на основе эмбеддингов объектов."""
    try:
        if not predictions or not text_features or not any(p[2] is not None for p in predictions):
            logger.warning("CLIPScoreVision: Пустые предсказания или отсутствуют text_features")
            return float("inf")

        if not all(isinstance(p, tuple) and len(p) == 3 for p in predictions):
            logger.warning("CLIPScoreVision: Некорректный формат predictions")
            return float("inf")

        text_features = text_features.to(device)
        scores = []
        for _, _, object_embeds in predictions:
            if object_embeds is None:
                continue
            object_embeds = object_embeds.to(device)
            score = torch.cosine_similarity(object_embeds, text_features, dim=-1)
            scores.append(score.mean().item())

        return float(torch.tensor(scores).mean().item()) if scores else float("inf")
    except Exception as e:
        logger.warning(f"Ошибка в compute_clip_score_vision: {str(e)}")
        return float("inf")