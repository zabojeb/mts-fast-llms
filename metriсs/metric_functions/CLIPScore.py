import torch
from typing import List


def compute_clip_score(
        *,
        image_features: torch.Tensor,
        text: List[str],
        model: torch.nn.Module,
        device: str = "cuda",
        **kwargs
) -> float:
    """Вычисляет CLIPScore между изображениями и текстом."""
    if image_features is None:
        return None

    text_features = model.encode_text(text).to(device)
    image_features = image_features.to(device)
    scores = torch.cosine_similarity(image_features, text_features)
    return scores.mean().item()