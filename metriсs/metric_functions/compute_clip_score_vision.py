import torch
from typing import List, Tuple, Optional

def compute_clip_score_vision(
    *,
    predictions: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],  # [(boxes, scores, object_embeds), ...]
    text: List[str],
    model: torch.nn.Module,
    device: str = "cuda",
    processor: Any = None,
    **kwargs
) -> float:
    """Вычисляет CLIPScore для vision-задач (OwlV2) на основе эмбеддингов объектов."""
    if not predictions or not any(p[2] is not None for p in predictions):
        return 0.0

    # Получаем текстовые эмбеддинги
    text_features = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
    text_features = model.get_text_features(**text_features)

    scores = []
    for _, _, object_embeds in predictions:
        if object_embeds is None:
            continue
        object_embeds = object_embeds.to(device)
        # Косинусное сходство между эмбеддингами объектов и текстом
        score = torch.cosine_similarity(object_embeds, text_features, dim=-1)
        scores.append(score.mean().item())

    return float(torch.tensor(scores).mean().item()) if scores else 0.0