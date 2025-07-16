import torch
import torch.nn.functional as F
from typing import List, Any

def compute_perplexity(
        *,
        model: torch.nn.Module,
        text: List[str],
        tokenizer: Any,
        device: str = "cuda",
        **kwargs
) -> float:
    """Вычисляет perplexity для текста на заданной модели."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    if not text:
        return 0.0  # Возвращаем 0.0 для пустого ввода

    with torch.no_grad():
        for sentence in text:
            if not sentence:  # Пропускаем пустые строки
                continue
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            if loss is not None:
                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)

    if total_tokens == 0:
        return 0.0  # Возвращаем 0.0, если нет токенов

    avg_loss = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_loss)).item()
