import torch
import torch.nn.functional as F
from typing import List, Any
import logging

logger = logging.getLogger(__name__)

def compute_perplexity(
        *,
        model: torch.nn.Module,
        text: List[str],
        tokenizer: Any,
        device: str = "cuda",
        batch_size: int = 8,
        **kwargs
) -> float:
    """Вычисляет perplexity для текста на заданной модели.

    Args:
        model: Модель (например, AutoModelForCausalLM).
        text: Список строк для оценки.
        tokenizer: Токенизатор (например, AutoTokenizer).
        device: Устройство ('cuda' или 'cpu').
        batch_size: Размер батча для обработки текста.

    Returns:
        float: Значение перплексии или inf при ошибке.
    """
    if not text or not all(isinstance(t, str) and t.strip() for t in text):
        logger.warning("Perplexity: Пустой или некорректный текст")
        return float("inf")

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    try:
        # Батчинг для эффективности
        for i in range(0, len(text), batch_size):
            batch_text = text[i:i + batch_size]
            inputs = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True).to(device)
            input_ids = inputs["input_ids"]

            if input_ids.size(0) == 0 or input_ids.size(1) == 0:
                logger.warning("Perplexity: Пустые токенизированные входы")
                continue

            with torch.no_grad():
                outputs = model(**inputs, labels=input_ids)
                if not hasattr(outputs, "loss") or outputs.loss is None:
                    logger.warning("Perplexity: Модель не вернула loss")
                    return float("inf")
                loss = outputs.loss
                total_loss += loss.item() * input_ids.size(0) * input_ids.size(1)
                total_tokens += input_ids.size(0) * input_ids.size(1)

        if total_tokens == 0:
            logger.warning("Perplexity: Нет токенов для обработки")
            return float("inf")

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        if not torch.isfinite(torch.tensor(perplexity)):
            logger.warning(f"Perplexity: Некорректное значение {perplexity}")
            return float("inf")

        return perplexity

    except Exception as e:
        logger.warning(f"Perplexity: Ошибка при вычислении: {str(e)}")
        return float("inf")