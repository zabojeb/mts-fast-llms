from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List


def compute_bleu(
        *,
        predictions: List[str],
        references: List[str],
        max_order: int = 4,
        **kwargs
) -> float:
    """Вычисляет BLEU score с правильным smoothing."""
    smoothing_fn = SmoothingFunction()  # Создаем экземпляр класса
    weights = [1.0 / max_order] * max_order

    scores = []
    for pred, ref in zip(predictions, references):
        try:
            score = sentence_bleu(
                [ref.split()],
                pred.split(),
                weights=weights,
                smoothing_function=smoothing_fn.method1  # Используем конкретный метод
            )
            scores.append(score)
        except:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0