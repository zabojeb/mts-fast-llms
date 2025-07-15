from nltk.translate.meteor_score import meteor_score
from typing import List
import nltk
import logging


def compute_meteor(
        *,
        predictions: List[str],
        references: List[List[str]],
        **kwargs
) -> float:
    """Вычисляет METEOR score для списка предсказаний и референсов."""
    try:
        # Загрузка необходимых ресурсов NLTK
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

        scores = []
        for pred, refs in zip(predictions, references):
            # Берём первый референс из списка
            ref = refs[0].split() if isinstance(refs, list) and refs else []
            scores.append(meteor_score([ref], pred.split() if pred else []))
        return sum(scores) / len(scores) if scores else 0.0
    except Exception as e:
        logging.warning(f"Ошибка в compute_meteor: {str(e)}")
        return 0.0
