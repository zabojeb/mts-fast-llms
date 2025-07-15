from nltk.translate.bleu_score import corpus_bleu
from typing import List
import logging


def compute_bleu(
        *,
        predictions: List[str],
        references: List[List[str]],
        **kwargs
) -> float:
    """Вычисляет BLEU score для списка предсказаний и референсов."""
    try:
        # Проверяем, что предсказания и референсы не пусты
        if not predictions or not references:
            logging.warning("Предсказания или референсы пусты")
            return 0.0

        # Форматируем референсы: каждый референс — список слов
        formatted_references = [[ref.split() for ref in refs] for refs in references]
        formatted_predictions = [pred.split() for pred in predictions]

        # Вычисляем BLEU
        score = corpus_bleu(formatted_references, formatted_predictions)
        return score
    except Exception as e:
        logging.warning(f"Ошибка в compute_bleu: {str(e)}")
        return 0.0
