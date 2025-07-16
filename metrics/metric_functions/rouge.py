from rouge_score import rouge_scorer
from typing import List, Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_rouge(
        *,
        predictions: List[str],
        references: List[List[str]],
        rouge_types: Optional[List[str]] = None,
        **kwargs
) -> Dict[str, float]:
    """Вычисляет ROUGE метрики (только F1) для списка предсказаний и референсов."""
    if not rouge_types:
        rouge_types = ["rouge1", "rouge2", "rougeL"]

    try:
        if not predictions or not references or len(predictions) != len(references):
            logger.warning("ROUGE: Пустые или несоответствующие списки предсказаний и референсов")
            return {rouge_type: float("inf") for rouge_type in rouge_types}

        if not all(isinstance(p, str) for p in predictions) or not all(
                isinstance(r, list) and all(isinstance(ref, str) for ref in r) for r in references
        ):
            logger.warning("ROUGE: Некорректный формат предсказаний или референсов")
            return {rouge_type: float("inf") for rouge_type in rouge_types}

        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        aggregated = {rouge_type: [] for rouge_type in rouge_types}

        for pred, refs in zip(predictions, references):
            if not refs or not pred.strip():
                continue
            best_scores = {rouge_type: 0.0 for rouge_type in rouge_types}
            for ref in refs:
                if not ref.strip():
                    continue
                scores = scorer.score(ref, pred)
                for rouge_type in rouge_types:
                    best_scores[rouge_type] = max(best_scores[rouge_type], scores[rouge_type].fmeasure)
            for rouge_type in rouge_types:
                aggregated[rouge_type].append(best_scores[rouge_type])

        if not any(aggregated[rouge_type] for rouge_type in rouge_types):
            logger.warning("ROUGE: Не удалось вычислить метрики из-за пустых данных")
            return {rouge_type: float("inf") for rouge_type in rouge_types}

        return {
            rouge_type: sum(scores) / len(scores) if scores else float("inf")
            for rouge_type, scores in aggregated.items()
        }
    except Exception as e:
        logger.warning(f"ROUGE: Ошибка при вычислении: {str(e)}")
        return {rouge_type: float("inf") for rouge_type in rouge_types}