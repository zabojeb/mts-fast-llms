import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset
from codecarbon import EmissionsTracker
from collections import defaultdict
from typing import Dict, List, Optional, Any, Set
from pycocoevalcap.cider.cider import Cider


class ModelComparator:
    def __init__(self, tokenizer: Optional[Any] = None, device: str = "cuda"):
        self.tokenizer = tokenizer
        self.device = device
        self.tracker = EmissionsTracker()

        # Настройка Seaborn
        sns.set(style="whitegrid", palette="muted", font_scale=1.2)

        # Словарь требований метрик
        self.metric_requirements = {
            "cider": {"predictions", "references"},
            "bleu": {"predictions", "references"},
            "perplexity": {"tokens"},
            "accuracy": {"pred_classes", "labels"},
            # Добавьте другие метрики по мере необходимости
        }

        # Реализованные метрики
        self.base_metrics = {
            "cider": self._compute_cider,
            # Добавьте другие реализованные метрики здесь
        }

    def evaluate_model(self, model: torch.nn.Module, dataset: Dataset, metrics: List[str]) -> Dict[str, float]:
        """
        Оценка производительности модели на заданных метриках

        Args:
            model: Модель для оценки
            dataset: Датасет для оценки
            metrics: Список метрик для вычисления

        Returns:
            Словарь с результатами метрик
        """
        # Определяем необходимые данные
        required_data: Set[str] = set()
        for metric in metrics:
            if metric in self.metric_requirements:
                required_data |= self.metric_requirements[metric]

        # Инициализация результатов
        results = {metric: [] for metric in metrics}

        # Обработка датасета
        for i in range(len(dataset)):
            cache = {}
            batch = dataset[i]

            # Подготовка необходимых данных
            if "tokens" in required_data and self.tokenizer:
                text = batch["text"] if "text" in batch else batch["caption"]
                cache["tokens"] = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

            if "predictions" in required_data:
                inputs = cache.get("tokens", batch)
                with torch.no_grad():
                    cache["predictions"] = model.generate(**inputs)

            # Вычисление метрик
            for metric in metrics:
                if metric in self.base_metrics:
                    try:
                        value = self.base_metrics[metric](batch, cache)
                        results[metric].append(value)
                    except Exception as e:
                        print(f"Error computing {metric}: {str(e)}")
                        results[metric].append(None)

        # Агрегация результатов
        return {metric: np.nanmean(values) for metric, values in results.items()}

    # Реализация метрики CIDEr
    def _compute_cider(self, batch: Dict, cache: Dict) -> float:
        """Вычисление CIDEr score для генерации подписей"""
        # Проверка наличия необходимых данных
        if "predictions" not in cache or "references" not in batch:
            raise ValueError("Missing data for CIDEr computation")

        # Подготовка данных для CIDEr
        predictions = cache["predictions"]
        if isinstance(predictions, torch.Tensor) and self.tokenizer:
            predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        res = {i: [pred] for i, pred in enumerate(predictions)}
        gts = {i: refs for i, refs in enumerate(batch["references"])}

        # Вычисление метрики
        scorer = Cider()
        score, _ = scorer.compute_score(gts, res)
        return score

    # Другие метрики могут быть добавлены здесь
    def _compute_bleu(self, batch: Dict, cache: Dict) -> float:
        """BLEU score для генерации текста"""
        # Реализация здесь
        pass

    def _compute_perplexity(self, batch: Dict, cache: Dict) -> float:
        """Перплексия языковой модели"""
        # Реализация здесь
        pass

    def _compute_accuracy(self, batch: Dict, cache: Dict) -> float:
        """Точность классификации"""
        # Реализация здесь
        pass