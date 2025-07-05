import numpy as np
from torch import Tensor
import torch
import seaborn as sns
from codecarbon import EmissionsTracker
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
import time
import psutil

#for decorator
from functools import wraps

# metric functions import
from .metric_functions import *


base_metrics = {
            "accuracy": compute_accuracy,
            "bleu": compute_bleu,
            "rouge": compute_rouge,
            "perplexity": compute_perplexity,
            "bert_score": compute_bert_score,
            "meteor": compute_meteor,
            "clip_score": compute_clip_score,
            "cider": CIDEr,
            "spice": compute_spice,
            "latency": compute_latency,
            "memory": compute_memory,
            "flops": compute_flops,
            "throughput": compute_throughput,
            "energy": compute_energy,
            "ece": compute_ece,
            "mce": compute_mce,
            "mmlu": compute_mmlu,
            "helm": compute_helm,
            "glue": compute_glue
        }

        # Сравнительные метрики (две модели)
        # comparative_metrics = {
        #     "kl_divergence": compute_kl_div,
        #     "wasserstein": compute_wasserstein,
        #     "js_divergence": compute_js_div,
        #     "grad_norm": compute_grad_norm,
        #     "grad_variance": compute_grad_variance,
        #     "relative_improvement": compute_relative_improvement
        # }



def collect_metrics_data(
    model: torch.nn.Module,
    text: Optional[List[str]] = None,
    images: Optional[List] = None,
    references: Optional[List[Any]] = None,
    task_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    device: str = "cuda",
    **fixed_kwargs
    ):
    """
    Декоратор, который оборачивает `train_func`, обеспечивая входными данными и собирает все данные для метрик.
    Жестко фиксирует все переданные параметры (model, text, images и др.)
    Возвращает:
    - predictions (результат `train_func`),
    - metrics_data (словарь с сырыми данными для метрик).
    """
    def decorator(train_func: Callable) -> Callable:
        @wraps(train_func)
        def wrapper() -> Tuple[Any, Dict[str, Any]]:  # Никаких параметров при вызове!
            # === 1. Подготовка всех данных ===
            metrics_data = {
                # Основные данные (фиксированные)
                "model": model,
                "text": text,
                "images": images,
                "references": references,
                "device": device,
                "batch_size": batch_size,
                "task_name": task_name,

                # Производительность (будет заполнено)
                "timestamps": [time.time()],  # Старт времени
                "gpu_memory": None,
                "cpu_memory": None,

                # Специфичные данные (будет заполнено)
                "image_features": None,
                "confidences": None,
                **fixed_kwargs
            }

            carbon_tracker = EmissionsTracker()
            carbon_tracker.start()

            # === 2. Замер памяти (GPU) ===
            if device == "cuda":
                torch.cuda.synchronize()
                metrics_data["gpu_memory"] = [torch.cuda.memory_allocated() / (1024 ** 2)]  # В МБ

            # === 3. Замер памяти (CPU) ===
            metrics_data["cpu_memory"] = psutil.Process().memory_info().rss / (1024 ** 2)

            # === 4. Вызов оригинальной функции ===
            with torch.no_grad():
                predictions = train_func()

            # === 5. Фиксация данных после выполнения ===
            metrics_data["timestamps"].append(time.time())
            metrics_data["predictions"] = predictions

            # Обновляем память GPU
            if device == "cuda":
                torch.cuda.synchronize()
                metrics_data["gpu_memory"].append(torch.cuda.memory_allocated() / (1024 ** 2))

            metrics_data["emissions"] = carbon_tracker.stop()

            # === 6. Специфичные данные (если доступны) ===
            # Конфиденсы модели (для ECE/MCE)
            if hasattr(model, "predict_proba") and text is not None:
                metrics_data["confidences"] = model.predict_proba(text)

            # Эмбеддинги изображений (для CLIPScore)
            if images is not None and hasattr(model, "encode_image"):
                metrics_data["image_features"] = [model.encode_image(img) for img in images]

            return predictions, metrics_data
        return wrapper
    return decorator


"""
Model running place
"""

@collect_metrics_data(
    model=...,
    text=["Explain quantum physics", "What is AI?"],
    references=["Study of subatomic...", "Artificial Intelligence..."],
    task_name="science_qa",
    device="cuda"
)
def train_function():
    return train_function.model.generate(train_function.text)

"""
Model running place
"""

def metrics_computing (train_function: Callable[[], Tensor], references, metrics: List[str]):
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)

    predictions, metrics_data = train_function()

    results = {metric: [] for metric in metrics}


    for metric in metrics:
        try:
            metric_func = base_metrics[metric]
            metric_value = metric_func(
                # Основные данные
                model=metrics_data['model'],
                text=metrics_data['text'],
                images=metrics_data['images'],
                references=metrics_data['references'],
                device=metrics_data['device'],
                batch_size=metrics_data['batch_size'],
                task_name=metrics_data['task_name'],

                # Производительность
                timestamps=metrics_data['timestamps'],
                gpu_memory=metrics_data['gpu_memory'],
                cpu_memory=metrics_data['cpu_memory'],
                emissions=metrics_data['emissions'],

                # Специфичные данные
                image_features=metrics_data['image_features'],
                confidences=metrics_data['confidences'],

                # Дополнительные параметры
                **{k: v for k, v in metrics_data.items()
                   if k not in {
                       'model', 'text', 'images', 'references',
                       'device', 'batch_size', 'task_name',
                       'timestamps', 'gpu_memory', 'cpu_memory',
                       'emissions', 'image_features', 'confidences'
                   }}
            )
            results[metric].append(metric_value)
        except Exception as e:
            print(f"Error computing {metric}: {str(e)}")
            results[metric].append(None)
            


    # Агрегация результатов
    return {metric: np.nanmean(values) for metric, values in results.items()}

