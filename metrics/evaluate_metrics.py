import numpy as np
import torch
import time
import seaborn as sns
import psutil
from codecarbon import EmissionsTracker
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from transformers import GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer, OwlViTProcessor, OwlViTForObjectDetection
from datasets import Dataset as HFDataset
from enum import Enum

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Импорт функций метрик
from .metric_functions import *

# Типы задач
class TaskType(Enum):
    CLASSIFICATION = "classification"
    TRANSLATION = "translation"
    GENERATION = "generation"
    VISION = "vision"

# Метрики для задач
TASK_METRICS = {
    TaskType.CLASSIFICATION: ["accuracy", "ece", "mce", "mmlu", "glue", "latency", "memory", "flops", "throughput", "energy"],
    TaskType.TRANSLATION: ["bleu", "rouge", "meteor", "bert_score", "perplexity", "latency", "memory", "throughput", "energy"],
    TaskType.GENERATION: ["bleu", "rouge", "meteor", "bert_score", "perplexity", "latency", "memory", "flops", "throughput", "energy"],
    TaskType.VISION: ["iou", "map", "precision", "recall", "f1", "clip_score_vision", "latency", "memory", "flops", "throughput", "energy"]
}

# Базовые метрики
base_metrics = {
    "accuracy": compute_accuracy,
    "bleu": compute_bleu,
    "rouge": compute_rouge,
    "perplexity": compute_perplexity,
    "bert_score": compute_bert_score,
    "meteor": compute_meteor,
    "iou": compute_iou,
    "map": compute_map,
    "precision": compute_precision,
    "recall": compute_recall,
    "f1": compute_f1,
    "cider": compute_cider,
    "spice": compute_spice,
    "clip_score_vision": compute_clip_score_vision,
    "latency": compute_latency,
    "memory": compute_memory,
    "flops": compute_flops_in_train,
    "throughput": compute_throughput,
    "energy": compute_energy,
    "ece": compute_ece,
    "mce": compute_mce,
    "mmlu": compute_mmlu,
    "helm": compute_helm,
    "glue": compute_glue
}

def load_model(model: Any, processor: Any, device: str = "cuda") -> Tuple[Any, Any]:
    """Загружает модель и процессор, перемещая модель на указанное устройство."""
    model = model.to(device)
    return model, processor

def process_in_batches(dataset: HFDataset, batch_size: int) -> List[HFDataset]:
    """Разбивает датасет на батчи, возвращая список Dataset объектов."""
    num_samples = len(dataset)
    return [dataset.select(range(i, min(i + batch_size, num_samples))) for i in range(0, num_samples, batch_size)]

def train_func(
        model: Any,
        processor: Any,
        dataset: HFDataset,
        task_type: TaskType,
        device: str = "cuda",
        batch_size: Optional[int] = None,
        field_mapping: Optional[Dict[str, str]] = None
) -> Tuple[Any, Dict[str, Any], List[str]]:
    model, processor = load_model(model, processor, device)
    field_mapping = field_mapping or {}

    # Извлечение данных
    text_field = field_mapping.get("text", "text")
    reference_field = field_mapping.get("reference", "references")
    text = dataset[text_field]
    references = [[ref] if isinstance(ref, str) else ref for ref in dataset[reference_field]]

    metrics_data = {
        "model": model,
        "raw_text": text,
        "references": references,
        "processor": processor,
        "device": device,
        "batch_size": batch_size if batch_size is not None else len(dataset),
        "task_name": task_type.value,
        "timestamps": [time.time()],
        "gpu_memory": [],
        "cpu_memory": [],
        "emissions": None,
        "predictions": [],
        "log_probs": [],  # Для перплексии
        "total_tokens": 0,
        "total_examples": len(dataset)
    }

    batches = process_in_batches(dataset, batch_size)
    carbon_tracker = EmissionsTracker()
    carbon_tracker.start()

    all_predictions = []

    for batch in batches:
        batch_text = batch[text_field]
        batch_references = [[ref] if isinstance(ref, str) else ref for ref in batch[reference_field]]

        # Токенизация промптов
        inputs = processor(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        metrics_data["inputs"] = inputs
        metrics_data["input_ids"] = inputs["input_ids"]

        # Замеры памяти
        if device == "cuda":
            torch.cuda.synchronize()
            metrics_data["gpu_memory"].append(torch.cuda.memory_allocated() / (1024 ** 2))
        metrics_data["cpu_memory"].append(psutil.Process().memory_info().rss / (1024 ** 2))

        with torch.no_grad(), torch.amp.autocast('cuda'):
            # Генерация предсказаний
            generate_kwargs = field_mapping.get("generate_kwargs",
                                               {"max_new_tokens": 150, "num_beams": 1, "do_sample": False})
            predictions = model.generate(**inputs, **generate_kwargs)
            decoded_predictions = [processor.decode(pred, skip_special_tokens=True).strip() for pred in predictions]
            all_predictions.extend([p if p else "<пусто>" for p in decoded_predictions])

            # Вычисление логарифмов вероятностей для перплексии
            if task_type in [TaskType.TRANSLATION, TaskType.GENERATION]:
                for refs in batch_references:
                    if not refs or not refs[0].strip():
                        logger.warning(f"Пропущен пустой референс: refs={refs}")
                        continue
                    target = refs[0]
                    target_ids = processor(target, return_tensors="pt", padding=False, truncation=True, max_length=512).input_ids.to(device)
                    if target_ids.size(1) < 2:  # Пропускаем слишком короткие тексты
                        logger.warning(f"Слишком короткий референс: target='{target}', len={target_ids.size(1)}")
                        continue
                    # Получаем логиты
                    outputs = model(target_ids)
                    log_probs = torch.log_softmax(outputs.logits, dim=-1)
                    # Сдвиг: предсказываем следующий токен
                    input_ids = target_ids[:, :-1]
                    target_ids_shifted = target_ids[:, 1:]
                    # Извлекаем логарифмы вероятностей для целевых токенов
                    selected_log_probs = torch.gather(
                        log_probs[:, :-1, :],  # Исключаем последний логит
                        dim=-1,
                        index=target_ids_shifted.unsqueeze(-1)
                    ).squeeze(-1)
                    # Суммируем логарифмы вероятностей
                    metrics_data["log_probs"].extend(selected_log_probs.sum(dim=-1).cpu().tolist())
                    metrics_data["total_tokens"] += target_ids_shifted.size(1)

        torch.cuda.empty_cache()

    metrics_data["timestamps"].append(time.time())
    metrics_data["emissions"] = carbon_tracker.stop()
    metrics_data["predictions"] = all_predictions

    if metrics_data["total_tokens"] == 0:
        logger.warning("Не удалось вычислить log_probs: total_tokens=0")
        metrics_data["log_probs"] = []

    metrics_to_check_list = TASK_METRICS[task_type]
    return all_predictions, metrics_data, metrics_to_check_list

def metrics_evaluate(
        model: Any,
        processor: Any,
        dataset: HFDataset,
        f_type: str,
        device: str = "cuda",
        batch_size: Optional[int] = None,
        field_mapping: Optional[Dict[str, str]] = None,
        log: bool = True
) -> Dict[str, Union[float, Dict[str, float]]]:
    task_type = TaskType(f_type)
    if batch_size is None:
        batch_size = len(dataset)

    predictions, metrics_data, metrics_to_check_list = train_func(
        model, processor, dataset, task_type, device, batch_size, field_mapping
    )

    if log:
        logger.info(f"Оценка метрик для задачи: {f_type}, модель: {type(model).__name__}")

    results = {}
    for metric in metrics_to_check_list:
        try:
            metric_func = base_metrics[metric]
            start_time = time.time()
            metric_value = metric_func(**metrics_data)
            duration = time.time() - start_time
            results[metric] = metric_value
            logger.info(f"Метрика {metric} посчитана за {duration:.2f} секунд: {metric_value}")
        except Exception as e:
            logger.warning(f"Ошибка при вычислении {metric}: {str(e)}")
            results[metric] = (
                {rt: float("inf") for rt in ["rouge1", "rouge2", "rougeL"]} if metric == "rouge" else
                {rt: float("inf") for rt in
                 ["bertscore_precision", "bertscore_recall", "bertscore_f1"]} if metric == "bert_score" else
                float("inf")
            )

    return results