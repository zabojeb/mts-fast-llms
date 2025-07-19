import numpy as np
import torch
import time
import concurrent.futures
from codecarbon import EmissionsTracker
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset as HFDataset
from enum import Enum

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    CLASSIFICATION = "classification"
    TRANSLATION = "translation"
    GENERATION = "generation"
    VISION = "vision"


TASK_METRICS = {
    TaskType.CLASSIFICATION: ["accuracy", "ece", "mce", "mmlu", "glue", "latency", "memory", "flops", "throughput",
                              "energy"],
    TaskType.TRANSLATION: ["bleu", "rouge", "meteor", "bert_score", "cider", "spice", "latency", "memory", "flops",
                           "throughput", "energy"],
    TaskType.GENERATION: ["bleu", "rouge", "meteor", "bert_score", "cider", "spice", "latency", "memory", "flops",
                          "throughput", "energy"],
    TaskType.VISION: ["iou", "map", "precision", "recall", "f1", "clip_score_vision", "latency", "memory", "flops",
                      "throughput", "energy"]
}


def load_model(model: Any, processor: Any, device: str = "cuda") -> Tuple[Any, Any]:
    model = model.to(device)
    return model, processor


def process_in_batches(dataset: HFDataset, batch_size: int) -> List[HFDataset]:
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
    image_field = field_mapping.get("image", "image") if task_type == TaskType.VISION else None
    label_field = field_mapping.get("label", "label") if task_type == TaskType.CLASSIFICATION else None
    reference_field = field_mapping.get("reference", "references") if task_type in [TaskType.TRANSLATION,
                                                                                    TaskType.GENERATION,
                                                                                    TaskType.VISION] else None

    # Проверка данных
    text = dataset[text_field] if text_field in dataset.column_names else None
    images = dataset[
        image_field] if image_field and image_field in dataset.column_names and task_type == TaskType.VISION else None
    labels = dataset[
        label_field] if label_field and label_field in dataset.column_names and task_type == TaskType.CLASSIFICATION else None
    references = dataset[reference_field] if reference_field and reference_field in dataset.column_names else None

    # Подготовка данных для метрик
    metrics_data = {
        "model": model,
        "raw_text": text,
        "inputs": None,
        "input_ids": None,
        "images": images,
        "labels": labels,
        "references": references,
        "processor": processor,
        "device": device,
        "batch_size": batch_size if batch_size is not None else len(dataset),
        "task_name": field_mapping.get("task_name", task_type.value),
        "timestamps": [time.time()],
        "gpu_memory": [],
        "cpu_memory": [],
        "emissions": None,
        "image_features": None,
        "text_features": None,
        "confidences": None,
        "predictions": [],
        "loss": None,
        "total_examples": len(dataset),
        "duration": 0.0
    }

    # Инференс
    carbon_tracker = EmissionsTracker(measure_power_secs=1, log_level="error")
    carbon_tracker.start()
    start_time = time.time()

    batches = process_in_batches(dataset, batch_size)
    all_predictions = []

    for batch in batches:
        # Подготовка батча
        batch_text = batch[text_field] if text_field in batch.column_names else None
        batch_images = batch[
            image_field] if image_field and image_field in batch.column_names and task_type == TaskType.VISION else None

        if task_type in [TaskType.CLASSIFICATION, TaskType.TRANSLATION, TaskType.GENERATION]:
            if batch_text and processor:
                inputs = processor(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                    device)
                metrics_data["inputs"] = inputs
                metrics_data["input_ids"] = inputs["input_ids"]

        # Выполнение модели
        with torch.no_grad():
            if task_type == TaskType.CLASSIFICATION:
                outputs = model(**metrics_data["inputs"])
                logits = outputs.logits
                pred_indices = torch.argmax(logits, dim=-1).cpu().tolist()
                all_predictions.extend([str(i) for i in pred_indices])

            elif task_type in [TaskType.TRANSLATION, TaskType.GENERATION]:
                generate_kwargs = field_mapping.get("generate_kwargs", {
                    "max_new_tokens": 150,
                    "num_beams": 1,
                    "do_sample": False,
                })
                predictions = model.generate(**metrics_data["inputs"], **generate_kwargs)
                all_predictions.extend(
                    [processor.decode(pred, skip_special_tokens=True).strip() or "<пусто>" for pred in predictions])

        # Мониторинг ресурсов
        if device == "cuda":
            torch.cuda.synchronize()
            metrics_data["gpu_memory"].append(torch.cuda.memory_allocated() / (1024 ** 2))
        metrics_data["cpu_memory"].append(psutil.Process().memory_info().rss / (1024 ** 2))
        torch.cuda.empty_cache()

    # Финализация
    metrics_data["duration"] = time.time() - start_time
    metrics_data["emissions"] = carbon_tracker.stop()
    metrics_data["predictions"] = all_predictions

    return all_predictions, metrics_data, TASK_METRICS[task_type]


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
    predictions, metrics_data, metrics_to_check = train_func(
        model, processor, dataset, task_type, device, batch_size, field_mapping
    )

    if log:
        logger.info(f"Оценка метрик для задачи: {f_type}")
        logger.info(f"Пример предсказания: {predictions[0] if predictions else None}")
        logger.info(f"Пример референса: {metrics_data['references'][0] if metrics_data.get('references') else None}")

    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        for metric in metrics_to_check:
            metric_func = globals().get(f"compute_{metric}")
            if metric_func:
                futures[executor.submit(metric_func, **metrics_data)] = metric

        for future in concurrent.futures.as_completed(futures):
            metric_name = futures[future]
            try:
                results[metric_name] = future.result()
            except Exception as e:
                logger.warning(f"Ошибка вычисления {metric_name}: {str(e)}")
                results[metric_name] = float("inf")

    return results