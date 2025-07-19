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


# Типы задач
class TaskType(Enum):
    CLASSIFICATION = "classification"
    TRANSLATION = "translation"
    GENERATION = "generation"
    VISION = "vision"


# Метрики для задач
TASK_METRICS = {
    TaskType.CLASSIFICATION: ["accuracy", "ece", "mce", "mmlu", "glue", "latency", "memory", "flops", "throughput",
                              "energy"],
    TaskType.TRANSLATION: ["bleu", "rouge", "meteor", "bert_score", "perplexity", "cider", "spice", "latency", "memory",
                           "flops", "throughput", "energy"],
    TaskType.GENERATION: ["bleu", "rouge", "meteor", "bert_score", "perplexity", "cider", "spice", "latency", "memory",
                          "flops", "throughput", "energy"],
    TaskType.VISION: ["iou", "map", "precision", "recall", "f1", "clip_score_vision", "latency", "memory", "flops",
                      "throughput", "energy"]
}

# Глобальное кэширование для тяжелых объектов
SPICE_CACHE = None
CIDER_CACHE = None


def load_model(model: Any, processor: Any, device: str = "cuda") -> Tuple[Any, Any]:
    """Загружает модель и процессор, перемещая модель на указанное устройство."""
    model = model.to(device)
    return model, processor


def process_in_batches(dataset: HFDataset, batch_size: int) -> List[HFDataset]:
    """Разбивает датасет на батчи."""
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
    """Выполняет инференс задачи и собирает данные для метрик."""
    model, processor = load_model(model, processor, device)
    field_mapping = field_mapping or {}

    # Извлечение данных из датасета
    text_field = field_mapping.get("text", "text")
    image_field = field_mapping.get("image", "image") if task_type == TaskType.VISION else None
    label_field = field_mapping.get("label", "label") if task_type == TaskType.CLASSIFICATION else None
    reference_field = field_mapping.get("reference", "references") if task_type in [TaskType.TRANSLATION,
                                                                                    TaskType.GENERATION,
                                                                                    TaskType.VISION] else None

    # Проверка наличия полей
    text = dataset[text_field] if text_field in dataset.column_names else None
    images = dataset[
        image_field] if image_field and image_field in dataset.column_names and task_type == TaskType.VISION else None
    labels = dataset[
        label_field] if label_field and label_field in dataset.column_names and task_type == TaskType.CLASSIFICATION else None
    references = dataset[reference_field] if reference_field and reference_field in dataset.column_names else None

    # Валидация входных данных
    if task_type == TaskType.CLASSIFICATION and (not text or not labels):
        raise ValueError("Для классификации нужны текст и метки")
    elif task_type == TaskType.TRANSLATION and (not text or not references):
        raise ValueError("Для перевода нужны текст и референсы")
    elif task_type == TaskType.GENERATION and not text:
        raise ValueError("Для генерации нужен текст")
    elif task_type == TaskType.VISION and (not text or not images or not references):
        raise ValueError("Для визуальной задачи нужны текст, изображения и референсы")

    # Подготовка данных
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
        "total_examples": len(dataset)  # Добавлено для вычисления throughput
    }

    # Разбиение на батчи
    batches = process_in_batches(dataset, batch_size)

    # Инициализация трекера ВНУТРИ функции инференса
    carbon_tracker = EmissionsTracker(measure_power_secs=1, log_level="error")
    carbon_tracker.start()

    all_predictions = []
    start_time = time.time()

    for batch in batches:
        batch_text = batch[text_field] if text_field in batch.column_names else None
        batch_images = batch[
            image_field] if image_field and image_field in batch.column_names and task_type == TaskType.VISION else None
        batch_references = batch[reference_field] if reference_field in batch.column_names else None

        # Подготовка данных для модели
        if task_type in [TaskType.CLASSIFICATION, TaskType.TRANSLATION, TaskType.GENERATION]:
            if batch_text and processor:
                inputs = processor(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                    device)
                metrics_data["inputs"] = inputs
                metrics_data["input_ids"] = inputs["input_ids"]
        elif task_type == TaskType.VISION:
            if processor:
                inputs = processor(text=batch_text, images=batch_images, return_tensors="pt", padding=True,
                                   truncation=True)
                metrics_data["inputs"] = {"input_ids": inputs["input_ids"].to(device),
                                          "pixel_values": inputs["pixel_values"].to(device)}
                metrics_data["input_ids"] = inputs["input_ids"]

        # Выполнение задачи
        with torch.no_grad():
            if task_type == TaskType.CLASSIFICATION:
                outputs = model(**metrics_data["inputs"])
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                confidences = probabilities.max(dim=-1)[0].cpu().tolist()
                pred_indices = torch.argmax(logits, dim=-1).cpu().tolist()
                decoded_predictions = [str(i) for i in pred_indices]
                all_predictions.extend(decoded_predictions)
                metrics_data["confidences"] = confidences
                metrics_data["labels"] = [str(label) for label in batch[label_field]]

            elif task_type in [TaskType.TRANSLATION, TaskType.GENERATION]:
                # Генерация предсказаний
                generate_kwargs = field_mapping.get("generate_kwargs", {
                    "max_new_tokens": 150,
                    "num_beams": 1,
                    "do_sample": False,
                })
                predictions = model.generate(**metrics_data["inputs"], **generate_kwargs)
                decoded_predictions = [processor.decode(pred, skip_special_tokens=True).strip() for pred in predictions]
                all_predictions.extend([p if p else "<пусто>" for p in decoded_predictions])

            elif task_type == TaskType.VISION:
                outputs = model.image_guided_detection(**metrics_data["inputs"])
                all_predictions.extend([(outputs.pred_boxes, outputs.scores, outputs.object_embeds) if hasattr(outputs,
                                                                                                               "pred_boxes") and hasattr(
                    outputs, "object_embeds") else (None, None, None)])

        # Замеры памяти
        if device == "cuda":
            torch.cuda.synchronize()
            metrics_data["gpu_memory"].append(torch.cuda.memory_allocated() / (1024 ** 2))
        metrics_data["cpu_memory"].append(psutil.Process().memory_info().rss / (1024 ** 2))

        # Очистка памяти после батча
        torch.cuda.empty_cache()

    # Финализация замеров
    metrics_data["timestamps"].append(time.time())
    metrics_data["emissions"] = carbon_tracker.stop()
    metrics_data["predictions"] = all_predictions
    metrics_data["duration"] = time.time() - start_time  # Явное сохранение длительности

    # Все метрики
    metrics_to_check_list = TASK_METRICS[task_type]

    return all_predictions, metrics_data, metrics_to_check_list


def compute_metric(metric: str, metrics_data: Dict[str, Any]) -> Tuple[str, Any]:
    """Вычисляет одну метрику с обработкой исключений."""
    try:
        metric_func = globals().get(f"compute_{metric}")
        if not metric_func:
            raise ValueError(f"Метрика {metric} не найдена")
        return metric, metric_func(**metrics_data)
    except Exception as e:
        logger.warning(f"Ошибка при вычислении {metric}: {str(e)}")
        return metric, float("inf")


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
    """Оценивает метрики с оптимизацией производительности."""
    task_type = TaskType(f_type)
    if batch_size is None:
        batch_size = len(dataset)

    # Выполнение инференса и сбор данных
    predictions, metrics_data, metrics_to_check_list = train_func(
        model, processor, dataset, task_type, device, batch_size, field_mapping
    )

    if log:
        logger.info(f"Оценка метрик для задачи: {f_type}, модель: {type(model).__name__}")
        logger.info(f"Предсказания (первые 5): {predictions[:5]}")
        logger.info(f"Референсы (первые 5): {metrics_data['references'][:5] if metrics_data['references'] else None}")

    # Параллельное вычисление метрик
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_metric = {
            executor.submit(compute_metric, metric, metrics_data): metric
            for metric in metrics_to_check_list
        }
        for future in concurrent.futures.as_completed(future_to_metric):
            metric_name, metric_value = future.result()
            results[metric_name] = metric_value

    return results