import numpy as np
import torch
import time
import seaborn as sns
import psutil
from codecarbon import EmissionsTracker
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from numpy import floating
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
    TaskType.TRANSLATION: ["bleu", "rouge", "meteor", "bert_score", "perplexity", "cider", "spice", "latency", "memory", "flops", "throughput", "energy"],
    TaskType.GENERATION: ["bleu", "rouge", "meteor", "bert_score", "perplexity", "cider", "spice", "latency", "memory", "flops", "throughput", "energy"],
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
    "throughput": compute_throughput,
    "energy": compute_energy,
    "ece": compute_ece,
    "mce": compute_mce,
    "mmlu": compute_mmlu,
    "helm": compute_helm,
    "glue": compute_glue
}

def load_model(model: Any, processor: Any, device: str = "cuda") -> Tuple[Any, Any]:
    """Загружает модель и процессор, перемещая модель на указанное устройство.

    Args:
        model: Уже загруженная модель (например, GPT2LMHeadModel, AutoModelForCausalLM).
        processor: Уже загруженный токенизатор или процессор (например, AutoTokenizer, OwlViTProcessor).
        device: Устройство для загрузки модели ('cuda' или 'cpu').

    Returns:
        Tuple из (модель, процессор).

    Raises:
        ValueError: Если модель или процессор не соответствуют ожидаемым типам.
    """
    # Проверка поддержки perplexity для задач GENERATION и TRANSLATION
    if "perplexity" in TASK_METRICS.get(TaskType.GENERATION, []) or "perplexity" in TASK_METRICS.get(TaskType.TRANSLATION, []):
        try:
            dummy_input = processor(["test"], return_tensors="pt").to(device)
            outputs = model(**dummy_input, labels=dummy_input["input_ids"])
            if not hasattr(outputs, "loss") or outputs.loss is None:
                logger.warning(f"Модель {type(model).__name__} может не поддерживать вычисление perplexity")
        except Exception as e:
            logger.warning(f"Модель {type(model).__name__} не поддерживает perplexity: {str(e)}")

    # Перемещение модели на устройство
    model = model.to(device)
    return model, processor

def train_func(
        model: Any,
        processor: Any,
        dataset: HFDataset,
        task_type: TaskType,
        device: str = "cuda",
        batch_size: Optional[int] = None,
        field_mapping: Optional[Dict[str, str]] = None
) -> Tuple[Any, Dict[str, Any], List[str]]:
    """Выполняет инференс задачи и собирает данные для метрик.

    Args:
        model: Загруженная модель (например, GPT2LMHeadModel, AutoModelForCausalLM).
        processor: Загруженный токенизатор или процессор (например, AutoTokenizer, OwlViTProcessor).
        dataset: Датасет Hugging Face с полями, специфичными для задачи.
        task_type: Тип задачи (CLASSIFICATION, TRANSLATION, GENERATION, VISION).
        device: Устройство для работы модели ('cuda' или 'cpu').
        batch_size: Количество образцов для обработки (по умолчанию — размер датасета).
        field_mapping: Пользовательские имена полей датасета (например, {"text": "input", "task_name": "glue_sst2"}).

    Returns:
        Кортеж из (предсказания, данные_метрик, список_метрик), где:
        - предсказания: Выходы задачи (List[str] для текстовых задач, List[Tuple] для визуальных).
        - данные_метрик: Словарь с входами модели и метриками производительности.
        - список_метрик: Список имен метрик для оценки.
    """
    # Загрузка модели и процессора
    model, processor = load_model(model, processor, device)
    field_mapping = field_mapping or {}

    # Извлечение данных из датасета
    text_field = field_mapping.get("text", "text")
    image_field = field_mapping.get("image", "image") if task_type == TaskType.VISION else None
    label_field = field_mapping.get("label", "label") if task_type == TaskType.CLASSIFICATION else None
    reference_field = field_mapping.get("reference", "references") if task_type in [TaskType.TRANSLATION, TaskType.GENERATION, TaskType.VISION] else None

    text = dataset[text_field] if text_field in dataset.column_names else None
    images = dataset[image_field] if image_field in dataset.column_names and task_type == TaskType.VISION else None
    labels = dataset[label_field] if label_field in dataset.column_names and task_type == TaskType.CLASSIFICATION else None
    references = dataset[reference_field] if reference_field in dataset.column_names else None

    # Минимальная валидация входных данных
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
        "gpu_memory": None,
        "cpu_memory": None,
        "emissions": None,
        "image_features": None,
        "text_features": None,
        "confidences": None,
        "predictions": None,
        "loss": None,
        "flops": None
    }

    # Подготовка данных для модели
    if task_type in [TaskType.CLASSIFICATION, TaskType.TRANSLATION, TaskType.GENERATION]:
        if text and processor:
            inputs = processor(text, return_tensors="pt", padding=True, truncation=True).to(device)
            metrics_data["inputs"] = inputs
            metrics_data["input_ids"] = inputs["input_ids"]
    elif task_type == TaskType.VISION:
        if processor:
            inputs = processor(text=text, images=images, return_tensors="pt", padding=True, truncation=True)
            metrics_data["inputs"] = {"input_ids": inputs["input_ids"].to(device), "pixel_values": inputs["pixel_values"].to(device)}
            metrics_data["input_ids"] = inputs["input_ids"]
            # Предвычисление text_features для clip_score_vision
            try:
                text_inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
                metrics_data["text_features"] = model.get_text_features(**text_inputs)
            except Exception as e:
                logger.warning(f"CLIPScore: Ошибка при вычислении text_features: {str(e)}")
                metrics_data["text_features"] = None

    # Предвычисление flops
    metrics_data["flops"] = compute_flops_in_train(
        model=model,
        processor=processor,
        raw_text=text,
        images=images,
        task_name=metrics_data["task_name"],
        device=device
    )

    # Замеры производительности
    carbon_tracker = EmissionsTracker()
    carbon_tracker.start()

    if device == "cuda":
        torch.cuda.synchronize()
        metrics_data["gpu_memory"] = [torch.cuda.memory_allocated() / (1024 ** 2)]

    metrics_data["cpu_memory"] = psutil.Process().memory_info().rss / (1024 ** 2)

    # Выполнение задачи
    with torch.no_grad():
        if task_type == TaskType.CLASSIFICATION:
            outputs = model(**metrics_data["inputs"])
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            confidences = probabilities.max(dim=-1)[0].cpu().tolist()
            pred_indices = torch.argmax(logits, dim=-1).cpu().tolist()
            decoded_predictions = [str(i) for i in pred_indices]
            predictions = decoded_predictions
            metrics_data["confidences"] = confidences
            metrics_data["labels"] = [str(label) for label in metrics_data["labels"]]
        elif task_type in [TaskType.TRANSLATION, TaskType.GENERATION]:
            # Вычисление loss для perplexity
            if metrics_data["input_ids"] is not None:
                try:
                    outputs = model(**metrics_data["inputs"], labels=metrics_data["input_ids"])
                    if hasattr(outputs, "loss") and outputs.loss is not None:
                        metrics_data["loss"] = outputs.loss.item()
                    else:
                        logger.warning("Perplexity: Модель не вернула loss")
                        metrics_data["loss"] = float("inf")
                except Exception as e:
                    logger.warning(f"Perplexity: Ошибка при вычислении loss: {str(e)}")
                    metrics_data["loss"] = float("inf")
            # Генерация предсказаний
            generate_kwargs = field_mapping.get("generate_kwargs", {"max_length": 50, "num_beams": 5, "do_sample": False})
            predictions = model.generate(**metrics_data["inputs"], **generate_kwargs)
            decoded_predictions = [processor.decode(pred, skip_special_tokens=True).strip() for pred in predictions]
            predictions = [p if p else "" for p in decoded_predictions]
            logger.info(f"Предсказания: {predictions}")
            logger.info(f"Референсы: {references}")
        elif task_type == TaskType.VISION:
            outputs = model.image_guided_detection(**metrics_data["inputs"])
            predictions = [(outputs.pred_boxes, outputs.scores, outputs.object_embeds) if hasattr(outputs, "pred_boxes") and hasattr(outputs, "object_embeds") else (None, None, None)]
            if hasattr(outputs, "image_embeds"):
                metrics_data["image_features"] = outputs.image_embeds

    # Финализация замеров
    metrics_data["timestamps"].append(time.time())
    if device == "cuda":
        torch.cuda.synchronize()
        metrics_data["gpu_memory"].append(torch.cuda.memory_allocated() / (1024 ** 2))

    metrics_data["emissions"] = carbon_tracker.stop()
    metrics_data["predictions"] = predictions

    # Все метрики
    metrics_to_check_list = TASK_METRICS[task_type]

    return predictions, metrics_data, metrics_to_check_list

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
    """Оценивает все применимые метрики для заданной задачи.

    Args:
        model: Загруженная модель.
        processor: Загруженный токенизатор или процессор.
        dataset: Датасет Hugging Face с полями, специфичными для задачи.
        f_type: Тип задачи (например, 'classification').
        device: Устройство для работы модели ('cuda' или 'cpu').
        batch_size: Количество образцов для обработки (по умолчанию — размер датасета).
        field_mapping: Пользовательские имена полей датасета.
        log: Нужно ли логировать результаты метрик.

    Returns:
        Словарь, сопоставляющий имена метрик их значениям (float, dict или None).
    """
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    task_type = TaskType(f_type)
    if batch_size is None:
        batch_size = len(dataset)

    # Выполнение инференса и сбор данных для метрик
    predictions, metrics_data, metrics_to_check_list = train_func(
        model, processor, dataset, task_type, device, batch_size, field_mapping
    )

    if log:
        logger.info(f"Оценка метрик для задачи: {f_type}, модель: {type(model).__name__}")

    # Вычисление каждой метрики
    results = {metric: [] for metric in metrics_to_check_list}
    for metric in metrics_to_check_list:
        try:
            if metric == "flops":
                # Для flops берем предвычисленное значение из metrics_data
                results[metric].append(metrics_data["flops"])
                if log:
                    logger.info(f"Метрика {metric}: {metrics_data['flops']}")
            else:
                metric_func = base_metrics[metric]
                logger.info(f"Вычисление метрики: {metric}")
                metric_value = metric_func(**metrics_data)
                results[metric].append(metric_value)
                if log:
                    logger.info(f"Метрика {metric}: {metric_value}")
        except Exception as e:
            logger.warning(f"Ошибка при вычислении {metric}: {str(e)}")
            results[metric].append(
                {rt: float("inf") for rt in ["rouge1", "rouge2", "rougeL"]} if metric == "rouge" else float("inf")
            )

    # Агрегация результатов
    results["failed_metrics"] = [
        metric for metric, values in results.items()
        if all(
            v is None or
            (isinstance(v, dict) and all(not np.isfinite(val) for val in v.values())) or
            (not isinstance(v, dict) and not np.isfinite(v))
            for v in values
        )
    ]
    return {
        metric: values[0] if isinstance(values[0], dict) else (
            np.nanmean([v for v in values if v is not None and np.isfinite(v)])
            if any(v is not None and np.isfinite(v) for v in values) else float("inf")
        )
        for metric, values in results.items() if metric != "failed_metrics"
    }