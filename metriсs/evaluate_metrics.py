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
from metric_functions import *


# Типы задач
class TaskType(Enum):
    CLASSIFICATION = "classification"
    TRANSLATION = "translation"
    GENERATION = "generation"
    VISION = "vision"


# Загружаем только эти модели или добавляем конфиги и тп.
MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, AutoTokenizer),
    "llama3_8b": (AutoModelForCausalLM, AutoTokenizer),
    "mistral_7b": (AutoModelForCausalLM, AutoTokenizer),
    "owlv2": (OwlViTForObjectDetection, OwlViTProcessor)
}

MODEL_CONFIGS = {
    "gpt2": "openai-community/gpt2",
    "llama3_8b": "meta-llama/Llama-3-8b",
    "mistral_7b": "mistralai/Mixtral-7B-v0.1",
    "owlv2": "google/owlv2-large-patch14-ensemble"
}

TASK_METRICS = {
    TaskType.CLASSIFICATION: ["accuracy", "ece", "mce", "mmlu", "glue", "latency", "memory", "flops", "throughput", "energy"],
    TaskType.TRANSLATION: ["bleu", "rouge", "meteor", "bert_score", "latency", "memory", "flops", "throughput", "energy"],
    TaskType.GENERATION: ["bleu", "rouge", "meteor", "bert_score", "perplexity", "clip_score", "latency", "memory", "flops", "throughput", "energy"],
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
    "clip_score": compute_clip_score,
    "clip_score_vision": compute_clip_score_vision,
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


def load_model(model_name: str, device: str = "cuda") -> Tuple[Any, Any]:
    """Загрузка модели и токенизатора/процессора."""
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(MODEL_CLASSES.keys())}")

    model_class, processor_class = MODEL_CLASSES[model_name]
    model = model_class.from_pretrained(MODEL_CONFIGS[model_name]).to(device)
    processor = processor_class.from_pretrained(MODEL_CONFIGS[model_name])
    return model, processor


def train_func(
        model_name: str,
        dataset: HFDataset,
        task_type: TaskType,
        device: str = "cuda",
        batch_size: Optional[int] = None,
        field_mapping: Optional[Dict[str, str]] = None
) -> Tuple[Any, Dict[str, Any], List[str]]:
    """Выполнение задачи и сбор данных для метрик.
        :param model_name:
        :param dataset:
        :param batch_size:
        :param device:
        :param task_type:
        :param field_mapping: измененные имена полей
    """
    model, processor = load_model(model_name, device)
    field_mapping = field_mapping or {}

    # Извлечение данных из датасета
    text_field = field_mapping.get("text", "text")
    image_field = field_mapping.get("image", "image") if task_type == TaskType.VISION else None
    label_field = field_mapping.get("label", "label") if task_type == TaskType.CLASSIFICATION else None
    reference_field = field_mapping.get("reference", "references") if task_type in [TaskType.TRANSLATION,
                                                                                    TaskType.GENERATION,
                                                                                    TaskType.VISION] else None

    text = dataset[text_field] if text_field in dataset.column_names else None
    images = dataset[image_field] if image_field in dataset.column_names and task_type == TaskType.VISION else None
    labels = dataset[
        label_field] if label_field in dataset.column_names and task_type == TaskType.CLASSIFICATION else None
    references = dataset[reference_field] if reference_field in dataset.column_names else None

    # Проверка входных данных
    if task_type == TaskType.CLASSIFICATION and (not text or not labels):
        raise ValueError("Classification requires text and labels")
    elif task_type == TaskType.TRANSLATION and (not text or not references):
        raise ValueError("Translation requires text and references")
    elif task_type == TaskType.GENERATION and not text:
        raise ValueError("Generation requires text")
    elif task_type == TaskType.VISION and (not text or not images):
        raise ValueError("Vision requires text and images")

    # Подготовка данных
    metrics_data = {
        "model": model,
        "text": text,
        "images": images,
        "labels": labels,
        "references": references,
        "processor": processor,
        "device": device,
        "batch_size": batch_size,
        "task_name": task_type.value,
        "timestamps": [time.time()],
        "gpu_memory": None,
        "cpu_memory": None,
        "emissions": None,
        "image_features": None,
        "confidences": None,
        "predictions": None
    }

    # Подготовка данных для модели
    if task_type in [TaskType.CLASSIFICATION, TaskType.TRANSLATION, TaskType.GENERATION]:
        if text and processor:
            text = processor(text, return_tensors="pt", padding=True, truncation=True).to(device)
            metrics_data["text"] = text
    elif task_type == TaskType.VISION:
        if processor:  # OwlViTProcessor
            inputs = processor(text=text, images=images, return_tensors="pt", padding=True, truncation=True)
            text = inputs["input_ids"].to(device)
            images = inputs["pixel_values"].to(device)
            metrics_data["text"] = text
            metrics_data["images"] = images

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
            predictions = model(text).logits
        elif task_type == TaskType.TRANSLATION:
            predictions = model.generate(text, max_length=512)
        elif task_type == TaskType.GENERATION:
            predictions = model.generate(text, max_length=512)
        elif task_type == TaskType.VISION:
            # Для OwlV2: predictions = [(boxes, scores, object_embeds), ...]
            outputs = model.image_guided_detection(images=images, text=text)
            predictions = [(outputs.pred_boxes, outputs.scores, outputs.object_embeds) if hasattr(outputs, "pred_boxes") and hasattr(outputs, "object_embeds") else (None, None, None)]  # Адаптация под OwlV2

    # Финализация замеров
    metrics_data["timestamps"].append(time.time())
    if device == "cuda":
        torch.cuda.synchronize()
        metrics_data["gpu_memory"].append(torch.cuda.memory_allocated() / (1024 ** 2))

    metrics_data["emissions"] = carbon_tracker.stop()
    metrics_data["predictions"] = predictions

    # Конфиденсы для классификации
    if task_type == TaskType.CLASSIFICATION and hasattr(model, "predict_proba"):
        metrics_data["confidences"] = model.predict_proba(text)

    # Эмбеддинги изображений для vision (OwlV2)
    if task_type == TaskType.VISION and model_name == "owlv2":
        metrics_data["image_features"] = outputs.image_embeds if hasattr(outputs, "image_embeds") else None

    # Все метрики
    metrics_to_check_list = TASK_METRICS[task_type]

    return predictions, metrics_data, metrics_to_check_list


def metrics_evaluate(
        model_name: str,
        dataset: HFDataset,
        f_type: str,
        device: str = "cuda",
        batch_size: Optional[int] = None,
        field_mapping: Optional[Dict[str, str]] = None,
        log: bool = True
) -> dict[str, floating]:
    """Вычисление всех применимых метрик для задачи."""
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    task_type = TaskType(f_type)
    # Выполнение задачи
    predictions, metrics_data, metrics_to_check_list = train_func(
        model_name, dataset, task_type, device, batch_size, field_mapping
    )

    if log:
        logger.info(f"Evaluating metrics for task: {f_type}, model: {model_name}")

    results = {metric: [] for metric in metrics_to_check_list}

    for metric in metrics_to_check_list:
        try:
            metric_func = base_metrics[metric]
            metric_value = metric_func(
                model=metrics_data["model"],
                text=metrics_data["text"],
                images=metrics_data["images"],
                references=metrics_data["references"],
                labels=metrics_data["labels"],
                processor=metrics_data["processor"],
                device=metrics_data["device"],
                batch_size=metrics_data["batch_size"],
                task_name=metrics_data["task_name"],
                timestamps=metrics_data["timestamps"],
                gpu_memory=metrics_data["gpu_memory"],
                cpu_memory=metrics_data["cpu_memory"],
                emissions=metrics_data["emissions"],
                image_features=metrics_data["image_features"],
                confidences=metrics_data["confidences"],
                predictions=metrics_data["predictions"]
            )
            results[metric].append(metric_value)
            if log:
                logger.info(f"Metric {metric}: {metric_value}")
        except Exception as e:
            logger.warning(f"Error computing {metric}: {str(e)}")
            results[metric].append(None)

    # Агрегация результатов
    return {metric: np.nanmean(values) for metric, values in results.items()}