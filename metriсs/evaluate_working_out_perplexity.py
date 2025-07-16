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


# Определение поддерживаемых классов моделей и процессоров
MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, AutoTokenizer),
    "llama3_8b": (AutoModelForCausalLM, AutoTokenizer),
    "mistral_7b": (AutoModelForCausalLM, AutoTokenizer),
    "owlv2": (OwlViTForObjectDetection, OwlViTProcessor),
    "qwen3_4b": (AutoModelForCausalLM, AutoTokenizer)
}

# Конфигурации моделей
MODEL_CONFIGS = {
    "gpt2": "openai-community/gpt2",
    "llama3_8b": "meta-llama/Llama-3-8b",
    "mistral_7b": "mistralai/Mixtral-7B-v0.1",
    "owlv2": "google/owlv2-large-patch14-ensemble",
    "qwen3_4b": "Qwen/Qwen3-4B"
}

# Метрики для задач
TASK_METRICS = {
    TaskType.CLASSIFICATION: ["accuracy", "ece", "mce", "mmlu", "glue", "latency", "memory", "flops", "throughput",
                              "energy"],
    TaskType.TRANSLATION: ["bleu", "rouge", "meteor", "bert_score", "perplexity", "latency", "memory", "flops",
                           "throughput", "energy"],  # Добавлено perplexity
    TaskType.GENERATION: ["bleu", "rouge", "meteor", "bert_score", "perplexity", "latency", "memory", "flops",
                          "throughput", "energy"],
    TaskType.VISION: ["iou", "map", "precision", "recall", "f1", "clip_score_vision", "latency", "memory", "flops",
                      "throughput", "energy"]
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
    "cider": compute_cider,
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


def load_model(model: Any, processor: Any, device: str = "cuda") -> Tuple[Any, Any]:
    """Загружает модель и процессор, перемещая модель на указанное устройство."""
    valid_model_types = tuple(cls[0] for cls in MODEL_CLASSES.values())
    valid_processor_types = tuple(cls[1] for cls in MODEL_CLASSES.values())

    if not isinstance(model, valid_model_types):
        raise ValueError(f"Модель должна быть одним из типов: {valid_model_types}")
    if not isinstance(processor, valid_processor_types):
        raise ValueError(f"Процессор должен быть одним из типов: {valid_processor_types}")

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
    """Выполняет инференс задачи и собирает данные для метрик."""
    model, processor = load_model(model, processor, device)
    field_mapping = field_mapping or {}

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

    if task_type in [TaskType.TRANSLATION, TaskType.GENERATION] and references:
        references = [ref if isinstance(ref, list) else [ref] for ref in references]

    if task_type == TaskType.CLASSIFICATION and (not text or not labels):
        raise ValueError("Для классификации нужны текст и метки")
    elif task_type == TaskType.TRANSLATION and (not text or not references):
        raise ValueError("Для перевода нужны текст и референсы")
    elif task_type == TaskType.GENERATION and not text:
        raise ValueError("Для генерации нужен текст")
    elif task_type == TaskType.VISION:
        if not text or not images or not references:
            raise ValueError("Для визуальной задачи нужны текст, изображения и референсы")
        for ref in references:
            if not isinstance(ref, tuple) or len(ref) != 2 or not isinstance(ref[0], torch.Tensor) or not isinstance(
                    ref[1], torch.Tensor):
                raise ValueError(
                    "Референсы для визуальной задачи должны быть списком кортежей [torch.Tensor, torch.Tensor]")

    metrics_data = {
        "model": model,
        "raw_text": text,
        "text": text,
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
        "confidences": None,
        "predictions": None
    }

    if task_type in [TaskType.CLASSIFICATION, TaskType.TRANSLATION, TaskType.GENERATION]:
        if text and processor:
            text = processor(text, return_tensors="pt", padding=True, truncation=True).to(device)
            metrics_data["text"] = text
    elif task_type == TaskType.VISION:
        if processor:
            inputs = processor(text=text, images=images, return_tensors="pt", padding=True, truncation=True)
            text = inputs["input_ids"].to(device)
            images = inputs["pixel_values"].to(device)
            metrics_data["text"] = text
            metrics_data["images"] = images

    carbon_tracker = EmissionsTracker()
    carbon_tracker.start()

    if device == "cuda":
        torch.cuda.synchronize()
        metrics_data["gpu_memory"] = [torch.cuda.memory_allocated() / (1024 ** 2)]

    metrics_data["cpu_memory"] = psutil.Process().memory_info().rss / (1024 ** 2)

    with torch.no_grad():
        if task_type == TaskType.CLASSIFICATION:
            outputs = model(**text)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            confidences = probabilities.max(dim=-1)[0].cpu().tolist()
            pred_indices = torch.argmax(logits, dim=-1).cpu().tolist()
            decoded_predictions = [str(i) for i in pred_indices]
            predictions = decoded_predictions
            metrics_data["confidences"] = confidences
            metrics_data["labels"] = [str(label) for label in metrics_data["labels"]]
        elif task_type in [TaskType.TRANSLATION, TaskType.GENERATION]:
            predictions = model.generate(**text, max_length=20, num_beams=10, do_sample=False, early_stopping=True)
            decoded_predictions = [processor.decode(pred, skip_special_tokens=True) for pred in predictions]
            predictions = decoded_predictions
            logger.info(f"Предсказания: {predictions}")
            logger.info(f"Референсы: {references}")
        elif task_type == TaskType.VISION:
            outputs = model.image_guided_detection(images=images, text=text)
            predictions = [(outputs.pred_boxes, outputs.scores, outputs.object_embeds) if hasattr(outputs,
                                                                                                  "pred_boxes") and hasattr(
                outputs, "object_embeds") else (None, None, None)]

    metrics_data["timestamps"].append(time.time())
    if device == "cuda":
        torch.cuda.synchronize()
        metrics_data["gpu_memory"].append(torch.cuda.memory_allocated() / (1024 ** 2))

    metrics_data["emissions"] = carbon_tracker.stop()
    metrics_data["predictions"] = predictions

    if task_type == TaskType.VISION and hasattr(outputs, "image_embeds"):
        metrics_data["image_features"] = outputs.image_embeds

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
) -> Dict[str, Union[float, Dict[str, float], None]]:
    """Оценивает все применимые метрики для заданной задачи."""
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    task_type = TaskType(f_type)
    if batch_size is None:
        batch_size = len(dataset)

    predictions, metrics_data, metrics_to_check_list = train_func(
        model, processor, dataset, task_type, device, batch_size, field_mapping
    )

    if log:
        logger.info(f"Оценка метрик для задачи: {f_type}, модель: {type(model).__name__}")

    results = {metric: [] for metric in metrics_to_check_list}
    for metric in metrics_to_check_list:
        try:
            metric_func = base_metrics[metric]
            text_input = metrics_data["text"] if metric == "flops" and task_type == TaskType.VISION else metrics_data[
                "raw_text"]
            logger.info(f"Вычисление метрики: {metric}")
            metric_value = metric_func(
                model=metrics_data["model"],
                text=text_input,
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
                logger.info(f"Метрика {metric}: {metric_value}")
        except Exception as e:
            logger.warning(f"Ошибка при вычислении {metric}: {str(e)}")
            results[metric].append(None)

    return {
        metric: values[0] if isinstance(values[0], dict) else np.nanmean([v for v in values if v is not None]) if any(
            v is not None for v in values) else None
        for metric, values in results.items()
    }