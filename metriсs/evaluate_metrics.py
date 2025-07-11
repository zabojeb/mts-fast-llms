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


def load_model(model_name: str, device: str = "cuda") -> Tuple[Any, Any]:
    """Load a model and its tokenizer/processor.

    Args:
        model_name: Name of the model (e.g., 'gpt2', 'owlv2').
        device: Device to load the model on ('cuda' or 'cpu').

    Returns:
        Tuple of (model, processor) where model is a PyTorch module and processor is a tokenizer or vision processor.

    Raises:
        ValueError: If model_name is not supported.

    Note:
        Ensure the model is compatible with the task (e.g., fine-tuned for classification if used with TaskType.CLASSIFICATION).
    """
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
    """Perform task inference and collect data for metrics.

    Args:
        model_name: Name of the model to load (e.g., 'gpt2', 'owlv2').
        dataset: Hugging Face dataset with task-specific fields.
        task_type: Type of task (CLASSIFICATION, TRANSLATION, GENERATION, VISION).
        device: Device to run the model on ('cuda' or 'cpu').
        batch_size: Number of samples to process (defaults to dataset size if None).
        field_mapping: Custom field names for dataset columns (e.g., {"text": "input", "task_name": "glue_sst2"}).
                       Use "task_name" for metrics like glue (e.g., "glue_sst2"), helm ("helm"), or mmlu ("mmlu").

    Returns:
        Tuple of (predictions, metrics_data, metrics_to_check_list), where:
        - predictions: Task-specific outputs (List[str] for text tasks, List[Tuple] for vision).
        - metrics_data: Dictionary with model inputs and performance metrics.
        - metrics_to_check_list: List of metric names to evaluate.
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

    #Извлечение данных из датасета
    text = dataset[text_field] if text_field in dataset.column_names else None
    images = dataset[image_field] if image_field in dataset.column_names and task_type == TaskType.VISION else None
    labels = dataset[
        label_field] if label_field in dataset.column_names and task_type == TaskType.CLASSIFICATION else None
    references = dataset[reference_field] if reference_field in dataset.column_names else None

    # Ensure references is List[List[str]] for TRANSLATION and GENERATION
    if task_type in [TaskType.TRANSLATION, TaskType.GENERATION] and references:
        references = [[ref] if isinstance(ref, str) else ref for ref in references]

    # Проверка входных данных
    if task_type == TaskType.CLASSIFICATION and (not text or not labels):
        raise ValueError("Classification requires text and labels")
    elif task_type == TaskType.TRANSLATION and (not text or not references):
        raise ValueError("Translation requires text and references")
    elif task_type == TaskType.GENERATION and not text:
        raise ValueError("Generation requires text")
    elif task_type == TaskType.VISION:
        if not text or not images or not references:
            raise ValueError("Vision requires text, images, and references")
        for ref in references:
            if not isinstance(ref, tuple) or len(ref) != 2 or not isinstance(ref[0], torch.Tensor) or not isinstance(
                    ref[1], torch.Tensor):
                raise ValueError("Vision references must be List[Tuple[torch.Tensor, torch.Tensor]]")

    # Подготовка данных
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
        "task_name": field_mapping.get("task_name", task_type.value),  # Allow custom task_name
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
            # Get model outputs (assuming logits for classification)
            outputs = model(**text)
            logits = outputs.logits  # Shape: [batch_size, num_classes]

            # Compute probabilities and confidences
            probabilities = torch.softmax(logits, dim=-1)
            confidences = probabilities.max(dim=-1)[0].cpu().tolist()  # Max probability per sample

            # Compute predictions as class indices
            pred_indices = torch.argmax(logits, dim=-1).cpu().tolist()
            decoded_predictions = [str(i) for i in pred_indices]  # Convert to strings for metric compatibility

            # Store in metrics_data
            predictions = decoded_predictions
            metrics_data["confidences"] = confidences
            # Ensure labels are strings for consistency with predictions
            metrics_data["labels"] = [str(label) for label in metrics_data["labels"]]
        elif task_type in [TaskType.TRANSLATION, TaskType.GENERATION]:
            predictions = model.generate(**text, max_length=512)
            decoded_predictions = [processor.decode(pred, skip_special_tokens=True) for pred in predictions]
            predictions = decoded_predictions
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
) -> Dict[str, Union[float, Dict[str, float], None]]:
    """Evaluate all applicable metrics for a given task.

    Args:
        model_name: Name of the model to evaluate.
        dataset: Hugging Face dataset with task-specific fields.
        f_type: Task type as a string (e.g., 'classification').
        device: Device to run the model on ('cuda' or 'cpu').
        batch_size: Number of samples to process (defaults to dataset size if None).
        field_mapping: Custom field names for dataset columns.
        log: Whether to log metric results.

    Returns:
        Dictionary mapping metric names to their values (float, dict, or None).
    """
    # Set plotting style
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    task_type = TaskType(f_type)
    if batch_size is None:
        batch_size = len(dataset)

    # Run inference and collect metrics data
    predictions, metrics_data, metrics_to_check_list = train_func(
        model_name, dataset, task_type, device, batch_size, field_mapping
    )

    if log:
        logger.info(f"Evaluating metrics for task: {f_type}, model: {model_name}")

    # Compute each metric
    results = {metric: [] for metric in metrics_to_check_list}
    for metric in metrics_to_check_list:
        try:
            metric_func = base_metrics[metric]
            # Select text input: raw_text for string-based metrics, text for tokenized inputs
            text_input = metrics_data["text"] if metric == "flops" and task_type == TaskType.VISION else metrics_data[
                "raw_text"]
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
                logger.info(f"Metric {metric}: {metric_value}")
        except Exception as e:
            logger.warning(f"Error computing {metric}: {str(e)}")
            results[metric].append(None)

    """
        Если values[0] — словарь (rouge, bert_score, glue), возвращаем его без изменений.
        Для скалярных значений (float или None) применяем np.nanmean, игнорируя None.
        Используем Union[float, Dict, None] в аннотации возвращаемого типа для совместимости.
        """

    # Aggregate results: return dicts as-is, compute mean for scalars
    return {
        metric: values[0] if isinstance(values[0], dict) else np.nanmean([v for v in values if v is not None]) if any(
            v is not None for v in values) else None
        for metric, values in results.items()
    }