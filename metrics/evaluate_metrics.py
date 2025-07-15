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
    """Выполняет инференс задачи и собирает данные для метрик."""
    model, processor = load_model(model, processor, device)
    field_mapping = field_mapping or {}

    # Извлечение данных из датасета
    text_field = field_mapping.get("text", "text")
    image_field = field_mapping.get("image", "image") if task_type == TaskType.VISION else None
    label_field = field_mapping.get("label", "label") if task_type == TaskType.CLASSIFICATION else None
    reference_field = field_mapping.get("reference", "references") if task_type in [TaskType.TRANSLATION, TaskType.GENERATION, TaskType.VISION] else None

    # Проверка наличия полей в датасете
    text = dataset[text_field] if text_field in dataset.column_names else None
    images = dataset[image_field] if image_field and image_field in dataset.column_names and task_type == TaskType.VISION else None
    labels = dataset[label_field] if label_field and label_field in dataset.column_names and task_type == TaskType.CLASSIFICATION else None
    references = dataset[reference_field] if reference_field and reference_field in dataset.column_names else None

    # Убедимся, что references — это List[List[str]] для текстовых задач
    if task_type in [TaskType.TRANSLATION, TaskType.GENERATION] and references:
        references = [[ref] if isinstance(ref, str) else ref for ref in references]
        references = [r for r in references if r and all(isinstance(s, str) and s.strip() for s in r)]

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
        "loss": None
    }

    # Разбиение на батчи
    batches = process_in_batches(dataset, batch_size)

    # Замеры производительности
    carbon_tracker = EmissionsTracker()
    carbon_tracker.start()

    total_loss = 0.0
    num_batches = 0
    all_predictions = []

    for batch in batches:
        batch_text = batch[text_field] if text_field in batch.column_names else None
        batch_images = batch[image_field] if image_field and image_field in batch.column_names and task_type == TaskType.VISION else None
        batch_references = batch[reference_field] if reference_field in batch.column_names else None

        # Подготовка данных для модели
        if task_type in [TaskType.CLASSIFICATION, TaskType.TRANSLATION, TaskType.GENERATION]:
            if batch_text and processor:
                # Tokenize prompts
                inputs = processor(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                metrics_data["inputs"] = inputs
                metrics_data["input_ids"] = inputs["input_ids"]
        elif task_type == TaskType.VISION:
            if processor:
                inputs = processor(text=batch_text, images=batch_images, return_tensors="pt", padding=True, truncation=True)
                metrics_data["inputs"] = {"input_ids": inputs["input_ids"].to(device), "pixel_values": inputs["pixel_values"].to(device)}
                metrics_data["input_ids"] = inputs["input_ids"]
                try:
                    text_inputs = processor(batch_text, return_tensors="pt", padding=True, truncation=True).to(device)
                    metrics_data["text_features"] = model.get_text_features(**text_inputs)
                except Exception as e:
                    logger.warning(f"CLIPScore: Ошибка при вычислении text_features: {str(e)}")
                    metrics_data["text_features"] = None

        # Замеры памяти
        if device == "cuda":
            torch.cuda.synchronize()
            metrics_data["gpu_memory"].append(torch.cuda.memory_allocated() / (1024 ** 2))
        metrics_data["cpu_memory"].append(psutil.Process().memory_info().rss / (1024 ** 2))

        # Выполнение задачи
        with torch.no_grad(), torch.amp.autocast('cuda'):
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
                # Вычисление loss для perplexity
                if metrics_data["input_ids"] is not None and batch_references is not None:
                    try:
                        # Tokenize references and ensure same batch size and sequence length
                        ref_inputs = processor([ref[0] for ref in batch_references], return_tensors="pt", padding="max_length", truncation=True, max_length=metrics_data["input_ids"].shape[1]).to(device)
                        outputs = model(**metrics_data["inputs"], labels=ref_inputs["input_ids"])
                        if hasattr(outputs, "loss") and outputs.loss is not None:
                            total_loss += outputs.loss.item() * len(batch_text)
                            num_batches += len(batch_text)
                        else:
                            logger.warning("Perplexity: Модель не вернула loss")
                    except Exception as e:
                        logger.warning(f"Perplexity: Ошибка при вычислении loss: {str(e)}")
                # Генерация предсказаний
                generate_kwargs = field_mapping.get("generate_kwargs", {
                    "max_new_tokens": 150,  # Use max_new_tokens instead of max_length
                    "num_beams": 1,
                    "do_sample": False,
                })
                predictions = model.generate(**metrics_data["inputs"], **generate_kwargs)
                decoded_predictions = [processor.decode(pred, skip_special_tokens=True).strip() for pred in predictions]
                all_predictions.extend([p if p else "<пусто>" for p in decoded_predictions])
            elif task_type == TaskType.VISION:
                outputs = model.image_guided_detection(**metrics_data["inputs"])
                all_predictions.extend([(outputs.pred_boxes, outputs.scores, outputs.object_embeds) if hasattr(outputs, "pred_boxes") and hasattr(outputs, "object_embeds") else (None, None, None)])
                if hasattr(outputs, "image_embeds"):
                    metrics_data["image_features"] = outputs.image_embeds

        # Очистка памяти после батча
        torch.cuda.empty_cache()

    # Финализация замеров
    metrics_data["timestamps"].append(time.time())
    metrics_data["emissions"] = carbon_tracker.stop()
    metrics_data["predictions"] = all_predictions
    metrics_data["loss"] = total_loss / num_batches if num_batches > 0 else float("inf")

    # Все метрики
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
    """Оценивает все применимые метрики для заданной задачи."""
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
        logger.info(f"Предсказания (первые 5): {predictions[:5]}")
        logger.info(f"Референсы (первые 5): {metrics_data['references'][:5] if metrics_data['references'] else None}")

    # Вычисление каждой метрики
    results = {metric: [] for metric in metrics_to_check_list}
    for metric in metrics_to_check_list:
        try:
            metric_func = base_metrics[metric]
            logger.info(f"Вычисление метрики: {metric}")
            metric_value = metric_func(**metrics_data)
            results[metric].append(metric_value)
            if log:
                logger.info(f"Метрика {metric}: {metric_value}")
        except Exception as e:
            logger.warning(f"Ошибка при вычислении {metric}: {str(e)}")
            results[metric].append(
                {rt: float("inf") for rt in ["rouge1", "rouge2", "rougeL"]} if metric == "rouge" else
                {rt: float("inf") for rt in ["bertscore_precision", "bertscore_recall", "bertscore_f1"]} if metric == "bert_score" else
                float("inf")
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