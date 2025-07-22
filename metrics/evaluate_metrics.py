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
    TaskType.GENERATION: ["bleu", "rouge", "meteor", "bert_score", "perplexity", "latency", "memory", "throughput", "energy"],
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
        "loss": None,
        "total_examples": len(dataset)
    }

    batches = process_in_batches(dataset, batch_size)
    carbon_tracker = EmissionsTracker()
    carbon_tracker.start()

    total_loss = 0.0
    total_tokens = 0
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

            # Вычисление loss для perplexity (для TRANSLATION и GENERATION)
            if task_type in [TaskType.TRANSLATION, TaskType.GENERATION]:
                for prompt, refs in zip(batch_text, batch_references):
                    if not refs or not prompt.strip() or not refs[0].strip():
                        logger.warning(f"Пропущен пустой промпт или референс: prompt='{prompt}', refs={refs}")
                        continue
                    target = refs[0]
                    # Токенизация промпта и эталонного текста
                    prompt_ids = processor(prompt, return_tensors="pt").input_ids.to(device)
                    target_ids = processor(target, return_tensors="pt").input_ids.to(device)
                    # Объединяем входные данные и метки
                    input_ids = target_ids  # Для perplexity используем только эталонный текст
                    labels = target_ids.clone()
                    # Вычисляем loss
                    outputs = model(input_ids, labels=labels)
                    if outputs.loss is not None and np.isfinite(outputs.loss.item()):
                        loss = outputs.loss.item() * target_ids.size(1)
                        total_loss += loss
                        total_tokens += target_ids.size(1)
                    else:
                        logger.warning(f"Loss не вычислен для target='{target}'")

        torch.cuda.empty_cache()

    metrics_data["timestamps"].append(time.time())
    metrics_data["emissions"] = carbon_tracker.stop()
    metrics_data["predictions"] = all_predictions

    # Средний loss для perplexity
    if total_tokens > 0:
        metrics_data["loss"] = total_loss / total_tokens
        logger.info(f"Вычислен средний loss: {metrics_data['loss']:.4f}, total_tokens: {total_tokens}")
    else:
        metrics_data["loss"] = float("inf")
        logger.warning("Не удалось вычислить loss: total_tokens=0")

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