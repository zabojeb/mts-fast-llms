import pandas as pd
import json
import uuid
from typing import Dict, Any, List, Optional
import logging
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pool_2_pool(
    metrics_results: Dict[str, Any],
    speed_coef: float,
    memory_coef: float,
    logs: Dict[str, Any],
    configs: List[Dict[str, Any]],
    model_name: str,
    task_type: str,
    excel_file: str = "configs_pool.xlsx"
) -> None:
    """
    Агрегирует результаты оптимизации и метрик в два листа Excel: 'pool' и 'leader_board'.

    Args:
        metrics_results: Словарь метрик из функции metrics_evaluate (rouge, bert_score, memory подаются как словари).
        speed_coef: Коэффициент скорости оптимизации.
        memory_coef: Коэффициент памяти оптимизации.
        logs: JSON-сериализуемый словарь логов оптимизации.
        configs: Список словарей конфигураций для конвейера оптимизации.
        model_name: Имя модели (например, "GPT2").
        task_type: Тип задачи (например, "text_generation", "vision").
        excel_file: Путь к файлу Excel для обновления.

    Returns:
        None: Обновляет файл Excel на месте.
    """
    # Определение применимых метрик для типов задач
    task_metrics = {
        "classification": [
            "accuracy", "ece", "mce", "mmlu", "glue",
            "latency", "memory_gpu_peak", "memory_cpu_peak", "flops", "throughput", "energy"
        ],
        "translation": [
            "bleu", "rouge1", "rouge2", "rougeL", "bert_precision", "bert_recall", "bert_f1",
            "meteor", "perplexity", "latency", "memory_gpu_peak", "memory_cpu_peak", "flops",
            "throughput", "energy"
        ],
        "text_generation": [
            "bleu", "rouge1", "rouge2", "rougeL", "bert_precision", "bert_recall", "bert_f1",
            "meteor", "perplexity", "cider", "spice", "latency", "memory_gpu_peak",
            "memory_cpu_peak", "flops", "throughput", "energy"
        ],
        "vision": [
            "iou", "map", "v_precision", "v_recall", "v_f1", "clip_score_vision",
            "latency", "memory_gpu_peak", "memory_cpu_peak", "flops", "throughput", "energy"
        ]
    }

    # Метод для определения типа метода
    def get_method_type(configs: List[Dict[str, Any]]) -> str:
        if len(configs) > 1:
            return "complex"
        method_name = configs[0].get("method", {}).get("__name__", "unknown")
        if method_name == "quantize":
            return "Q"
        elif method_name == "prune":
            return "P"
        elif method_name == "distill":
            return "D"
        return "unknown"

    # Определение типов данных для pool
    pool_dtypes = {
        "id": str,
        "config": str,
        "method": str,
        "logs": str,
        "task_type": str,
        "speed_coef": float,
        "memory_coef": float,
        "accuracy": float,
        "bleu": float,
        "rouge1": float,
        "rouge2": float,
        "rougeL": float,
        "perplexity": float,
        "bert_precision": float,
        "bert_recall": float,
        "bert_f1": float,
        "meteor": float,
        "iou": float,
        "map": float,
        "v_precision": float,
        "v_recall": float,
        "v_f1": float,
        "cider": float,
        "spice": float,
        "clip_score_vision": float,
        "latency": float,
        "memory_gpu_peak": float,
        "memory_cpu_peak": float,
        "flops": float,
        "throughput": float,
        "energy": float,
        "ece": float,
        "mce": float,
        "mmlu": float,
        "helm": float,
        "glue": float
    }

    # Определение типов данных для leader_board
    leader_board_dtypes = {
        "config_id": str,
        "method": str,
        "config": str,
        "task_type": str,
        "metric": str,
        "score": float,
        "memory_coef": float,
        "speed_coef": float,
        "model": str
    }

    # Подготовка данных для pool
    pool_id = str(uuid.uuid4())
    method_type = get_method_type(configs)
    config_json = json.dumps(configs)
    logs_json = json.dumps(logs)

    # Разворачивание метрик rouge, bert_score и memory из входных словарей
    rouge_scores = metrics_results.get("rouge", {})
    bert_scores = metrics_results.get("bert_score", {})
    memory_scores = metrics_results.get("memory", {})

    # Данные для pool
    pool_data = {
        "id": pool_id,
        "config": config_json,
        "method": method_type,
        "logs": logs_json,
        "task_type": task_type,
        "speed_coef": speed_coef,
        "memory_coef": memory_coef,
        "accuracy": None,
        "bleu": None,
        "rouge1": None,
        "rouge2": None,
        "rougeL": None,
        "perplexity": None,
        "bert_precision": None,
        "bert_recall": None,
        "bert_f1": None,
        "meteor": None,
        "iou": None,
        "map": None,
        "v_precision": None,
        "v_recall": None,
        "v_f1": None,
        "cider": None,
        "spice": None,
        "clip_score_vision": None,
        "latency": None,
        "memory_gpu_peak": None,
        "memory_cpu_peak": None,
        "flops": None,
        "throughput": None,
        "energy": None,
        "ece": None,
        "mce": None,
        "mmlu": None,
        "helm": None,
        "glue": None
    }

    # Заполнение метрик с учетом task_type и проверки на inf
    applicable_metrics = task_metrics.get(task_type, [])
    for metric in pool_dtypes.keys():
        if metric in ["id", "config", "method", "logs", "task_type", "speed_coef", "memory_coef"]:
            continue
        if metric not in applicable_metrics:
            continue
        if metric in ["rouge1", "rouge2", "rougeL"]:
            value = rouge_scores.get(metric, float("inf"))
            pool_data[metric] = None if not np.isfinite(value) else value
        elif metric in ["bert_precision", "bert_recall", "bert_f1"]:
            value = bert_scores.get(metric, float("inf"))
            pool_data[metric] = None if not np.isfinite(value) else value
        elif metric == "memory_gpu_peak":
            value = memory_scores.get("gpu_peak_mb", float("inf"))
            pool_data[metric] = None if not np.isfinite(value) else value
        elif metric == "memory_cpu_peak":
            value = memory_scores.get("cpu_mb", float("inf"))
            if isinstance(value, list) and value:
                value = max(value) if all(np.isfinite(v) for v in value) else float("inf")
            pool_data[metric] = None if not np.isfinite(value) else value
        else:
            value = metrics_results.get(metric, float("inf"))
            pool_data[metric] = None if not np.isfinite(value) else value

    # Загрузка существующего Excel файла или создание нового
    try:
        xl = pd.ExcelFile(excel_file)
        pool_df = pd.read_excel(xl, sheet_name="pool")
        # Приведение типов данных
        pool_df = pool_df.astype({k: v for k, v in pool_dtypes.items() if k in pool_df.columns}, errors="ignore")
    except (FileNotFoundError, ValueError):
        pool_df = pd.DataFrame(columns=list(pool_dtypes.keys())).astype(pool_dtypes)

    # Добавление новой строки в pool
    new_pool_row = pd.DataFrame([pool_data]).astype(pool_dtypes, errors="ignore")
    pool_df = pd.concat([pool_df, new_pool_row], ignore_index=True)

    # Подготовка данных для leader_board
    leader_board_data = []
    # Перебираем все числовые метрики из pool_dtypes
    for metric in pool_dtypes.keys():
        if metric in ["id", "config", "method", "logs", "task_type", "speed_coef", "memory_coef"]:
            continue
        if metric not in applicable_metrics:
            continue
        # Получаем значение из pool_data (уже обработанное)
        score = pool_data.get(metric)
        if score is None or not np.isfinite(score):
            continue
        # Формируем имя метрики для leader_board
        if metric in ["rouge1", "rouge2", "rougeL"]:
            full_metric = f"rouge_{metric}"
        elif metric in ["bert_precision", "bert_recall", "bert_f1"]:
            full_metric = metric
        elif metric in ["memory_gpu_peak", "memory_cpu_peak"]:
            full_metric = metric
        else:
            full_metric = metric
        leader_board_data.append({
            "config_id": pool_id,
            "method": method_type,
            "config": config_json,
            "task_type": task_type,
            "metric": full_metric,
            "score": score,
            "memory_coef": memory_coef,
            "speed_coef": speed_coef,
            "model": model_name
        })

    # Загрузка или создание leader_board
    try:
        leader_board_df = pd.read_excel(xl, sheet_name="leader_board")
        # Приведение типов данных
        leader_board_df = leader_board_df.astype({k: v for k, v in leader_board_dtypes.items() if k in leader_board_df.columns}, errors="ignore")
    except (FileNotFoundError, ValueError):
        leader_board_df = pd.DataFrame(columns=list(leader_board_dtypes.keys())).astype(leader_board_dtypes)

    # Добавление новых строк в leader_board
    if leader_board_data:
        new_leader_board_rows = pd.DataFrame(leader_board_data).astype(leader_board_dtypes, errors="ignore")
        leader_board_df = pd.concat([leader_board_df, new_leader_board_rows], ignore_index=True)
    else:
        logger.warning("Нет валидных данных для добавления в leader_board.")

    # Сохранение в Excel
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        pool_df.to_excel(writer, sheet_name="pool", index=False)
        leader_board_df.to_excel(writer, sheet_name="leader_board", index=False)

    logger.info(f"Данные успешно сохранены в {excel_file}: pool ({len(pool_df)} строк), leader_board ({len(leader_board_df)} строк)")

# Пример использования
if __name__ == "__main__":
    # Пример данных
    metrics_results = {
        "bleu": 0.01123156547865983,
        "rouge": {
            "rouge1": 0.18505802145768055,
            "rouge2": 0.09023545457034798,
            "rougeL": 0.1669645181182756
        },
        "meteor": 0.14090284396855438,
        "bert_score": {
            "bert_precision": 0.18041212856769562,
            "bert_recall": 0.4437793791294098,
            "bert_f1": 0.24777436256408691
        },
        "perplexity": 9.20208823683661,
        "latency": 111.06169772148132,
        "memory": {
            "gpu_peak_mb": 2778.34375,
            "cpu_mb": [1544.421875, 1646.2265625, 1646.5]
        },
        "flops": float("inf"),
        "throughput": 0.09004004265338923,
        "energy": 3.2245852281379372,
        "accuracy": float("inf")  # Метрика, не относящаяся к text_generation
    }

    speed_coef = 0.85
    memory_coef = 0.92
    logs = {"optimization_steps": ["step1", "step2"], "status": "success"}
    configs = [{"method": {"__name__": "quantize"}, "params": {"bits": 4}}]
    model_name = "Qwen7B"
    task_type = "text_generation"

    # Вызов функции
    pool_2_pool(
        metrics_results=metrics_results,
        speed_coef=speed_coef,
        memory_coef=memory_coef,
        logs=logs,
        configs=configs,
        model_name=model_name,
        task_type=task_type,
        excel_file="configs_pool.xlsx"
    )