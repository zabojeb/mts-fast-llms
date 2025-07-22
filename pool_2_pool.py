import pandas as pd
import json
import uuid
from typing import Dict, Any, List, Optional
import logging
import numpy as np
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pool_2_pool(
    metrics_results: Dict[str, Any],
    logs: Dict[str, Any] = None,
    configs: List[Dict[str, Any]] = None,
    model_name: str = None,
    task_type: str = None,
    excel_file: str = "configs_pool.xlsx",
    original: bool = False
) -> None:
    """
    Агрегирует результаты оптимизации и метрик в два листа Excel: 'pool' и 'leader_board'.
    Не вычисляет и не сохраняет speed_coef и memory_coef. Использует реальный конфиг квантизации.

    Args:
        metrics_results: Словарь метрик из функции metrics_evaluate (rouge, bert_score, memory подаются как словари).
        logs: JSON-сериализуемый словарь логов оптимизации.
        configs: Список словарей конфигураций для конвейера оптимизации.
        model_name: Имя модели (например, "Qwen/Qwen3-4B").
        task_type: Тип задачи (например, "generation").
        excel_file: Путь к файлу Excel для обновления.
        original: Если True, сохраняет метрики как для исходной модели (method='original').

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
        "generation": [
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
        if not configs or len(configs) > 1:
            return "complex"
        method_name = configs[0].get("method", {}).get("__name__", "unknown")
        if method_name == "quantize":
            return "Q"
        elif method_name == "prune":
            return "P"
        elif method_name == "distill":
            return "D"
        elif method_name == "original":
            return "original"
        return "unknown"

    # Определение типов данных для pool (без speed_coef и memory_coef)
    pool_dtypes = {
        "id": str,
        "config": str,
        "method": str,
        "logs": str,
        "task_type": str,
        "model": str,
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

    # Определение типов данных для leader_board (без speed_coef и memory_coef)
    leader_board_dtypes = {
        "config_id": str,
        "method": str,
        "config": str,
        "task_type": str,
        "metric": str,
        "score": float,
        "model": str
    }

    # Подготовка данных для pool
    pool_id = str(uuid.uuid4())
    method_type = "original" if original else get_method_type(configs)
    config_json = json.dumps(configs, ensure_ascii=False) if configs else "[]"
    logs_json = json.dumps(logs, ensure_ascii=False) if logs else "{}"

    # Разворачивание метрик rouge, bert_score и memory из входных словарей
    rouge_scores = metrics_results.get("rouge", {})
    bert_scores = metrics_results.get("bert_score", {})
    memory_scores = metrics_results.get("memory", {})
    logger.info(f"Получены метрики: rouge={rouge_scores}, bert_score={bert_scores}, memory={memory_scores}")

    # Данные для pool
    pool_data = {
        "id": pool_id,
        "config": config_json,
        "method": method_type,
        "logs": logs_json,
        "task_type": task_type,
        "model": model_name,
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
        if metric in ["id", "config", "method", "logs", "task_type", "model"]:
            continue
        if metric not in applicable_metrics:
            continue
        if metric in ["rouge1", "rouge2", "rougeL"]:
            value = rouge_scores.get(metric, float("inf"))
            pool_data[metric] = None if not np.isfinite(value) else value
        elif metric in ["bert_precision", "bert_recall", "bert_f1"]:
            value = bert_scores.get(f"bertscore_{metric.split('_')[1]}", float("inf"))
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

    # Логирование метрик перед добавлением в pool
    metrics_to_log = {k: v for k, v in pool_data.items() if k in applicable_metrics and v is not None and np.isfinite(v)}
    logger.info(f"Метрики для записи в pool (id={pool_id}, method={method_type}, model={model_name}): {metrics_to_log}")

    # Загрузка или создание pool
    pool_df = pd.DataFrame(columns=list(pool_dtypes.keys())).astype(pool_dtypes)
    if os.path.exists(excel_file):
        try:
            xl = pd.ExcelFile(excel_file, engine="openpyxl")
            logger.info(f"Доступные листы в {excel_file}: {xl.sheet_names}")
            if "pool" in xl.sheet_names:
                pool_df = pd.read_excel(xl, sheet_name="pool", engine="openpyxl")
                pool_df = pool_df.astype({k: v for k, v in pool_dtypes.items() if k in pool_df.columns}, errors="ignore")
                pool_df = pool_df.drop_duplicates(subset=["id"], keep="last")
                logger.info(f"Загружен pool_df с {len(pool_df)} строками")
                logger.info(f"Столбцы pool_df: {list(pool_df.columns)}")
                if "model" in pool_df.columns and "method" in pool_df.columns:
                    logger.info(f"Уникальные значения в столбце model: {pool_df['model'].unique()}")
                    logger.info(f"Уникальные значения в столбце method: {pool_df['method'].unique()}")
                    logger.info(f"Все строки pool_df:\n{pool_df[['id', 'model', 'method', 'latency', 'memory_gpu_peak', 'bert_precision', 'bert_recall', 'bert_f1', 'perplexity']].to_dict()}")
            else:
                logger.warning("Лист 'pool' не найден в файле. Создаётся новый.")
        except Exception as e:
            logger.warning(f"Ошибка при чтении {excel_file}: {str(e)}. Создаётся новый файл.")
            pool_df = pd.DataFrame(columns=list(pool_dtypes.keys())).astype(pool_dtypes)

    # Добавление новой строки в pool
    new_pool_row = pd.DataFrame([pool_data]).astype(pool_dtypes, errors="ignore")
    pool_df = pd.concat([pool_df, new_pool_row], ignore_index=True)

    # Загрузка или создание leader_board
    leader_board_df = pd.DataFrame(columns=list(leader_board_dtypes.keys())).astype(leader_board_dtypes)
    if os.path.exists(excel_file):
        try:
            xl = pd.ExcelFile(excel_file, engine="openpyxl")
            if "leader_board" in xl.sheet_names:
                leader_board_df = pd.read_excel(xl, sheet_name="leader_board", engine="openpyxl")
                leader_board_df = leader_board_df.astype({k: v for k, v in leader_board_dtypes.items() if k in leader_board_df.columns}, errors="ignore")
                leader_board_df = leader_board_df.drop_duplicates(subset=["config_id", "metric"], keep="last")
                if "config_id" in leader_board_df.columns and "method" in leader_board_df.columns:
                    original_ids = pool_df[pool_df["method"] == "original"]["id"].tolist()
                    leader_board_df.loc[leader_board_df["config_id"].isin(original_ids), "method"] = "original"
                    logger.info(f"Исправлены значения method в leader_board для config_id: {original_ids}")
        except Exception as e:
            logger.warning(f"Ошибка при чтении листа leader_board: {str(e)}. Создаётся новый лист.")

    # Подготовка данных для leader_board
    leader_board_data = []
    for metric in pool_dtypes.keys():
        if metric in ["id", "config", "method", "logs", "task_type", "model"]:
            continue
        if metric not in applicable_metrics:
            continue
        score = pool_data.get(metric)
        if score is None or not np.isfinite(score):
            continue
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
            "model": model_name
        })

    # Добавление новых строк в leader_board
    if leader_board_data:
        new_leader_board_rows = pd.DataFrame(leader_board_data).astype(leader_board_dtypes, errors="ignore")
        leader_board_df = pd.concat([leader_board_df, new_leader_board_rows], ignore_index=True)
    else:
        logger.warning("Нет валидных данных для добавления в leader_board.")

    # Сохранение в Excel
    try:
        if os.path.exists(excel_file):
            xl = pd.ExcelFile(excel_file, engine="openpyxl")
            existing_sheets = xl.sheet_names
        else:
            existing_sheets = []

        with pd.ExcelWriter(excel_file, engine="openpyxl", mode="a" if existing_sheets else "w", if_sheet_exists="replace") as writer:
            pool_df.to_excel(writer, sheet_name="pool", index=False)
            leader_board_df.to_excel(writer, sheet_name="leader_board", index=False)
        logger.info(f"Данные успешно сохранены в {excel_file}: pool ({len(pool_df)} строк), leader_board ({len(leader_board_df)} строк)")
    except Exception as e:
        logger.error(f"Ошибка при записи в {excel_file}: {str(e)}")
        raise