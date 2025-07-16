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
    speed_cf: float,
    mem_cf: float,
    logs: Dict[str, Any],
    configs: List[Dict[str, Any]],
    model_name: str,
    excel_file: str = "configs_pool.xlsx",
    author_id: Optional[str] = None
) -> None:
    """
    Aggregates optimization and metrics results into two Excel sheets: 'pool' and 'leader_board'.

    Args:
        metrics_results: Dictionary of metrics from metrics_evaluate function.
        speed_cf: Speed coefficient from optimization.
        mem_cf: Memory coefficient from optimization.
        logs: JSON-serializable dictionary of optimization logs.
        configs: List of configuration dictionaries for the optimization pipeline.
        model_name: Name of the model (e.g., "GPT2").
        excel_file: Path to the Excel file to update.
        author_id: Optional identifier for the author (default: None).

    Returns:
        None: Updates the Excel file in-place.
    """
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

    # Подготовка данных для pool
    pool_id = str(uuid.uuid4())
    method_type = get_method_type(configs)
    config_json = json.dumps(configs)
    logs_json = json.dumps(logs)

    # Разворачивание метрик (rouge и bert_score — словари)
    pool_data = {
        "id": pool_id,
        "author_id": author_id if author_id else "",
        "config": config_json,
        "logs": logs_json,
        "speed_cf": speed_cf,
        "mem_cf": mem_cf,
        "accuracy": metrics_results.get("accuracy", float("inf")),
        "bleu": metrics_results.get("bleu", float("inf")),
        "perplexity": metrics_results.get("perplexity", float("inf")),
        "meteor": metrics_results.get("meteor", float("inf")),
        "iou": metrics_results.get("iou", float("inf")),
        "map": metrics_results.get("map", float("inf")),
        "precision": metrics_results.get("precision", float("inf")),
        "recall": metrics_results.get("recall", float("inf")),
        "f1": metrics_results.get("f1", float("inf")),
        "cider": metrics_results.get("cider", float("inf")),
        "spice": metrics_results.get("spice", float("inf")),
        "clip_score_vision": metrics_results.get("clip_score_vision", float("inf")),
        "latency": metrics_results.get("latency", float("inf")),
        "memory": metrics_results.get("memory", float("inf")),
        "flops": metrics_results.get("flops", float("inf")),
        "throughput": metrics_results.get("throughput", float("inf")),
        "energy": metrics_results.get("energy", float("inf")),
        "ece": metrics_results.get("ece", float("inf")),
        "mce": metrics_results.get("mce", float("inf")),
        "mmlu": metrics_results.get("mmlu", float("inf")),
        "helm": metrics_results.get("helm", float("inf")),
        "glue": metrics_results.get("glue", float("inf")),
    }

    # Разворачивание rouge
    rouge_scores = metrics_results.get("rouge", {})
    pool_data.update({
        "rouge1": rouge_scores.get("rouge1", float("inf")),
        "rouge2": rouge_scores.get("rouge2", float("inf")),
        "rougeL": rouge_scores.get("rougeL", float("inf")),
    })

    # Разворачивание bert_score
    bert_scores = metrics_results.get("bert_score", {})
    pool_data.update({
        "bertscore_precision": bert_scores.get("bertscore_precision", float("inf")),
        "bertscore_recall": bert_scores.get("bertscore_recall", float("inf")),
        "bertscore_f1": bert_scores.get("bertscore_f1", float("inf")),
    })

    # Загрузка существующего Excel файла или создание нового
    xl = pd.ExcelFile(excel_file)
    pool_df = pd.read_excel(xl, sheet_name="pool")

    # Добавление новой строки в pool
    pool_df = pd.concat([pool_df, pd.DataFrame([pool_data])], ignore_index=True)

    # Подготовка данных для leader_board
    leader_board_data = []
    for metric, score in metrics_results.items():
        if metric == "failed_metrics":
            continue
        if isinstance(score, dict):
            for sub_metric, sub_score in score.items():
                if not isinstance(sub_score, (int, float)) or not np.isfinite(sub_score):
                    continue
                leader_board_data.append({
                    "config_id": pool_id,
                    "metod": method_type,
                    "config": config_json,
                    "metric": f"{metric}_{sub_metric}",
                    "score": sub_score,
                    "model": model_name
                })
        else:
            if not isinstance(score, (int, float)) or not np.isfinite(score):
                continue
            leader_board_data.append({
                "config_id": pool_id,
                "metod": method_type,
                "config": config_json,
                "metric": metric,
                "score": score,
                "model": model_name
            })

    # Загрузка или создание leader_board
    try:
        leader_board_df = pd.read_excel(xl, sheet_name="leader_board")
    except (FileNotFoundError, ValueError):
        leader_board_df = pd.DataFrame(columns=["config_id", "metod", "config", "metric", "score", "model"])

    # Добавление новых строк в leader_board
    leader_board_df = pd.concat([leader_board_df, pd.DataFrame(leader_board_data)], ignore_index=True)

    # Сохранение в Excel
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        pool_df.to_excel(writer, sheet_name="pool", index=False)
        leader_board_df.to_excel(writer, sheet_name="leader_board", index=False)

    logger.info(f"Данные успешно сохранены в {excel_file}: pool ({len(pool_df)} строк), leader_board ({len(leader_board_df)} строк)")