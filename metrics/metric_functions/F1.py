import torch
from typing import List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_iou_matrix(pred_boxes: torch.Tensor, true_boxes: torch.Tensor) -> torch.Tensor:
    """Вычисляет IoU для всех пар предсказанных и истинных боксов."""
    x_min1, y_min1, x_max1, y_max1 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    x_min2, y_min2, x_max2, y_max2 = true_boxes[:, 0], true_boxes[:, 1], true_boxes[:, 2], true_boxes[:, 3]
    x_min_inter = torch.max(x_min1[:, None], x_min2)
    y_min_inter = torch.max(y_min1[:, None], y_min2)
    x_max_inter = torch.min(x_max1[:, None], x_max2)
    y_max_inter = torch.min(y_max1[:, None], y_max2)
    inter_area = torch.clamp(x_max_inter - x_min_inter, min=0) * torch.clamp(y_max_inter - y_min_inter, min=0)
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    union_area = area1[:, None] + area2 - inter_area
    iou = inter_area / (union_area + 1e-6)
    return iou

def compute_matches(pred_boxes: torch.Tensor, true_boxes: torch.Tensor, iou_threshold: float) -> Tuple[int, int, int]:
    """Вычисляет TP, FP и общее количество истинных боксов."""
    if pred_boxes.shape[0] == 0:
        return 0, 0, len(true_boxes)
    iou = compute_iou_matrix(pred_boxes, true_boxes)
    matched = torch.zeros(len(true_boxes), dtype=torch.bool)
    tp = 0
    fp = 0
    for i in range(len(pred_boxes)):
        max_iou, max_idx = torch.max(iou[i], dim=0)
        if max_iou >= iou_threshold and not matched[max_idx]:
            tp += 1
            matched[max_idx] = True
        else:
            fp += 1
    return tp, fp, len(true_boxes)

def compute_precision(
        *,
        predictions: List[Tuple[torch.Tensor, torch.Tensor]],
        references: List[Tuple[torch.Tensor, torch.Tensor]],
        iou_threshold: float = 0.5,
        **kwargs
) -> float:
    """Вычисляет Precision для объектного детектирования."""
    try:
        if not predictions or not references or len(predictions) != len(references):
            logger.warning("Precision: Пустые или несоответствующие списки predictions или references")
            return float("inf")

        if not all(isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], torch.Tensor) and p[0].shape[-1] == 4 for p in predictions) or \
           not all(isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], torch.Tensor) and r[0].shape[-1] == 4 for r in references):
            logger.warning("Precision: Некорректный формат predictions или references")
            return float("inf")

        tp_total = 0
        fp_total = 0
        for (pred_boxes, _), (true_boxes, _) in zip(predictions, references):
            tp, fp, _ = compute_matches(pred_boxes, true_boxes, iou_threshold)
            tp_total += tp
            fp_total += fp

        return tp_total / (tp_total + fp_total + 1e-6) if (tp_total + fp_total) > 0 else float("inf")
    except Exception as e:
        logger.warning(f"Precision: Ошибка при вычислении: {str(e)}")
        return float("inf")

def compute_recall(
        *,
        predictions: List[Tuple[torch.Tensor, torch.Tensor]],
        references: List[Tuple[torch.Tensor, torch.Tensor]],
        iou_threshold: float = 0.5,
        **kwargs
) -> float:
    """Вычисляет Recall для объектного детектирования."""
    try:
        if not predictions or not references or len(predictions) != len(references):
            logger.warning("Recall: Пустые или несоответствующие списки predictions или references")
            return float("inf")

        if not all(isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], torch.Tensor) and p[0].shape[-1] == 4 for p in predictions) or \
           not all(isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], torch.Tensor) and r[0].shape[-1] == 4 for r in references):
            logger.warning("Recall: Некорректный формат predictions или references")
            return float("inf")

        tp_total = 0
        n_true_total = 0
        for (pred_boxes, _), (true_boxes, _) in zip(predictions, references):
            tp, _, n_true = compute_matches(pred_boxes, true_boxes, iou_threshold)
            tp_total += tp
            n_true_total += n_true

        return tp_total / (n_true_total + 1e-6) if n_true_total > 0 else float("inf")
    except Exception as e:
        logger.warning(f"Recall: Ошибка при вычислении: {str(e)}")
        return float("inf")

def compute_f1(
        *,
        predictions: List[Tuple[torch.Tensor, torch.Tensor]],
        references: List[Tuple[torch.Tensor, torch.Tensor]],
        iou_threshold: float = 0.5,
        **kwargs
) -> float:
    """Вычисляет F1 Score для объектного детектирования."""
    try:
        precision = compute_precision(predictions=predictions, references=references, iou_threshold=iou_threshold)
        recall = compute_recall(predictions=predictions, references=references, iou_threshold=iou_threshold)
        return 2 * (precision * recall) / (precision + recall + 1e-6) if (precision + recall) > 0 else float("inf")
    except Exception as e:
        logger.warning(f"F1: Ошибка при вычислении: {str(e)}")
        return float("inf")