import torch
from typing import List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_iou_matrix(pred_boxes: torch.Tensor, true_boxes: torch.Tensor) -> torch.Tensor:
    """Вычисляет IoU для всех пар предсказанных и истинных боксов."""
    try:
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
    except Exception as e:
        logger.warning(f"compute_iou_matrix: Ошибка при вычислении: {str(e)}")
        return torch.tensor([])

def compute_iou(
        *,
        predictions: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        references: List[Tuple[torch.Tensor, torch.Tensor]],
        **kwargs
) -> float:
    """Вычисляет средний IoU между предсказанными и истинными bounding boxes."""
    try:
        if not predictions or not references or len(predictions) != len(references):
            logger.warning("IoU: Пустые или несоответствующие списки predictions или references")
            return float("inf")

        if not all(isinstance(p, tuple) and len(p) == 3 and isinstance(p[0], torch.Tensor) and p[0].shape[-1] == 4 for p in predictions) or \
           not all(isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], torch.Tensor) and r[0].shape[-1] == 4 for r in references):
            logger.warning("IoU: Некорректный формат predictions или references")
            return float("inf")

        ious = []
        for (pred_boxes, _, _), (true_boxes, _) in zip(predictions, references):
            if pred_boxes is None or true_boxes is None or pred_boxes.shape[0] == 0 or true_boxes.shape[0] == 0:
                ious.append(0.0)
                continue

            iou = compute_iou_matrix(pred_boxes, true_boxes)
            max_iou, _ = torch.max(iou, dim=1)
            ious.extend(max_iou.tolist())

        return float(torch.tensor(ious).mean().item()) if ious else float("inf")
    except Exception as e:
        logger.warning(f"Ошибка в compute_iou: {str(e)}")
        return float("inf")