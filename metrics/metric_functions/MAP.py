import torch
from typing import List, Tuple
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
        area2 = (x_max2 - x_min2) * (x_max2 - y_min2)
        union_area = area1[:, None] + area2 - inter_area
        iou = inter_area / (union_area + 1e-6)
        return iou
    except Exception as e:
        logger.warning(f"compute_iou_matrix: Ошибка при вычислении: {str(e)}")
        return torch.tensor([])

def compute_map(
        *,
        predictions: List[Tuple[torch.Tensor, torch.Tensor]],
        references: List[Tuple[torch.Tensor, torch.Tensor]],
        iou_threshold: float = 0.5,
        **kwargs
) -> float:
    """Вычисляет mean Average Precision (mAP) для объектного детектирования."""
    try:
        if not predictions or not references or len(predictions) != len(references):
            logger.warning("MAP: Пустые или несоответствующие списки predictions или references")
            return float("inf")

        if not all(isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], torch.Tensor) and p[0].shape[-1] == 4 for p in predictions) or \
           not all(isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], torch.Tensor) and r[0].shape[-1] == 4 for r in references):
            logger.warning("MAP: Некорректный формат predictions или references")
            return float("inf")

        aps = []
        for (pred_boxes, pred_scores), (true_boxes, _) in zip(predictions, references):
            if pred_boxes.shape[0] == 0 or true_boxes.shape[0] == 0:
                aps.append(0.0)
                continue

            sorted_indices = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[sorted_indices]

            iou = compute_iou_matrix(pred_boxes, true_boxes)
            true_positives = torch.zeros(len(pred_boxes), dtype=torch.bool)
            false_positives = torch.zeros(len(pred_boxes), dtype=torch.bool)
            matched = torch.zeros(len(true_boxes), dtype=torch.bool)

            for i in range(len(pred_boxes)):
                max_iou, max_idx = torch.max(iou[i], dim=0)
                if max_iou >= iou_threshold and not matched[max_idx]:
                    true_positives[i] = True
                    matched[max_idx] = True
                else:
                    false_positives[i] = True

            tp_cumsum = torch.cumsum(true_positives, dim=0)
            fp_cumsum = torch.cumsum(false_positives, dim=0)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            recalls = tp_cumsum / len(true_boxes)
            ap = 0.0
            for t in torch.linspace(0, 1, 11):
                prec = precisions[recalls >= t]
                ap += prec.max().item() / 11 if len(prec) > 0 else 0.0
            aps.append(ap)

        return float(torch.tensor(aps).mean().item()) if aps else float("inf")
    except Exception as e:
        logger.warning(f"Ошибка в compute_map: {str(e)}")
        return float("inf")