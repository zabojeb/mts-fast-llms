import torch
from typing import List, Tuple

def compute_precision(
    *,
    predictions: List[Tuple[torch.Tensor, torch.Tensor]],
    references: List[Tuple[torch.Tensor, torch.Tensor]],
    iou_threshold: float = 0.5,
    **kwargs
) -> float:
    """Вычисляет Precision для объектного детектирования."""
    if not predictions or not references:
        return 0.0

    tp_total = 0
    fp_total = 0
    for (pred_boxes, pred_scores), (true_boxes, _) in zip(predictions, references):
        if pred_boxes.shape[0] == 0:
            continue
        matched = torch.zeros(len(true_boxes), dtype=torch.bool)
        for pred_box in pred_boxes:
            max_iou = 0.0
            max_idx = -1
            for j, true_box in enumerate(true_boxes):
                if matched[j]:
                    continue
                x_min1, y_min1, x_max1, y_max1 = pred_box
                x_min2, y_min2, x_max2, y_max2 = true_box
                x_min_inter = max(x_min1, x_min2)
                y_min_inter = max(y_min1, y_min2)
                x_max_inter = min(x_max1, x_max2)
                y_max_inter = min(y_max1, y_max2)
                inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
                area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
                area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0.0
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            if max_iou >= iou_threshold and max_idx >= 0 and not matched[max_idx]:
                tp_total += 1
                matched[max_idx] = True
            else:
                fp_total += 1

    return tp_total / (tp_total + fp_total + 1e-6)


def compute_recall(
    *,
    predictions: List[Tuple[torch.Tensor, torch.Tensor]],
    references: List[Tuple[torch.Tensor, torch.Tensor]],
    iou_threshold: float = 0.5,
    **kwargs
) -> float:
    """Вычисляет Recall для объектного детектирования."""
    if not predictions or not references:
        return 0.0

    tp_total = 0
    n_true_total = sum(len(true_boxes) for _, (true_boxes, _) in references)
    for (pred_boxes, pred_scores), (true_boxes, _) in zip(predictions, references):
        if pred_boxes.shape[0] == 0:
            continue
        matched = torch.zeros(len(true_boxes), dtype=torch.bool)
        for pred_box in pred_boxes:
            max_iou = 0.0
            max_idx = -1
            for j, true_box in enumerate(true_boxes):
                if matched[j]:
                    continue
                x_min1, y_min1, x_max1, y_max1 = pred_box
                x_min2, y_min2, x_max2, y_max2 = true_box
                x_min_inter = max(x_min1, x_min2)
                y_min_inter = max(y_min1, y_min2)
                x_max_inter = min(x_max1, x_max2)
                y_max_inter = min(y_max1, y_max2)
                inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
                area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
                area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0.0
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            if max_iou >= iou_threshold and max_idx >= 0 and not matched[max_idx]:
                tp_total += 1
                matched[max_idx] = True

    return tp_total / (n_true_total + 1e-6)


def compute_f1(
    *,
    predictions: List[Tuple[torch.Tensor, torch.Tensor]],
    references: List[Tuple[torch.Tensor, torch.Tensor]],
    iou_threshold: float = 0.5,
    **kwargs
) -> float:
    """Вычисляет F1 Score для объектного детектирования."""
    precision = compute_precision(predictions=predictions, references=references, iou_threshold=iou_threshold)
    recall = compute_recall(predictions=predictions, references=references, iou_threshold=iou_threshold)
    return 2 * (precision * recall) / (precision + recall + 1e-6) if (precision + recall) > 0 else 0.0