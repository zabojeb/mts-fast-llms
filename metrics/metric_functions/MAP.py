import torch
from typing import List, Tuple

def compute_map(
    *,
    predictions: List[Tuple[torch.Tensor, torch.Tensor]],  # [(boxes, scores), ...]
    references: List[Tuple[torch.Tensor, torch.Tensor]],  # [(boxes, labels), ...]
    iou_threshold: float = 0.5,
    **kwargs
) -> float:
    """Вычисляет mean Average Precision (mAP) для объектного детектирования.

    Note:
        Assumes single-class or class-agnostic detection. For multi-class mAP, extend to group by true_labels.
    """
    if not predictions or not references:
        return 0.0

    aps = []
    for (pred_boxes, pred_scores), (true_boxes, true_labels) in zip(predictions, references):
        if pred_boxes.shape[0] == 0 or true_boxes.shape[0] == 0:
            aps.append(0.0)
            continue

        # Сортировка предсказаний по убыванию scores
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]

        true_positives = torch.zeros(len(pred_boxes), dtype=torch.bool)
        false_positives = torch.zeros(len(pred_boxes), dtype=torch.bool)
        n_true = len(true_boxes)

        matched = torch.zeros(n_true, dtype=torch.bool)

        for i, pred_box in enumerate(pred_boxes):
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

            if max_iou >= iou_threshold and max_idx >= 0:
                if not matched[max_idx]:
                    true_positives[i] = True
                    matched[max_idx] = True
                else:
                    false_positives[i] = True
            else:
                false_positives[i] = True

        # Вычисление precision и recall
        tp_cumsum = torch.cumsum(true_positives, dim=0)
        fp_cumsum = torch.cumsum(false_positives, dim=0)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / n_true

        # Интерполяция precision для 11 точек recall
        ap = 0.0
        for t in torch.linspace(0, 1, 11):
            prec = precisions[recalls >= t]
            ap += prec.max().item() / 11 if len(prec) > 0 else 0.0
        aps.append(ap)

    return float(torch.tensor(aps).mean().item()) if aps else 0.0