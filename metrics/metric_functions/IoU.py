import torch
from typing import List, Tuple, Optional

def compute_iou(
    *,
    predictions: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],  # [(boxes, scores, object_embeds), ...]
    references: List[Tuple[torch.Tensor, torch.Tensor]],  # [(boxes, labels), ...]
    **kwargs
) -> float:
    """Вычисляет средний IoU между предсказанными и истинными bounding boxes."""
    if not predictions or not references:
        return 0.0

    ious = []
    for (pred_boxes, _, _), (true_boxes, _) in zip(predictions, references):
        if pred_boxes is None or true_boxes is None or pred_boxes.shape[0] == 0 or true_boxes.shape[0] == 0:
            ious.append(0.0)
            continue

        # Проверка формата boxes: [N, 4] или [M, 4]
        if pred_boxes.shape[-1] != 4 or true_boxes.shape[-1] != 4:
            raise ValueError(f"Invalid box format: pred_boxes shape {pred_boxes.shape}, true_boxes shape {true_boxes.shape}")

        # Вычисление IoU для каждой пары pred_box и true_box
        for pred_box in pred_boxes:
            max_iou = 0.0
            for true_box in true_boxes:
                # Координаты: [x_min, y_min, x_max, y_max]
                x_min1, y_min1, x_max1, y_max1 = pred_box
                x_min2, y_min2, x_max2, y_max2 = true_box

                # Площадь пересечения
                x_min_inter = max(x_min1, x_min2)
                y_min_inter = max(y_min1, y_min2)
                x_max_inter = min(x_max1, x_max2)
                y_max_inter = min(y_max1, y_max2)

                inter_area = max(0.0, x_max_inter - x_min_inter) * max(0.0, y_max_inter - y_min_inter)

                # Площадь объединения
                area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
                area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
                union_area = area1 + area2 - inter_area

                # IoU
                iou = inter_area / union_area if union_area > 0 else 0.0
                max_iou = max(max_iou, iou)

            ious.append(max_iou)

    return float(torch.tensor(ious).mean().item()) if ious else 0.0