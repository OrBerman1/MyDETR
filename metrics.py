from torchmetrics.detection import MeanAveragePrecision
import torch


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


def calculate_ap_and_map(pred_boxes, pred_scores, pred_labels, target):
    """
    Calculate Average Precision (AP) and Mean Average Precision (mAP) for object detection.

    Args:
    - pred_boxes: Predicted bounding boxes (Tensor of shape [batch_size, num_predictions, 4]).
    - pred_scores: Predicted class scores (Tensor of shape [batch_size, num_predictions, num_classes]).
    - pred_labels: Predicted labels (Tensor of shape [batch_size, num_predictions]).
    - true_boxes: Ground truth bounding boxes (Tensor of shape [batch_size, num_ground_truths, 4]).
    - true_labels: Ground truth labels (Tensor of shape [batch_size, num_ground_truths]).

    Returns:
    - AP and mAP scores.
    """
    # Initialize the MeanAveragePrecision metric
    metric = MeanAveragePrecision(iou_type="bbox")

    # Prepare the data for evaluation
    # We need to create a list of dictionaries with `boxes` and `labels` for both predictions and ground truths
    pred = [{
        "boxes": pred_boxes[i],
        "scores": pred_scores[i],
        "labels": pred_labels[i]
    } for i in range(len(pred_boxes))]

    # target = [{
    #     "boxes": true_boxes[i],
    #     "labels": true_labels[i]
    # } for i in range(true_boxes.shape[0])]

    # Update the metric with the predictions and targets
    metric.update(pred, target)

    # Compute the mAP and AP
    results = metric.compute()

    # Extract the mAP score (mean AP over all classes)
    mAP = results['map']

    print(f"mAP: {mAP:.4f}")
    try:
        for class_id, ap in enumerate(results['map_per_class']):
            print(f"Class {class_id} AP: {ap:.4f}")
    except TypeError:
        print("type error")

    return results
