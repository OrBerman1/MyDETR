from torchmetrics.detection import MeanAveragePrecision
import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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


def _convert_target_to_coco(target):
    ground_truths = []
    for i, t in enumerate(target):
        boxes = t["boxes"]
        labels = t["labels"]
        for j, (bb, l) in enumerate(zip(boxes, labels)):
            box = bb.tolist()
            l = l.item()
            area = (box[3] - box[1]) * (box[2] - box[0])
            box[2] -= box[0]
            box[3] -= box[1]
            d = {"id": j, "image_id": i, 'category_id': l, 'bbox': box, 'area': area, "iscrowd": 0}
            ground_truths.append(d)
    return ground_truths


def _convert_preds_to_coco(pred_boxes, pred_scores, pred_labels):
    predictions = []
    for i, (boxes, labels, scores) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        for bb, l, s in zip(boxes, labels, scores):
            box = bb.tolist()
            l = l.item()
            score = s.item()
            box[2] -= box[0]
            box[3] -= box[1]
            d = {"image_id": i, 'category_id': l, 'bbox': box, 'score': score}
            predictions.append(d)
    return predictions


def _get_class_ap(coco_eval, coco_gt):
    ap_per_category = {}
    precisions = coco_eval.eval["precision"]
    for i in range(len(coco_gt.dataset["categories"])):
        precision = precisions[:, :, i, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        ap_per_category[i + 1] = ap * 100
    return ap_per_category


def _get_class_ar(coco_eval, coco_gt):
    ar_per_category = {}
    recalls = coco_eval.eval["recall"]
    for i in range(len(coco_gt.dataset["categories"])):
        recall = recalls[:, i, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        ar_per_category[i + 1] = ar * 100
    return ar_per_category


def get_class_ap(coco_eval, num_classes):
    ap_per_category = {}
    precisions = coco_eval.eval["precision"]
    for i in range(num_classes):
        precision = precisions[:, :, i, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        ap_per_category[i + 1] = ap * 100
    return ap_per_category


def get_class_ar(coco_eval, num_classes):
    ar_per_category = {}
    recalls = coco_eval.eval["recall"]
    for i in range(num_classes):
        recall = recalls[:, i, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        ar_per_category[i + 1] = ar * 100
    return ar_per_category


def calculate_ap(pred_boxes, pred_scores, pred_labels, target, iou_threshold=0.5):
    """
    Calculate Average Precision (AP) using COCO evaluation.

    :param predictions: List of prediction dictionaries in COCO format.
    :param ground_truths: List of ground truth dictionaries in COCO format.
    :param iou_threshold: The IOU threshold to consider a detection correct (default is 0.5).
    :return: Average Precision (AP) for each category and Mean Average Precision (MAP).
    """
    ground_truths = _convert_target_to_coco(target)
    predictions = _convert_preds_to_coco(pred_boxes, pred_scores, pred_labels)
    # Load ground truth data
    image_ids = list(set(gt['image_id'] for gt in ground_truths))

    # Extract unique categories
    category_ids = list(set(gt['category_id'] for gt in ground_truths))
    categories = [{'id': cat_id, 'name': f'category_{cat_id}', "supercategory": "none"} for cat_id in category_ids]

    ds = {
        'images': [{'id': img_id} for img_id in image_ids],  # Image metadata
        'annotations': ground_truths,  # Ground truth bounding boxes
        'categories': categories,  # Categories information
    }

    # Construct a proper COCO-formatted dataset
    coco_gt = COCO()
    coco_gt.dataset = ds
    coco_gt.createIndex()

    # Load predictions
    coco_dt = coco_gt.loadRes(predictions)

    # Initialize COCOeval object
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    # Evaluate the predictions
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Return the MAP (mean AP)
    mean_ap = coco_eval.stats[0] * 100  # MAP is the first metric in the stats list.

    # Return the individual AP for each category as well
    ap_per_category = _get_class_ap(coco_eval, coco_gt)
    ar_per_category = _get_class_ar(coco_eval, coco_gt)

    print(f'Mean Average Precision (MAP): {mean_ap}')
    print(f'Average Precision (AP) per category: {ap_per_category}')
    print(f'Average Recall (AR) per category: {ar_per_category}')

    return mean_ap, ap_per_category
