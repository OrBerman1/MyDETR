from transformers import DetrForObjectDetection, DetrImageProcessor
import torch


def get_detr(num_classes, device):
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", num_labels=num_classes,
            ignore_mismatched_sizes=True)  # Pretrained DETR model
    # in_features = model.class_labels_classifier.in_features  # Get input size of the classifier
    # model.class_labels_classifier = torch.nn.Linear(in_features, num_classes + 1)  # +1 for background
    model.to(device)

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    return model, processor
