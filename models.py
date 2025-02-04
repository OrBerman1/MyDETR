import torchvision
import os
from transformers import DetrImageProcessor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from coco_eval import CocoEvaluator
from pycocotools.coco import COCO
from metrics import get_class_ar, get_class_ap
from pycocotools.cocoeval import COCOeval


class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, num_classes=2):
        super().__init__()
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                            revision="no_timm",
                                                            num_labels=num_classes,
                                                            ignore_mismatched_sizes=True)
        self.num_classes = num_classes
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.outputs = []

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in
                  batch["labels"]]  # these are in DETR format, resized + normalized

        # forward pass
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0)

        predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
        predictions = prepare_for_coco_detection(predictions)
        self.outputs.append(predictions)

        return predictions

    def on_validation_epoch_end(self):
        evaluator = CocoEvaluator(coco_gt=self.val_dataloader().dataset.coco, iou_types=["bbox"])
        for o in self.outputs:
            evaluator.update(o)
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()

        # Create the COCOeval object
        coco_eval = evaluator.coco_eval["bbox"]

        # Optionally, you can evaluate the result using coco_eval
        ar_per_category = get_class_ar(coco_eval, self.num_classes)
        ap_per_category = get_class_ap(coco_eval, self.num_classes)
        print(f'Average Precision (AP) per category: {ap_per_category}')
        print(f'Average Recall (AR) per category: {ar_per_category}')

        self.outputs = []
        # self.log("avg_val_loss", avg_loss)
        # print(f"\nEpoch {self.current_epoch}: Avg Validation Loss = {avg_loss:.4f}")

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                      weight_decay=self.weight_decay)

        return optimizer