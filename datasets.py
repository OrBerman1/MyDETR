import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision.ops import box_convert


DETR_FORMAT = 'xywh'


class DTERDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, processor=None, bbox_format="xywh", normalized=True,
                 train=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.processor = processor
        self.bbox_format = bbox_format
        self.normalized = normalized   # if the bboxes are already normalized
        self.train = train
        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        width, height = image.size

        # Load label
        label_path = os.path.join(self.label_dir, self.labels[idx])
        boxes = []
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                l, a, b, c, d = map(float, line.strip().split())
                labels.append(l)    #  + 1)
                if self.bbox_format != DETR_FORMAT:
                    a, b, c, d = self.convert_to_xywh([a, b, c, d], width, height)
                if not self.train:  # inference format should be xyxy
                    c += a
                    d += b
                boxes.append([a, b, c, d])

        # Convert to tensor
        if self.processor:
            annotations = {
                'image_id': idx,  # Unique ID for the image (could be anything, here we use 0)
                'annotations': [
                    {
                        'bbox': box,
                        'category_id': label,
                        'area': box[3] * box[2]
                    }
                    for box, label in zip(boxes, labels)
                ]
            }
            encoding = self.processor(images=image, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        if not self.train:
            target["boxes"] = torch.tensor(boxes)

        if self.transform:
            print("transform is not supported at the moment!")
            # image = self.transform(image)

        return pixel_values, target

    def convert_to_xywh(self, bbox, image_width, image_height):
        if self.bbox_format != DETR_FORMAT:
            bbox = box_convert(torch.tensor(bbox).unsqueeze(0), self.bbox_format, DETR_FORMAT)

        bbox = bbox.squeeze(0)
        x, y, width, height = bbox

        if self.normalized:
            x *= image_width
            y *= image_height
            width *= image_width
            height *= image_height

        return [x.item(), y.item(), width.item(), height.item()]
