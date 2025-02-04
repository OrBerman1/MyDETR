import os
import json
import glob
from PIL import Image

# Define paths
yolo_dataset_path = "PersonCar/valid"
coco_output_path = "PersonCar/valid/annotations.json"

# Category mapping (modify according to your classes)
# category_mapping = {0: "class1", 1: "class2", 2: "class3"}


def yolo_to_coco(yolo_path, output_json):
    images = []
    annotations = []
    categories = []
    ann_id = 0

    # Create categories list
    # for class_id, class_name in category_mapping.items():
    #     categories.append({"id": class_id, "name": class_name, "supercategory": "none"})

    # Get image files
    image_files = glob.glob(os.path.join(yolo_path, "images", "*.jpg"))
    class_ids = []
    for img_id, img_path in enumerate(image_files):
        img = Image.open(img_path)
        width, height = img.size
        file_name = os.path.basename(img_path)

        # Add image info
        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        # Read corresponding YOLO annotation
        txt_path = os.path.join(yolo_path, "labels", file_name.replace(".jpg", ".txt"))
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    values = line.strip().split()
                    class_id = int(values[0])
                    class_ids.append(class_id)
                    x_center, y_center, bbox_width, bbox_height = map(float, values[1:])

                    # Convert YOLO format (relative) to COCO format (absolute)
                    x_min = (x_center - bbox_width / 2) * width
                    y_min = (y_center - bbox_height / 2) * height
                    abs_width = bbox_width * width
                    abs_height = bbox_height * height

                    # Add annotation
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": class_id,
                        "bbox": [x_min, y_min, abs_width, abs_height],
                        "area": abs_width * abs_height,
                        "iscrowd": 0
                    })
                    ann_id += 1

    class_ids = list(set(class_ids))
    for class_id in class_ids:
        categories.append({"id": class_id, "name": f"class_{class_id}", "supercategory": "none"})

    # Create final COCO dataset
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Save JSON file
    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)
    print(f"COCO annotations saved to {output_json}")


# Convert the dataset
yolo_to_coco(yolo_dataset_path, coco_output_path)