import json
import os

base_dir = r"E:\aero"
original_json = os.path.join(base_dir, "annotations_3_coco_classes.json")
filtered_json = os.path.join(base_dir, "filtered_annotations.json")
images_dir = os.path.join(base_dir, "images")

image_files = set(os.listdir(images_dir))

with open(original_json, "r") as f:
    coco_data = json.load(f)

filtered_images = [img for img in coco_data['images'] if img['file_name'] in image_files]
filtered_image_ids = set(img['id'] for img in filtered_images)

filtered_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in filtered_image_ids]

filtered_coco = {
    "images": filtered_images,
    "annotations": filtered_annotations,
    "categories": coco_data['categories']
}

with open(filtered_json, "w") as f:
    json.dump(filtered_coco, f)

print(f"Novo arquivo salvo como {filtered_json} com {len(filtered_images)} imagens")
