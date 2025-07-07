import json
import os

dataset_dir = r"E:\aero"
images_dir = os.path.join(dataset_dir, 'images')
annotations_path = os.path.join(dataset_dir, 'annotations_3_coco_classes.json')

with open(annotations_path, 'r') as f:
    coco_data = json.load(f)

annotated_image_ids = set(ann['image_id'] for ann in coco_data['annotations'])

unannotated_images = [img['file_name'] for img in coco_data['images'] if img['id'] not in annotated_image_ids]

deleted_count = 0
for fname in unannotated_images:
    image_path = os.path.join(images_dir, fname)
    if os.path.exists(image_path):
        os.remove(image_path)
        deleted_count += 1

print(f"✅ Deleted {deleted_count} unannotated images.")
