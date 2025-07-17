# Convert COCO (json) to YOLO (txt) annotations

import os
import json
from collections import defaultdict

annotations_dir = '../../datasets/seadronessee/annotations'
images_base_dir = '../../datasets/seadronessee/images'
labels_base_dir = '../../datasets/seadronessee/labels'

os.makedirs(os.path.join(labels_base_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(labels_base_dir, 'val'), exist_ok=True)

def convert_json_to_txt(json_path, split):
    with open(json_path, 'r') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco['images']}
    category_map = {cat['id']: i for i, cat in enumerate(coco['categories'])}

    annotations_by_image = defaultdict(list)
    for ann in coco['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    for image_id, anns in annotations_by_image.items():
        image_info = images[image_id]
        w, h = image_info['width'], image_info['height']
        file_stem = os.path.splitext(image_info['file_name'])[0]
        label_path = os.path.join(labels_base_dir, split, file_stem + '.txt')

        with open(label_path, 'w') as f:
            for ann in anns:
                cat_id = ann['category_id']
                cls_id = category_map[cat_id]
                x, y, bw, bh = ann['bbox']
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                bw /= w
                bh /= h
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

convert_json_to_txt(os.path.join(annotations_dir, 'instances_train.json'), 'train')
convert_json_to_txt(os.path.join(annotations_dir, 'instances_val.json'), 'val')

print("Conversion ended!")