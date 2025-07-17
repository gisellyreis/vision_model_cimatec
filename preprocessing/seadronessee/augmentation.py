import os
import cv2
import random
import albumentations as A
import numpy as np

img_dir = '../../datasets/seadronessee/images/train'
label_dir = '../../datasets/seadronessee/labels/train'
save_dir = 'images/augmentation'
os.makedirs(save_dir, exist_ok=True)

image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
sample_imgs = random.sample(image_files, min(4, len(image_files)))

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.4),
    A.Rotate(limit=45, p=0.5),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_ids']))

def yolo_to_voc(bbox, w, h):
    x, y, bw, bh = bbox
    x_min = max((x - bw / 2) * w, 0)
    y_min = max((y - bh / 2) * h, 0)
    x_max = min((x + bw / 2) * w, w)
    y_max = min((y + bh / 2) * h, h)
    return [x_min, y_min, x_max, y_max]

def draw_boxes(img, boxes, color=(0, 255, 0)):
    img_copy = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
    return img_copy

for img_file in sample_imgs:
    img_path = os.path.join(img_dir, img_file)
    label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    bboxes = []
    class_ids = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                cls, x, y, bw, bh = map(float, line.strip().split())
                bboxes.append(yolo_to_voc((x, y, bw, bh), w, h))
                class_ids.append(int(cls))

    augmented = transform(image=image, bboxes=bboxes, class_ids=class_ids)
    aug_img = augmented['image']
    aug_boxes = augmented['bboxes']

    original_with_boxes = draw_boxes(image, bboxes, color=(0, 255, 0))
    augmented_with_boxes = draw_boxes(aug_img, aug_boxes, color=(255, 0, 0))

    gap = 20
    height = original_with_boxes.shape[0]
    gap_img = np.ones((height, gap, 3), dtype=np.uint8) * 255 
    side_by_side = np.concatenate((original_with_boxes, gap_img, augmented_with_boxes), axis=1)


    out_path = os.path.join(save_dir, f'aug_{os.path.splitext(img_file)[0]}.jpg')
    cv2.imwrite(out_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
    print(f"Saved: {out_path}")
