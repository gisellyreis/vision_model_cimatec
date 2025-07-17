import os
import cv2
import random

IMG_DIR = "../../datasets/seadronessee/images/train"
SAVE_DIR = "images"

os.makedirs(SAVE_DIR, exist_ok=True)

image_files = [f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png'))]

sample_files = random.sample(image_files, min(4, len(image_files)))

for i, fname in enumerate(sample_files):
    img_path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(img_path)

    save_path = os.path.join(SAVE_DIR, f"sample_{i+1}_{fname}")
    cv2.imwrite(save_path, img)
    print(f"Image saved in: {save_path}")
