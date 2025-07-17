import os
import random
from PIL import Image
import matplotlib.pyplot as plt

from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

img_folder = '../../datasets/seadronessee/images/test'
save_folder = 'inference_pretrained_images'

os.makedirs(save_folder, exist_ok=True)

image_files = [f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png'))]
sample_imgs = random.sample(image_files, min(9, len(image_files)))

for i, img_file in enumerate(sample_imgs):
    img_path = os.path.join(img_folder, img_file)
    results = model(img_path)

    plotted_img = results[0].plot()

    img = Image.fromarray(plotted_img)
    save_path = os.path.join(save_folder, f"inference_{i+1}_{img_file}")
    img.save(save_path)

print(f"Saved inference images to: {save_folder}")
