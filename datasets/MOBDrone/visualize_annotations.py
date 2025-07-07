import os
import json
import random
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pycocotools.coco import COCO

dataset_dir = r"E:\aero"  # <- Raw string to avoid issues with backslashes
images_dir = os.path.join(dataset_dir, 'images')
annotations_path = os.path.join(dataset_dir, 'filtered_annotations.json')

num_images_to_show = 100

coco = COCO(annotations_path)

img_ids = coco.getImgIds()
random.shuffle(img_ids)
selected_ids = img_ids[:num_images_to_show]

current_index = [0]

def show_image(index):
    plt.clf()
    img_info = coco.loadImgs(selected_ids[index])[0]
    image_path = os.path.join(images_dir, img_info['file_name'])

    image = cv2.imread(image_path)
    if image is None:
        print(f"Imagem não encontrada: {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ann_ids = coco.getAnnIds(imgIds=img_info['id'])
    annotations = coco.loadAnns(ann_ids)

    plt.imshow(image)
    for ann in annotations:
        x, y, w, h = ann['bbox']
        cat_id = ann['category_id']
        cat_name = coco.loadCats(cat_id)[0]['name']
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='lime', facecolor='none', linewidth=1))
        plt.text(x, y - 5, cat_name, color='yellow', fontsize=8 )

    plt.title(f"{img_info['file_name']}  ({index+1}/{len(selected_ids)})")
    plt.axis('off')
    plt.draw()

def on_key(event):
    if event.key == 'right':
        current_index[0] = (current_index[0] + 1) % len(selected_ids)
        show_image(current_index[0])
    elif event.key == 'left':
        current_index[0] = (current_index[0] - 1) % len(selected_ids)
        show_image(current_index[0])

fig = plt.figure("Visualizador MOBDrone")
fig.canvas.mpl_connect('key_press_event', on_key)
show_image(current_index[0])
plt.show()
