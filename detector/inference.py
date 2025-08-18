import sys
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os, random

from model import Model
from YOLOModel import YOLOModel
from FasterRCNN import FasterRCNN


class Inference(Model):
    def __init__(self, model):
        self._model = model
    
    def run(self, image):
        results = self._model.model(image)
        annotated_image = results[0]
        annotated_image.show()
        return results
    
    def run(self, image_path, output_path="output.png", score_threshold=0.6):
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0)

        with torch.no_grad():
            preds = self._model(image_tensor)[0]

        fig, ax = plt.subplots(1, figsize=(16, 12))
        ax.imshow(image)
        count = 0

        for box, score, label in zip(preds["boxes"], preds["scores"], preds["labels"]):
            if score < score_threshold:
                continue
            count += 1
            x1, y1, x2, y2 = box
            color, name = self.colors[label], self.classes[label]

            ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor="none"))
            ax.text(x1, y1-10, f"{name} {score:.2f}", fontsize=12, color="white", bbox=dict(facecolor=color, alpha=0.7, pad=2))

        print(f"- {count} objetos detectados (threshold={score_threshold}).")

        os.makedirs("results", exist_ok=True)
        save_path = os.path.join("results", output_path)
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Resultado salvo em: {save_path}")




def main():
    image = 'images/input_example.jpg'

    yolo_model_path = 'models/v1_yolov8.pt'

    rcnn_model_path = 'models/fasterrcnn_resnet50_fpn_coco.pth'

    # passar o caminho do modelo como argumento
    if len(sys.argv) > 1:
        image = sys.argv[1]

    yolo = YOLOModel(yolo_model_path)

    #inference = Inference(yolo)
    #inference.run(image)

    fasterRCNN = FasterRCNN(rcnn_model_path)

    inference = Inference(fasterRCNN)
    inference.run(image, output_path="detected_output.png", score_threshold=0.6)


if __name__ == '__main__':
    main()