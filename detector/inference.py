from datetime import datetime
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

from torchvision import transforms
import torchvision.transforms as T

from model import Model
from YOLOModel import YOLOModel
from FasterRCNNModel import FasterRCNNModel


class Inference(Model):
    def __init__(self, model):
        self._model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def run(self, image):
        if self._model.name == "YOLO":
            return self.run_yolo(image)
        else:
            return self.run_cnn(image)
    
    def run_yolo(self, image_path):
        results = self._model.model(image_path)
        annotated_image = results[0]
        annotated_image.show()

        os.makedirs("detections", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(image_path)
        save_path = os.path.join("detections", f"detection_{timestamp}_{filename}")
        annotated_image.save(filename=save_path)
        print(f"Resultado salvo em: {save_path}")
        return results
    
    def run_cnn(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
        output_path="output.png"
        score_threshold=0.8

        with torch.no_grad():
            preds = self._model.model(image_tensor)[0]

        fig, ax = plt.subplots(1, figsize=(16, 12))
        ax.imshow(image)
        count = 0

        for box, score, label in zip(preds["boxes"], preds["scores"], preds["labels"]):
            score = score.item()
            label = label.item()
            box = box.to("cpu").numpy()

            if score < score_threshold:
                continue
            
            if label == 3:
                continue
            count += 1
            x1, y1, x2, y2 = box
            color, name = self._model.colors[label], self._model.classes[label]


            ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor="none"))
            ax.text(x1, y1-10, f"{name} {score:.2f}", fontsize=12, color="white", bbox=dict(facecolor=color, alpha=0.7, pad=2))

        print(f"- {count} objetos detectados (threshold={score_threshold}).")

        os.makedirs("detections", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(image_path)
        save_path = os.path.join("detections", f"detection_{timestamp}_{filename}")
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Resultado salvo em: {save_path}")

        




def main():
    image = 'images/input_example.jpg'

    yolo_model_path = 'models/v1_yolov10.pt'

    rcnn_model_path = 'models/checkpoint_epoch2.pth'

    # passar o caminho do modelo como argumento
    if len(sys.argv) > 1:
        image = sys.argv[1]

    yolo = YOLOModel(yolo_model_path)

    inference = Inference(yolo)
    inference.run(image)

    # fasterRCNN = FasterRCNNModel(rcnn_model_path)

    # inference = Inference(fasterRCNN)
    # inference.run(image)


if __name__ == '__main__':
    main()