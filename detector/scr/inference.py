import argparse
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

from model.model import Model
from model.YOLOModel import YOLOModel
from model.FasterRCNNModel import FasterRCNNModel


class Inference(Model):
    def __init__(self, model):
        self.model = model  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def run(self, image_path, score_threshold=0.5):
        
        results = self.model.predict(image_path)

        image = Image.open(image_path).convert("RGB")
        fig, ax = plt.subplots(1, figsize=(16, 12))
        ax.imshow(image)

        count = 0
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            if score < score_threshold:
                continue
            
            if label.lower() in ["other", "others"]:
                continue

            count += 1
            x1, y1, x2, y2 = box
            
            if hasattr(self.model, 'colors'):
                if isinstance(label, str):
                    if hasattr(self.model, 'classes') and label in self.model.classes:
                        color_idx = self.model.classes.index(label)
                        color = self.model.colors[color_idx] if color_idx < len(self.model.colors) else [1, 0, 0]
                    else:
                        color = [1, 0, 0]  # Default red
                else:
                    color = self.model.colors[label] if label < len(self.model.colors) else [1, 0, 0]
            else:
                color = [1, 0, 0]  # Default red

            ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                           linewidth=2, edgecolor=color, facecolor="none"))
            ax.text(x1, y1-10, f"{label} {score:.2f}", fontsize=12, color="white",
                    bbox=dict(facecolor=color, alpha=0.7, pad=2))

        os.makedirs("../detections", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(image_path)
        save_path = os.path.join("../detections", f"{self.model.name}_detection_{timestamp}_{filename}")
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Resultado salvo em: {save_path}")       


def main():
    parser = argparse.ArgumentParser(description="Model Inference.")
    parser.add_argument("--model", type=str, default="v8", help="Model name (v5, v8, v10, rcnn)")
    parser.add_argument("--image", type=str, default="../images/input_example.jpg", help="Image Path ex: ../images/input_example.jpg")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")

    args = parser.parse_args()

    model_paths = {
        "v5": "../models/v1_yolov5.pt",
        "v8": "../models/v1_yolov8.pt",
        "v10": "../models/v1_yolov10.pt",
        "rcnn": "../models/v1_fasterRCNN.pth"
    }

    if args.model in ["v5", "v8", "v10"]:
        model = YOLOModel(model_paths[args.model], conf_threshold=args.conf)
    elif args.model == "rcnn":
        model = FasterRCNNModel(model_paths["rcnn"], conf_threshold=args.conf)
    else:
        raise ValueError("Modelo não suportado")

    image_path = args.image if args.image else '../images/input_example.jpg'

    inference = Inference(model)
    inference.run(image_path, score_threshold=args.conf)


if __name__ == '__main__':
    main()