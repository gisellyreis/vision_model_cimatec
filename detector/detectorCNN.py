import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os, random

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

class FasterRCNNInference:
    def __init__(self, model_path, classes):
        self.device = "cpu"
        self.classes = classes
        self.colors = [[c/255 for c in [random.randint(0, 255) for _ in range(3)]] for _ in classes]
        self.model = get_model(len(classes))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()

    def run(self, image_path, output_path="output.png", score_threshold=0.6):
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(image_tensor)[0]

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

if __name__ == "__main__":
    inference = FasterRCNNInference( model_path="../models/checkpoint_epoch2.pth", classes=["__background__", "swimmer", "boat", "other"])
    inference.run(image_path="../models/a_101.jpg", output_path="detected_output.png", score_threshold=0.6)
