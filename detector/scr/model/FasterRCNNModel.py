import random
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import ImageFolder
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .model import Model

class FasterRCNNModel(Model):
    def __init__(self, model_path, conf_threshold=0.8):
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        self.name = "fasterRCNN"
        self.conf_threshold = conf_threshold
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)

        self.classes = ["__background__", "swimmer", "boat", "other"]
        self.colors = [[c/255 for c in [random.randint(0, 255) for _ in range(3)]] for _ in self.classes]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)

    def predict(self, image_path):
        from PIL import Image
        from torchvision.transforms import functional as F

        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(image_tensor)[0]

        boxes = preds["boxes"].cpu().numpy().tolist()
        scores = preds["scores"].cpu().numpy().tolist()
        labels = [self.classes[i] for i in preds["labels"].cpu().numpy().tolist()]

        h, w = image.size
        valid = [(b, s, l) for b, s, l in zip(boxes, scores, labels) if s >= self.conf_threshold]
        print(f"Input Path: {image_path}: | {w}x{h} | {len(valid)} objects")

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels
        }
