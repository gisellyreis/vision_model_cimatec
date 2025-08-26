import random
from ultralytics import YOLO
from .model import Model
        
        

class YOLOModel(Model):
    def __init__(self, model_path, conf_threshold=0.3, iou_threshold=0.5):
        super().__init__()
        self.model = YOLO(model_path)
        self.name = model_path.split('/')[-1].split('\\')[-1].split('_')[-1].split('.')[0]
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        
        self.classes = ["swimmer", "boat", "other"]
        self.colors = [[c/255 for c in [random.randint(0, 255) for _ in range(3)]] for _ in self.classes]

    def predict(self, image_path):
        results = self.model.predict(image_path, conf=self.conf_threshold, iou=self.iou_threshold)

        detections = results[0]
        boxes = detections.boxes.xyxy.cpu().numpy().tolist()
        scores = detections.boxes.conf.cpu().numpy().tolist()
        
        class_indices = detections.boxes.cls.cpu().numpy().tolist()
        labels = []
        for cls_idx in class_indices:
            cls_idx = int(cls_idx)
            if cls_idx < len(self.classes):
                labels.append(self.classes[cls_idx])
            else:
                labels.append(f"class_{cls_idx}")

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels
        }