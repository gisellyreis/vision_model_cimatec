from ultralytics import YOLO
from model import Model


class YOLOModel(Model):
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.5):
        super().__init__()
        self.model = YOLO(model_path)
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold