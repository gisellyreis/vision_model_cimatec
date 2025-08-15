import sys
from ultralytics import YOLO

class Model:
    def __init__(self):
        pass


class CNNModel(Model):
    def __init__(self, model_path):
        super().__init__()
        pass

class YOLOModel(Model):
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.5):
        super().__init__()
        self.model = YOLO(model_path)
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold

class Inference(Model):
    def __init__(self, model):
        self._model = model
    
    def run(self, image):
        results = self._model.model(image)
        annotated_image = results[0]
        annotated_image.show()
        return results



def main():
    image = 'images/input_example.jpg'

    yolo_model_path = '../models/best.pt'

    if len(sys.argv) > 1:
        image = sys.argv[1]

    yolo = YOLOModel(yolo_model_path)

    inference = Inference(yolo)
    inference.run(image)


if __name__ == '__main__':
    main()