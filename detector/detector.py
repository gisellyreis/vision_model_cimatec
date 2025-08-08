import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO

class Detector:
  def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.5):
    self.model = YOLO(model_path)
    self.model.conf = conf_threshold
    self.model.iou = iou_threshold 
   
  def preprocess(self, image):
    image = cv2.resize(image, (640, 640))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).float()
    return image
  
  def inference(self, image):
    results = self.model(image)
    results[0].show()
    return


if __name__ == "__main__":
  model_path = '../models/best.pt'
  image_path = '../datasets/seadronessee/images/test/8823.jpg'

  if len(sys.argv) > 1:
    image_path = sys.argv[1]

  detector = Detector(model_path)
  image = cv2.imread(image_path)
  detector.inference(image)