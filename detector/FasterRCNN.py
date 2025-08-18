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

class FasterRCNN(Model):
    def __init__(self, model_path):
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        self.model.load_state_dict(torch.load(model_path))