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
        self.model = torch.load(model_path)