import torchvision
import torch

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
