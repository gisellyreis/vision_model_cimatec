import torch
import torchvision
 
def baixar_pesos():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
    caminho = "fasterrcnn_resnet50_fpn_coco.pth"
    torch.save(model.state_dict(), caminho)
    print(f"Pesos salvos em: {caminho}")
 
if __name__ == "__main__":
    baixar_pesos()
 