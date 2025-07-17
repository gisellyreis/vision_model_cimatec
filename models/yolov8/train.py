import torch
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model_path='yolov8s.pt', data_path='../data.yaml'):
    model = YOLO(model_path)

    model.train(
        data=data_path,
        name='yolov8s_sds',
        epochs=10,
        imgsz=640,
        batch=16,
        device=0,
        lr0=0.00001,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2
    )


if __name__ == '__main__':
    train_model()
