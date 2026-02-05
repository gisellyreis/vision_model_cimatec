import os
import cv2
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import logging
from datetime import datetime

class Config:
    DATASET_DIR = "datasets/sea_voc"
    CLASSES = ["__background__", "swimmer", "boat", "other"]
    NUM_CLASSES = len(CLASSES)
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.00001
    DEVICE = 0
    NUM_WORKERS = 2
    IMG_SIZE = 480
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def parse_voc_annotation(xml_file, class_to_idx):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes, labels = [], []
    for obj in root.findall("object"):
        label = obj.find("name").text
        if label not in class_to_idx:
            continue
        xml_box = obj.find("bndbox")
        bbox = [
            int(xml_box.find("xmin").text),
            int(xml_box.find("ymin").text),
            int(xml_box.find("xmax").text),
            int(xml_box.find("ymax").text),
        ]
        boxes.append(bbox)
        labels.append(class_to_idx[label])
    return boxes, labels

class VOCDataset(Dataset):
    EXTENSIONS = [".jpg", ".jpeg", ".png"]

    def __init__(self, root, transforms=None, class_map=None):
        self.root = root
        self.transforms = transforms
        self.img_dir = os.path.join(root, "images")
        self.ann_dir = os.path.join(root, "annotations")
        self.imgs = sorted([f for f in os.listdir(self.img_dir) if any(f.endswith(ext) for ext in self.EXTENSIONS)])
        self.class_map = class_map or {}

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        ann_path = os.path.join(self.ann_dir, os.path.splitext(self.imgs[idx])[0] + ".xml")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes, labels = parse_voc_annotation(ann_path, self.class_map)
        target = {"boxes": np.array(boxes, dtype=np.float32),
                  "labels": np.array(labels, dtype=np.int64)}

        if self.transforms:
            transformed = self.transforms(image=img, bboxes=target['boxes'], labels=target['labels'])
            img = transformed['image']
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            target['labels'] = torch.as_tensor(transformed['labels'], dtype=torch.int64)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transform(train=True, size=480):
    transforms = [A.Resize(size, size)]
    if train:
        transforms += [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ]
    transforms += [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0),
        ToTensorV2()
    ]
    return A.Compose(
        transforms, bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"])
    )

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

class Trainer:
    def __init__(self, model, optimizer, lr_scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.best_val_loss = float("inf")

    def train_one_epoch(self, data_loader):
        self.model.train()
        total_loss = 0
        loop = tqdm(data_loader, desc="Training", leave=False)
        for imgs, targets in loop:
            imgs = [img.to(self.device) for img in imgs]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            total_loss += losses.item()
            loop.set_postfix(loss=losses.item())
        return total_loss / len(data_loader)

    def validate(self, data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for imgs, targets in tqdm(data_loader, desc="Validation", leave=False):
                imgs = [img.to(self.device) for img in imgs]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                loss_dict = self.model(imgs, targets)
                total_loss += sum(loss for loss in loss_dict.values()).item()
        return total_loss / len(data_loader)

    def save_checkpoint(self, epoch, val_loss):
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        filename = f"epoch{epoch+1}_valloss{val_loss:.4f}.pth"
        path = os.path.join(Config.CHECKPOINT_DIR, filename)
        torch.save(self.model.state_dict(), path)
        if is_best:
            torch.save(self.model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"))
            logging.info(f"Novo melhor modelo salvo! {path}")

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    cfg = Config()
    logging.info(f"Rodando em dispositivo {cfg.DEVICE}")

    class_map = {cls: i for i, cls in enumerate(cfg.CLASSES)}

    dataset_train = VOCDataset(
        os.path.join(cfg.DATASET_DIR, "train"),
        transforms=get_transform(train=True, size=cfg.IMG_SIZE),
        class_map=class_map
    )
    dataset_valid = VOCDataset(
        os.path.join(cfg.DATASET_DIR, "val"),
        transforms=get_transform(train=False, size=cfg.IMG_SIZE),
        class_map=class_map
    )

    train_loader = DataLoader(dataset_train, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn)
    valid_loader = DataLoader(dataset_valid, batch_size=cfg.BATCH_SIZE, shuffle=False,
                              num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn)

    model = get_model(cfg.NUM_CLASSES).to(cfg.DEVICE)
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=cfg.LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    trainer = Trainer(model, optimizer, lr_scheduler, cfg.DEVICE)

    for epoch in range(cfg.NUM_EPOCHS):
        train_loss = trainer.train_one_epoch(train_loader)
        val_loss = trainer.validate(valid_loader)
        lr_scheduler.step()
        logging.info(f"[Epoch {epoch+1}/{cfg.NUM_EPOCHS}] Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        trainer.save_checkpoint(epoch, val_loss)

    logging.info("Treinamento concluído")

if __name__ == "__main__":
    main()
