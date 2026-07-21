"""
Uso:
    python train_cnn.py --model fasterrcnn_resnet50_fpn_coco.pth
    python train_cnn.py --epochs 50 --lr 0.001 --batch 8

Requer: pip install torchmetrics
"""

import argparse
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import albumentations as A
import cv2
import mlflow
import numpy as np
import torch
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from tqdm import tqdm

from mlflow_utils import (
    BASE_DIR,
    MODELS_DIR,
    RUNS_DIR,
    log_common_params,
    setup_mlflow,
)

DATASET_DIR = BASE_DIR / "datasets" / "dataset_all_voc"

CLASSES = ["__background__", "swimmer", "boat", "other"]
NUM_CLASSES = len(CLASSES)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Treino Faster R-CNN com MLflow")
    parser.add_argument("--model", default="fasterrcnn_resnet50_fpn_coco.pth",
                        help="Arquivo .pth em models/ (se ausente, usa pesos COCO)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--run-suffix", default="cnn", help="Sufixo do nome do run")
    return parser.parse_args()


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_voc_annotation(xml_file, class_to_idx):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes, labels = [], []
    for obj in root.findall("object"):
        label = obj.find("name").text
        if label not in class_to_idx:
            continue

        xml_box = obj.find("bndbox")
        xmin = int(float(xml_box.find("xmin").text))
        ymin = int(float(xml_box.find("ymin").text))
        xmax = int(float(xml_box.find("xmax").text))
        ymax = int(float(xml_box.find("ymax").text))

        # Filtra bounding boxes corrompidas
        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_to_idx[label])
    return boxes, labels


class VOCDataset(Dataset):
    EXTENSIONS = [".jpg", ".jpeg", ".png"]

    def __init__(self, root, transforms=None, class_map=None):
        self.root = Path(root)
        self.transforms = transforms

        self.img_dir = self.root / "images"
        self.ann_dir = self.root / "annotations"

        self.imgs = sorted(
            [f for f in os.listdir(self.img_dir)
             if any(f.endswith(ext) for ext in self.EXTENSIONS)]
        )
        self.class_map = class_map or {}

    def __getitem__(self, idx):
        img_path = self.img_dir / self.imgs[idx]
        ann_path = self.ann_dir / (img_path.stem + ".xml")

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes, labels = parse_voc_annotation(str(ann_path), self.class_map)

        if len(boxes) == 0:
            target = {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64),
            }
        else:
            target = {
                "boxes": np.array(boxes, dtype=np.float32),
                "labels": np.array(labels, dtype=np.int64),
            }

        if self.transforms:
            if len(boxes) > 0:
                transformed = self.transforms(
                    image=img, bboxes=target["boxes"], labels=target["labels"]
                )
                target["boxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
                target["labels"] = torch.as_tensor(transformed["labels"], dtype=torch.int64)
            else:
                transformed = self.transforms(image=img)

            img = transformed["image"]

        # Garante tensores mesmo quando nao ha transforms com bboxes
        if not torch.is_tensor(target["boxes"]):
            target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
            target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)

        return img, target

    def __len__(self):
        return len(self.imgs)


class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc="Train", leave=False)
        for images, targets in progress_bar:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            self.optimizer.step()

            total_loss += losses.item()
            progress_bar.set_postfix({"loss": f"{losses.item():.4f}"})

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        # O Faster R-CNN so retorna o dicionario de losses em modo train,
        # por isso a val_loss e calculada assim (sem gradiente).
        self.model.train()
        total_loss = 0

        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Valid", leave=False)
            for images, targets in progress_bar:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                total_loss += losses.item()
                progress_bar.set_postfix({"val_loss": f"{losses.item():.4f}"})

        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate_map(self, dataloader, score_thr=0.5, iou_thr=0.5):
        self.model.eval()
        metric = MeanAveragePrecision(iou_type="bbox", backend="faster_coco_eval")

        tp = fp = fn = 0

        progress_bar = tqdm(dataloader, desc="mAP", leave=False)
        for images, targets in progress_bar:
            images = list(image.to(self.device) for image in images)
            outputs = self.model(images)

            preds = [{k: v.cpu() for k, v in out.items()} for out in outputs]
            gts = [
                {"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()}
                for t in targets
            ]
            metric.update(preds, gts)

            # Contagem de TP/FP/FN por imagem no ponto de operacao
            for pred, gt in zip(preds, gts):
                keep = pred["scores"] >= score_thr
                pred_boxes = pred["boxes"][keep]
                pred_labels = pred["labels"][keep]
                gt_boxes = gt["boxes"]
                gt_labels = gt["labels"]

                if len(gt_boxes) == 0:
                    fp += len(pred_boxes)
                    continue
                if len(pred_boxes) == 0:
                    fn += len(gt_boxes)
                    continue

                ious = box_iou(pred_boxes, gt_boxes)
                matched_gt = set()
                # Ordena predicoes por score decrescente para o matching greedy
                order = torch.argsort(pred["scores"][keep], descending=True)
                for pi in order.tolist():
                    best_iou, best_gt = 0.0, -1
                    for gi in range(len(gt_boxes)):
                        if gi in matched_gt:
                            continue
                        if pred_labels[pi] != gt_labels[gi]:
                            continue
                        iou = ious[pi, gi].item()
                        if iou > best_iou:
                            best_iou, best_gt = iou, gi
                    if best_iou >= iou_thr and best_gt >= 0:
                        tp += 1
                        matched_gt.add(best_gt)
                    else:
                        fp += 1
                fn += len(gt_boxes) - len(matched_gt)

        results = metric.compute()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        return {
            "val_mAP50": float(results["map_50"]),
            "val_mAP50_95": float(results["map"]),
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
        }


def get_model(num_classes, model_path=None):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    if model_path and model_path.exists():
        logging.info(f"Carregando pesos customizados de: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        logging.warning(f"Arquivo não encontrado em {model_path}")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train_model(args):
    model_path = MODELS_DIR / args.model

    print("===== AMBIENTE =====")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA do PyTorch: {torch.version.cuda}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível. Treino interrompido.")

    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Pastas train/val não encontradas em: {DATASET_DIR}")

    run_name = f"{Path(args.model).stem}_{args.run_suffix}_{args.epochs}ep"
    run_dir = RUNS_DIR / "train_cnn" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    tracking_uri = setup_mlflow()

    print("\n===== TREINO =====")
    print(f"Pesos iniciais: {model_path if model_path.exists() else 'COCO Pre-trained'}")
    print(f"Dataset VOC: {DATASET_DIR}")
    print(f"Run name e Saída: {run_dir}")
    print(f"MLflow tracking: {tracking_uri}")
    print(f"Tamanho Batch: {args.batch} | Épocas: {args.epochs} | Classes: {NUM_CLASSES}\n")

    class_map = {name: idx for idx, name in enumerate(CLASSES)}

    train_transform = A.Compose([
        A.Resize(args.imgsz, args.imgsz),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.015, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

    valid_transform = A.Compose([
        A.Resize(args.imgsz, args.imgsz),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

    train_dataset = VOCDataset(train_dir, transforms=train_transform, class_map=class_map)
    valid_dataset = VOCDataset(val_dir, transforms=valid_transform, class_map=class_map)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, collate_fn=collate_fn)

    model = get_model(NUM_CLASSES, model_path if model_path.exists() else None).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model, optimizer, device)

    with mlflow.start_run(run_name=run_name):
        log_common_params(
            framework="fasterrcnn",
            model_file=args.model,
            dataset=str(DATASET_DIR),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            lr=args.lr,
            patience=args.patience,
        )
        mlflow.log_params(
            {
                "workers": args.workers,
                "num_classes": NUM_CLASSES,
                "classes": ",".join(CLASSES),
                "optimizer": "Adam",
                "pretrained_weights": str(model_path) if model_path.exists() else "COCO",
            }
        )

        best_map = -1.0
        epochs_without_improvement = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = trainer.train_one_epoch(train_loader)
            val_loss = trainer.validate(valid_loader)
            val_metrics = trainer.evaluate_map(valid_loader)
            val_map50_95 = val_metrics["val_mAP50_95"]

            logging.info(
                f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - mAP50: {val_metrics['val_mAP50']:.4f} - "
                f"mAP50-95: {val_map50_95:.4f} - P: {val_metrics['val_precision']:.4f} - "
                f"R: {val_metrics['val_recall']:.4f} - F1: {val_metrics['val_f1']:.4f}"
            )
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **val_metrics,
                },
                step=epoch,
            )

            torch.save(model.state_dict(), weights_dir / "last.pt")

            if val_map50_95 > best_map:
                best_map = val_map50_95
                epochs_without_improvement = 0
                torch.save(model.state_dict(), weights_dir / "best.pt")
                logging.info(f"   -> Novo melhor modelo salvo em {weights_dir / 'best.pt'}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    logging.warning(
                        f"Early stopping at epoch {epoch}. Sem melhoras há {args.patience} épocas."
                    )
                    mlflow.log_param("stopped_early_at_epoch", epoch)
                    break

        mlflow.log_metric("best_val_mAP50_95", best_map)

        best_weights = weights_dir / "best.pt"
        if best_weights.exists():
            mlflow.log_artifact(str(best_weights), artifact_path="weights")


if __name__ == "__main__":
    train_model(parse_args())
