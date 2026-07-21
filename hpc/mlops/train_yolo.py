"""
Uso:
    python train_yolo.py --model yolov10.pt
    python train_yolo.py --model yolov8.pt --epochs 50 --lr 0.001 --batch 128
"""

import argparse

import mlflow
import pandas as pd
import torch
from ultralytics import YOLO, settings

from mlflow_utils import (
    BASE_DIR,
    MODELS_DIR,
    RUNS_DIR,
    log_common_params,
    sanitize_metric_name,
    setup_mlflow,
)

DATA_YAML = BASE_DIR / "datasets" / "data_no_other.yaml"
YOLO_RUNS_DIR = RUNS_DIR / "train_combined"

settings.update({"mlflow": False})


def parse_args():
    parser = argparse.ArgumentParser(description="Treino YOLO com MLflow")
    parser.add_argument("--model", default="yolov10.pt", help="Arquivo .pt em models/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="0,1,2,3")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--run-suffix", default="combined", help="Sufixo do nome do run")
    return parser.parse_args()


def log_metrics_from_results_csv(csv_path):
    if not csv_path.exists():
        print(f"AVISO: results.csv não encontrado em {csv_path}; métricas não logadas.")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    best_map = None
    for i, row in df.iterrows():
        step = int(row["epoch"]) if "epoch" in df.columns else i
        epoch_metrics = {}
        precision = recall = None

        for col in df.columns:
            val = row[col]
            if not isinstance(val, (int, float)):
                continue
            if "mAP50-95" in col:
                epoch_metrics["val_mAP50_95"] = float(val)
                best_map = float(val) if best_map is None else max(best_map, float(val))
            elif "mAP50" in col:
                epoch_metrics["val_mAP50"] = float(val)
            elif "precision" in col:
                precision = float(val)
                epoch_metrics["val_precision"] = precision
            elif "recall" in col:
                recall = float(val)
                epoch_metrics["val_recall"] = recall
            elif "val/box_loss" in col:
                epoch_metrics["val_box_loss"] = float(val)
            elif "val/cls_loss" in col:
                epoch_metrics["val_cls_loss"] = float(val)
            elif "train/box_loss" in col:
                epoch_metrics["train_box_loss"] = float(val)

        if precision is not None and recall is not None and (precision + recall) > 0:
            epoch_metrics["val_f1"] = 2 * precision * recall / (precision + recall)

        if epoch_metrics:
            mlflow.log_metrics(epoch_metrics, step=step)

    if best_map is not None:
        mlflow.log_metric("best_val_mAP50_95", best_map)


def train_model(args):
    model_path = MODELS_DIR / args.model

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Arquivo data.yaml não encontrado: {DATA_YAML}")

    print("===== AMBIENTE =====")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA do PyTorch: {torch.version.cuda}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA não está disponível. O treino foi interrompido para evitar rodar em CPU."
        )

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    run_name = f"{model_path.stem}_{args.run_suffix}_{args.epochs}ep"

    tracking_uri = setup_mlflow()

    print("===== TREINO =====")
    print(f"Modelo: {model_path}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Run name: {run_name}")
    print(f"MLflow tracking: {tracking_uri}")

    model = YOLO(str(model_path))

    with mlflow.start_run(run_name=run_name):
        log_common_params(
            framework="yolo",
            model_file=args.model,
            dataset=str(DATA_YAML),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            lr=args.lr,
            patience=args.patience,
        )
        mlflow.log_params(
            {
                "device": args.device,
                "workers": args.workers,
                "mosaic": args.mosaic,
                "mixup": args.mixup,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "translate": 0.1,
                "scale": 0.5,
                "fliplr": 0.5,
            }
        )

        results = model.train(
            data=str(DATA_YAML),
            project=str(YOLO_RUNS_DIR),
            name=run_name,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            lr0=args.lr,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            mosaic=args.mosaic,
            mixup=args.mixup,
            plots=True,
            save=True,
            val=True,
        )

        save_dir = YOLO_RUNS_DIR / run_name

        log_metrics_from_results_csv(save_dir / "results.csv")

        # Metricas finais agregadas do run
        if results is not None and hasattr(results, "results_dict"):
            final_metrics = {
                sanitize_metric_name(f"final_{k}"): float(v)
                for k, v in results.results_dict.items()
                if isinstance(v, (int, float))
            }
            if final_metrics:
                mlflow.log_metrics(final_metrics)

        best_weights = save_dir / "weights" / "best.pt"
        if best_weights.exists():
            mlflow.log_artifact(str(best_weights), artifact_path="weights")
        results_csv = save_dir / "results.csv"
        if results_csv.exists():
            mlflow.log_artifact(str(results_csv))


if __name__ == "__main__":
    train_model(parse_args())
