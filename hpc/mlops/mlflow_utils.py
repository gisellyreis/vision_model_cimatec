import os
import re
from pathlib import Path
 
import mlflow
 
BASE_DIR = Path("/scratch/projetos/ccad/ccad-ogun/vision_model_cimatec")
MODELS_DIR = BASE_DIR / "models"
RUNS_DIR = BASE_DIR / "runs"
MLRUNS_DIR = Path("/scratch/projetos/ccad/ccad-ogun/mlops_maritime/mlruns")

EXPERIMENT_NAME = "vision_model_cimatec"
DEFAULT_TRACKING_URI = f"file://{MLRUNS_DIR}"
 
 
def setup_mlflow(experiment_name: str = EXPERIMENT_NAME) -> str:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri
 
 
def sanitize_metric_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_\-./ ]", "", name)
 
 
def log_common_params(
    framework: str,
    model_file: str,
    dataset: str,
    epochs: int,
    batch: int,
    imgsz: int,
    lr: float,
    patience: int,
) -> None:
    mlflow.log_params(
        {
            "framework": framework,
            "model": model_file,
            "dataset": dataset,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "lr": lr,
            "patience": patience,
        }
    )
