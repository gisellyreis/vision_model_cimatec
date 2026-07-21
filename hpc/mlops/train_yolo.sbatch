#!/bin/bash
#SBATCH --job-name=train_yolo_mlflow
#SBATCH --account=ccad
#SBATCH --output=saida_train_yolo_%j.txt
#SBATCH --error=erro_train_yolo_%j.txt
#SBATCH --partition=gpulongc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=72:00:00

cd /scratch/projetos/ccad/ccad-ogun/vision_model_cimatec/hpc

PYTHON=/home/cimatec/laurinne.lira/.conda/envs/env01/bin/python

# Um único sbatch serve para qualquer modelo YOLOs:
#   sbatch train_yolo.sbatch yolov5.pt
#   sbatch train_yolo.sbatch yolov8.pt
#   sbatch train_yolo.sbatch yolov10.pt
MODEL=${1:-yolov8s.pt}

echo "===== AMBIENTE ====="
$PYTHON --version
$PYTHON -c "import sys; print('Executável:', sys.executable)"

echo -e "\n===== GPU ====="
nvidia-smi

echo -e "\n===== TESTE PYTORCH ====="
$PYTHON -c "import torch; print('Torch:', torch.__version__); print('CUDA PyTorch:', torch.version.cuda); print('CUDA disponível:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA não disponível no PyTorch'"

echo -e "\n===== TESTE ULTRALYTICS E MLFLOW ====="
$PYTHON -c "from ultralytics import YOLO; print('Ultralytics OK')"
$PYTHON -c "import mlflow; print('MLflow:', mlflow.__version__)"

export MLFLOW_ALLOW_FILE_STORE=true
export GIT_PYTHON_REFRESH=quiet

echo -e "\n===== INICIANDO TREINO YOLO: $MODEL ====="
#   $PYTHON -u mlops/train_yolo.py --model $MODEL --epochs 50 --lr 0.001
$PYTHON -u mlops/train_yolo.py --model $MODEL
