#!/bin/bash
#SBATCH --job-name=train_cnn_mlflow
#SBATCH --account=ccad
#SBATCH --output=saida_train_cnn_%j.txt
#SBATCH --error=erro_train_cnn_%j.txt
#SBATCH --partition=gpulongc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=72:00:00
 
cd /scratch/projetos/ccad/ccad-ogun/vision_model_cimatec/hpc
 
PYTHON=/home/cimatec/laurinne.lira/.conda/envs/env01/bin/python
 
echo "===== AMBIENTE ====="
$PYTHON --version
$PYTHON -c "import sys; print('Executável:', sys.executable)"
 
echo -e "\n===== GPU ====="
nvidia-smi
 
echo -e "\n===== TESTE PYTORCH ====="
$PYTHON -c "import torch; print('Torch:', torch.__version__); print('CUDA PyTorch:', torch.version.cuda); print('CUDA disponível:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA não disponível no PyTorch'"
 
echo -e "\n===== TESTE MLFLOW E TORCHMETRICS ====="
$PYTHON -c "import mlflow; print('MLflow:', mlflow.__version__)"
$PYTHON -c "import torchmetrics; print('Torchmetrics:', torchmetrics.__version__)"
$PYTHON -c "from torchmetrics.detection.mean_ap import MeanAveragePrecision; MeanAveragePrecision(iou_type='bbox')" && echo "mAP backend OK"

export MLFLOW_ALLOW_FILE_STORE=true
export GIT_PYTHON_REFRESH=quiet

echo -e "\n===== INICIANDO TREINO CNN ====="
#   $PYTHON -u mlops/train_cnn.py --epochs 50 --lr 0.001 --batch 8
$PYTHON -u mlops/train_cnn.py --model fasterrcnn_resnet50_fpn_coco.pth --workers 4
