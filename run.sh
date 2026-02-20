#!/bin/bash

# 1. Criar e Ativar Ambiente Virtual
if [ ! -d "venv" ]; then
    echo -e "\e[36m--- Criando ambiente virtual ---\e[0m"
    python3 -m venv venv
fi

echo -e "\e[36m--- Ativando ambiente virtual ---\e[0m"
source venv/bin/activate

# 2. Limpeza e Instalação
echo -e "\e[33m--- Limpando versões antigas do Torch ---\e[0m"
pip uninstall torch torchvision -y

echo -e "\e[36m--- Instalando dependências do requirements.txt ---\e[0m"
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# Validação CUDA
echo -e "\e[35m--- Validando CUDA ---\e[0m"
cuda_available=$(python3 -c "import torch; print(torch.cuda.is_available())")

if [ "$cuda_available" == "False" ]; then
    echo -e "\e[31m❌ ERRO: CUDA não detectado. Verifique os drivers NVIDIA (nvidia-smi).\e[0m"
    exit 1
fi

# 3. Rodar o MLflow UI em segundo plano
echo -e "\e[36m--- Iniciando MLflow UI (http://127.0.0.1:5000) ---\e[0m"
mlflow ui --port 5000 > mlflow.log 2>&1 & 
disown

# 4. Rodar o treinamento
echo -e "\e[36m--- Iniciando treinamento YOLO ---\e[0m"
python3 train/yolo.py v8