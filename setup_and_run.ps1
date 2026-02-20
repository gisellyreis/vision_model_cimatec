# 1. Criar e Ativar Ambiente Virtual
if (!(Test-Path -Path "venv")) {
    Write-Host "--- Criando ambiente virtual ---" -ForegroundColor Cyan
    python -m venv venv
}
Write-Host "--- Ativando ambiente virtual ---" -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1

# 2. Limpeza e Instalação de Dependências
Write-Host "--- Limpando versões antigas do Torch ---" -ForegroundColor Yellow
pip uninstall torch torchvision -y

Write-Host "--- Instalando dependências do requirements.txt ---" -ForegroundColor Cyan
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "--- Validando CUDA ---" -ForegroundColor Magenta
$cudaAvailable = python -c "import torch; print(torch.cuda.is_available())"
if ($cudaAvailable -like "*False*") {
    Write-Host "❌ ERRO: CUDA não detectado após instalação. Verifique seus drivers NVIDIA." -ForegroundColor Red
    exit
}

# 3. Rodar o MLflow UI em segundo plano
Write-Host "--- Iniciando MLflow UI (http://127.0.0.1:5000) ---" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "mlflow ui" -WindowStyle Minimized

# 4. Rodar o treinamento
Write-Host "--- Iniciando treinamento YOLO ---" -ForegroundColor Cyan
python train/yolo.py v8