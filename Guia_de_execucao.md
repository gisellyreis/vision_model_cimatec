
````markdown
# Guia de Execução e Treinamento de Modelos YOLOv8 no HPC

---

## 1. Preparar o Ambiente Local

Acesse o diretório compartilhado no HPC e ative o ambiente virtual.

```bash
# Acessar a pasta compartilhada
cd /scratch/academico-cimatec/ccad

# Executar script bash para ativar o ambiente
source setup_env.sh
````

O arquivo **`setup_env.sh`** deve conter:

```bash
# Carregar módulo do Anaconda
module load anaconda3/2023.07

# Ativar ambiente virtual Conda
source activate ccad
```

---

## 2. Configuração do Dataset e Modelo

### 2.1 Arquivo YAML do Dataset

Edite o arquivo `.yaml` para definir corretamente os caminhos e classes:

```yaml
path: /scratch/academico-cimatec/ccad/vision_model_cimatec/datasets/mobdrone
train: images/train
val: images/val
test: images/test
names:
  0: swimmer
  1: boat
  2: surfboard
```

---

### 2.2 Download do Modelo YOLOv8

Se ocorrer erro como:

```
ERROR Error writing to /tmp/Ultralytics/settings.json: [Errno 13] Permission denied
Erro na validação: Download failure for https://ultralytics.com/assets/Arial.ttf
```

Isso significa que a biblioteca **Ultralytics** não pôde criar arquivos no `/tmp`.
Nesse caso, remova ou mova qualquer modelo existente e baixe novamente:

```bash
rm yolov8n.pt || mv yolov8n.pt /seu_diretorio/
```

Crie o arquivo `modelo.py`:

```python
from ultralytics import YOLO
YOLO('yolov8n.pt')
```

Execute para baixar o modelo:

```bash
python modelo.py
```

---

## 3. Rodar Inferência Local

Execute o script de inferência:

```bash
python inferenciaYolov.py
```

O script irá:

1. Carregar o modelo `yolov8n.pt`.
2. Validar com o dataset definido no `.yaml`.
3. Executar a inferência em todas as imagens.
4. Salvar métricas em `resultados_modelos.csv`.

---

## 4. Verificar Resultados

```bash
ls
nano resultados_modelos.csv
```

Os resultados incluem:

* Detecções por imagem
* Estatísticas de confiança (mínimo, máximo, média, desvio padrão)
* Tempo médio de processamento por imagem

---

## 5. Execução de Inferência com SLURM (GPU)

Crie ou edite o arquivo `inferencia.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=inferencia_yolo
#SBATCH --output=saida_inferencia_%j.txt
#SBATCH --error=erro_inferencia_%j.txt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

cd ../preprocessing/inference
python inferenciaYolo.py
```

### Submeter o job:

```bash
sbatch inferencia.slurm
```

### Acompanhar o job:

```bash
squeue -u $USER
```

### Verificar saídas e erros:

```bash
cat saida_inferencia_<JOBID>.txt
cat erro_inferencia_<JOBID>.txt
```

---

## 6. Treinamento YOLOv8 com SLURM (GPU)

Crie o script `treinamento.slurm` para treinar o modelo:

```bash
#!/bin/bash
#SBATCH --account=
#SBATCH --job-name=treino_yolo
#SBATCH --output=saida_treino_%j.txt
#SBATCH --error=erro_treino_%j.txt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

cd ../models/yolov8
python train.py
```

### Submeter para execução:

```bash
sbatch treinamento.slurm
```

### Acompanhar execução:

```bash
squeue -u $USER
```

---

## 7. Saídas e Resultados

1. **CSV de métricas** (`resultados_modelos.csv` ou `resultados_cnn.csv`)
2. **Logs do SLURM** (`saida_inferencia_<JOBID>.txt`, `erro_inferencia_<JOBID>.txt`)
3. **Resumo de métricas no terminal**
4. **Distribuição de classes detectadas**

---


