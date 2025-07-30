import os
import time
import torch
import torchvision
import numpy as np
from glob import glob
from collections import defaultdict
from torchvision.transforms import functional as F
from PIL import Image
from prettytable import PrettyTable
from datetime import datetime
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Dataset e caminhos
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_base = "/scratch/academico-cimatec/ccad/vision_model_cimatec/datasets/mobdrone/images"
caminho_imagens = [
    os.path.join(dataset_base, subset, "*.png") for subset in ["train", "val", "test"]
]
lista_imagens = [img for path in caminho_imagens for img in glob(path)]

# Modelo Faster R-CNN
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device).eval()
conf_threshold = 0.25

total_deteccoes, confidencias, classes_detectadas, tempos = 0, [], defaultdict(int), []

# Inferência
for img_path in lista_imagens:
    image_tensor = F.to_tensor(Image.open(img_path).convert("RGB")).to(device)
    with torch.no_grad():
        inicio = time.time()
        outputs = model([image_tensor])[0]
        tempos.append(time.time() - inicio)

    mask = outputs["scores"] >= conf_threshold
    scores = outputs["scores"][mask].cpu().numpy()
    labels = outputs["labels"][mask].cpu().numpy()

    total_deteccoes += len(scores)
    confidencias.extend(scores)
    for lbl in labels:
        classes_detectadas[int(lbl)] += 1

# Métricas
media_deteccoes = total_deteccoes / len(lista_imagens)
media_confianca = np.mean(confidencias) if confidencias else 0
conf_min = np.min(confidencias) if confidencias else 0
conf_max = np.max(confidencias) if confidencias else 0
conf_std = np.std(confidencias) if confidencias else 0
tempo_medio = np.mean(tempos)

# Tabela de resultados
tabela = PrettyTable()
tabela.title = "Resumo CNN Faster R-CNN"
tabela.field_names = ["Métrica", "Valor"]
for metrica, valor in [
    ("Modelo", "fasterrcnn_resnet50_fpn"),
    ("Detecções por imagem", f"{media_deteccoes:.2f}"),
    ("Confiança média", f"{media_confianca:.4f}"),
    ("Confiança min", f"{conf_min:.4f}"),
    ("Confiança max", f"{conf_max:.4f}"),
    ("Desvio padrão conf.", f"{conf_std:.4f}"),
    ("Tempo médio por imagem (s)", f"{tempo_medio:.4f}")
]:
    tabela.add_row([metrica, valor])

print("\n" + str(tabela))

print("\nDistribuição de classes detectadas:")
for cls_id, count in classes_detectadas.items():
    print(f"   - Classe {cls_id}: {count} detecções")

# Salvar resultados em CSV
csv_path = os.path.join(base_dir, "resultados_cnn.csv")
cabecalho = [
    "Modelo", "Det/img", "Conf média", "Conf min", "Conf max",
    "Conf std", "Tempo médio (s)", "Data/Hora"
]
dados_csv = [
    "fasterrcnn_resnet50_fpn", f"{media_deteccoes:.2f}", f"{media_confianca:.4f}",
    f"{conf_min:.4f}", f"{conf_max:.4f}", f"{conf_std:.4f}", f"{tempo_medio:.4f}",
    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
]

modo = "a" if os.path.exists(csv_path) else "w"
with open(csv_path, mode=modo, newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if modo == "w":
        writer.writerow(cabecalho)
    writer.writerow(dados_csv)

print(f"\nResultados salvos em: {csv_path}")
