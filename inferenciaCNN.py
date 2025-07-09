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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

base_dir = os.path.dirname(os.path.abspath(__file__))
caminho_imagens = [
    os.path.join(base_dir, 'AFO', 'AFO', 'PART_1', 'PART_1', 'images', '*.jpg'),
    os.path.join(base_dir, 'AFO', 'AFO', 'PART_2', 'PART_2', 'images', '*.jpg'),
    os.path.join(base_dir, 'AFO', 'AFO', 'PART_3', 'PART_3', 'images', '*.jpg')
]

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

conf_threshold = 0.25

lista_imagens = []
for caminho in caminho_imagens:
    lista_imagens.extend(glob(caminho))

total_deteccoes = 0
confidencias = []
classes_detectadas = defaultdict(int)
tempos = []

for img_path in lista_imagens:
    image = Image.open(img_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        inicio = time.time()
        outputs = model([image_tensor])
        fim = time.time()
        tempos.append(fim - inicio)

    pred = outputs[0]
    for score, label in zip(pred['scores'], pred['labels']):
        if score.item() >= conf_threshold:
            total_deteccoes += 1
            confidencias.append(score.item())
            classes_detectadas[int(label.item())] += 1

media_deteccoes = total_deteccoes / len(lista_imagens)
media_confianca = np.mean(confidencias) if confidencias else 0
conf_min = np.min(confidencias) if confidencias else 0
conf_max = np.max(confidencias) if confidencias else 0
conf_std = np.std(confidencias) if confidencias else 0
tempo_medio = np.mean(tempos)

tabela = PrettyTable()
tabela.title = "Resumo CNN Faster R-CNN"
tabela.field_names = ["Métrica", "Valor"]

linhas = [
    ("Modelo", "fasterrcnn_resnet50_fpn"),
    ("Detecções por imagem", f"{media_deteccoes:.2f}"),
    ("Confiança média", f"{media_confianca:.4f}"),
    ("Confiança min", f"{conf_min:.4f}"),
    ("Confiança max", f"{conf_max:.4f}"),
    ("Desvio padrão conf.", f"{conf_std:.4f}"),
    ("Tempo médio por imagem (s)", f"{tempo_medio:.4f}")
]

for metrica, valor in linhas:
    tabela.add_row([metrica, valor])

print()
print(tabela)

print("\nDistribuição de classes detectadas:")
for cls_id, count in classes_detectadas.items():
    print(f"   - Classe {cls_id}: {count} detecções")

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

if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(cabecalho)
        writer.writerow(dados_csv)
else:
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(dados_csv)

print(f"\nResultados salvos em: {csv_path}")
