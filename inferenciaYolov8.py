from ultralytics import YOLO
from glob import glob
import os
import torch
from collections import defaultdict
import numpy as np
from prettytable import PrettyTable
from datetime import datetime
import csv
import time
import matplotlib.pyplot as plt

print("...")

base_dir = os.path.dirname(os.path.abspath(__file__))
modelo_path = os.path.join(base_dir, 'yolov8n.pt')
dataset_yaml = os.path.join(base_dir, 'AFO', 'AFO', 'AFO.yaml')
caminho_imagens = [
    os.path.join(base_dir, 'AFO', 'AFO', 'PART_1', 'PART_1', 'images', '*.jpg'),
    os.path.join(base_dir, 'AFO', 'AFO', 'PART_2', 'PART_2', 'images', '*.jpg'),
    os.path.join(base_dir, 'AFO', 'AFO', 'PART_3', 'PART_3', 'images', '*.jpg')
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo selecionado: {device}")
model = YOLO(modelo_path)

metrics = model.val(
    data=dataset_yaml,
    imgsz=640,
    conf=0.25,
    iou=0.5,
    device=device,
    cache=False
)

lista_imagens = []
for caminho in caminho_imagens:
    lista_imagens.extend(glob(caminho))

total_deteccoes = 0
confidencias = []
classes_detectadas = defaultdict(int)
tempos = []

for img_path in lista_imagens:
    inicio = time.time()
    resultados = model.predict(source=img_path, save=False, verbose=False, device=device)
    fim = time.time()
    tempos.append(fim - inicio)

    for r in resultados:
        total_deteccoes += len(r.boxes)
        confidencias.extend(r.boxes.conf.cpu().numpy())
        for cls in r.boxes.cls.cpu().numpy():
            classes_detectadas[int(cls)] += 1

media_deteccoes = total_deteccoes / len(lista_imagens)
media_confianca = np.mean(confidencias) if confidencias else 0
conf_min = np.min(confidencias) if confidencias else 0
conf_max = np.max(confidencias) if confidencias else 0
conf_std = np.std(confidencias) if confidencias else 0
tempo_medio = np.mean(tempos)

tabela_metricas = PrettyTable()
tabela_metricas.title = "Resumo das Métricas do Modelo"
tabela_metricas.field_names = ["Métrica", "Valor"]
tabela_metricas.align = "l"

linhas_metricas = [
    ("Modelo", os.path.basename(modelo_path)),
    ("mAP50 (val)", f"{metrics.box.map50:.4f}"),
    ("mAP50-95 (val)", f"{metrics.box.map:.4f}"),
    ("Precisão (val)", f"{metrics.box.mp:.4f}"),
    ("Revocação (val)", f"{metrics.box.mr:.4f}"),
    ("Detecções por imagem (predict)", f"{media_deteccoes:.2f}"),
    ("Confiança média (predict)", f"{media_confianca:.4f}"),
    ("Confiança min", f"{conf_min:.4f}"),
    ("Confiança max", f"{conf_max:.4f}"),
    ("Desvio padrão conf.", f"{conf_std:.4f}"),
    ("Tempo médio por imagem (s)", f"{tempo_medio:.4f}")
]

for metrica, valor in linhas_metricas:
    tabela_metricas.add_row([metrica, valor])

print()
print(tabela_metricas)

print("\nDistribuição de classes detectadas:")
for cls_id, count in classes_detectadas.items():
    nome_classe = model.names[cls_id]
    print(f"   - {nome_classe} ({cls_id}): {count} detecções")


csv_path = os.path.join(base_dir, "resultados_modelos.csv")
cabecalho = [
    "Modelo", "mAP50", "mAP50-95", "Precision", "Recall", "Det/img", "Conf média", 
    "Conf min", "Conf max", "Conf std", "Tempo médio (s)", "Data/Hora"
]
dados_csv = [
    os.path.basename(modelo_path), f"{metrics.box.map50:.4f}", f"{metrics.box.map:.4f}",
    f"{metrics.box.mp:.4f}", f"{metrics.box.mr:.4f}", f"{media_deteccoes:.2f}", f"{media_confianca:.4f}",
    f"{conf_min:.4f}", f"{conf_max:.4f}", f"{conf_std:.4f}", f"{tempo_medio:.4f}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
