import os
import sys
import torch
import time
import numpy as np
import csv
from pathlib import Path
from glob import glob
from collections import defaultdict
from datetime import datetime
from prettytable import PrettyTable

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / 'yolov5'
sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

device = select_device('')
base_dir = os.path.dirname(os.path.abspath(__file__))

modelo_path = os.path.join(base_dir, 'yolov5s.pt')  
caminhos = [
    os.path.join(base_dir, 'AFO', 'AFO', 'PART_1', 'PART_1', 'images', '*.jpg'),
    os.path.join(base_dir, 'AFO', 'AFO', 'PART_2', 'PART_2', 'images', '*.jpg'),
    os.path.join(base_dir, 'AFO', 'AFO', 'PART_3', 'PART_3', 'images', '*.jpg')
]

imgsz = 640
conf_thres = 0.25
iou_thres = 0.45

model = DetectMultiBackend(modelo_path, device=device)
model.eval()

lista_imagens = []
for caminho in caminhos:
    lista_imagens.extend(glob(caminho))

total_deteccoes = 0
confidencias = []
classes_detectadas = defaultdict(int)
tempos = []

for path_img in lista_imagens:
    dataset = LoadImages(path_img, img_size=imgsz, stride=model.stride, auto=True)
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time.time()
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        t2 = time.time()

        tempos.append(t2 - t1)

        for det in pred:
            if len(det):
                total_deteccoes += len(det)
                for *xyxy, conf, cls in det:
                    confidencias.append(conf.item())
                    classes_detectadas[int(cls.item())] += 1

media_deteccoes = total_deteccoes / len(lista_imagens)
media_confianca = np.mean(confidencias) if confidencias else 0
conf_min = np.min(confidencias) if confidencias else 0
conf_max = np.max(confidencias) if confidencias else 0
conf_std = np.std(confidencias) if confidencias else 0
tempo_medio = np.mean(tempos)
tabela = PrettyTable()
tabela.title = "Resumo YOLOv5"
tabela.field_names = ["Métrica", "Valor"]

linhas = [
    ("Modelo", os.path.basename(modelo_path)),
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

csv_path = os.path.join(base_dir, "resultados_yolov5.csv")
cabecalho = [
    "Modelo", "Det/img", "Conf média", "Conf min", "Conf max",
    "Conf std", "Tempo médio (s)", "Data/Hora"
]
dados_csv = [
    os.path.basename(modelo_path), f"{media_deteccoes:.2f}", f"{media_confianca:.4f}",
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
