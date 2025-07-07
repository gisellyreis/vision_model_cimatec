import os
import shutil

base_dir = os.path.dirname(os.path.abspath(__file__))

labels_dir = os.path.join(base_dir, 'AFO', 'AFO', 'PART_1', 'PART_1', '6categories')
partes = ['PART_2', 'PART_3']

for parte in partes:
    images_dir = os.path.join(base_dir, 'AFO', 'AFO', parte, parte, 'images')
    labels_dest_dir = os.path.join(base_dir, 'AFO', 'AFO', parte, parte, 'labels')

    if not os.path.exists(images_dir):
        print(f"Diretório não encontrado: {images_dir}")
        continue

    os.makedirs(labels_dest_dir, exist_ok=True)
    copiados, movidos = 0, 0

    for arquivo in os.listdir(images_dir):
        if arquivo.endswith('.jpg'):
            nome_label = arquivo.replace('.jpg', '.txt')
            origem_label = os.path.join(labels_dir, nome_label)
            destino_temporario = os.path.join(images_dir, nome_label)
            destino_final = os.path.join(labels_dest_dir, nome_label)

            if os.path.exists(origem_label):
                shutil.copy(origem_label, destino_temporario)
                copiados += 1
            else:
                print(f"Anotação não encontrada: {nome_label}")
            
            if os.path.exists(destino_temporario):
                shutil.move(destino_temporario, destino_final)
                movidos += 1
