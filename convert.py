import os
import xml.etree.ElementTree as ET

xml_train_path = 'datasets/sea_voc/train/annotations' 
xml_val_path = 'datasets/sea_voc/val/annotations'
labels_base_dir = 'datasets/sea_voc/train/labels'

classes = ["swimmer", "boat", "other"]

os.makedirs(os.path.join(labels_base_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(labels_base_dir, 'val'), exist_ok=True)

def convert_bbox_to_yolo(size, box):
    """Converte coordenadas absolutas para x_center, y_center, width, height normalizados."""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def convert_xml_to_txt(xml_folder, split):
    files = [f for f in os.listdir(xml_folder) if f.endswith('.xml')]
    
    for filename in files:
        tree = ET.parse(os.path.join(xml_folder, filename))
        root = tree.getroot()
        
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        file_stem = os.path.splitext(filename)[0]
        label_path = os.path.join(labels_base_dir, split, file_stem + '.txt')
        
        with open(label_path, 'w') as f:
            for obj in root.iter('object'):
                cls_name = obj.find('name').text
                if cls_name not in classes:
                    continue
                
                cls_id = classes.index(cls_name)
                xmlbox = obj.find('bndbox')
                
                b = (float(xmlbox.find('xmin').text), 
                     float(xmlbox.find('xmax').text), 
                     float(xmlbox.find('ymin').text), 
                     float(xmlbox.find('ymax').text))
                
                bb = convert_bbox_to_yolo((w, h), b)
                f.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}\n")

convert_xml_to_txt(xml_train_path, 'train')
convert_xml_to_txt(xml_val_path, 'val')

print("Conversão de XML para TXT finalizada!")