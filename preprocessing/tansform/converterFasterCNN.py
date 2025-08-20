import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from PIL import Image
import shutil

yolo_base_dir = "datasets/afo"  
splits = ["train", "val", "test"]

voc_base_dir = "datasets/afo_voc"
os.makedirs(voc_base_dir, exist_ok=True)

classes = [
    "swimmer", "boat", "other"  
]

def create_voc_xml(filename, width, height, objects):
    """Cria o arquivo de anotação no formato Pascal VOC."""
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = "images"

    fname = ET.SubElement(annotation, "filename")
    fname.text = filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    for obj in objects:
        obj_tag = ET.SubElement(annotation, "object")
        ET.SubElement(obj_tag, "name").text = obj["name"]
        ET.SubElement(obj_tag, "pose").text = "Unspecified"
        ET.SubElement(obj_tag, "truncated").text = "0"
        ET.SubElement(obj_tag, "difficult").text = "0"

        bbox = ET.SubElement(obj_tag, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(obj["xmin"])
        ET.SubElement(bbox, "ymin").text = str(obj["ymin"])
        ET.SubElement(bbox, "xmax").text = str(obj["xmax"])
        ET.SubElement(bbox, "ymax").text = str(obj["ymax"])

    xml_str = ET.tostring(annotation, encoding="utf-8")
    dom = parseString(xml_str)
    return dom.toprettyxml(indent="  ")

for split in splits:
    img_dir = os.path.join(yolo_base_dir, "images", split)
    lbl_dir = os.path.join(yolo_base_dir, "labels", split)

    voc_img_dir = os.path.join(voc_base_dir, split, "images")
    voc_annot_dir = os.path.join(voc_base_dir, split, "annotations")
    os.makedirs(voc_img_dir, exist_ok=True)
    os.makedirs(voc_annot_dir, exist_ok=True)

    if not os.path.exists(lbl_dir):
        continue

    label_files = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]
    if not label_files:
        print(f"[AVISO] Nenhuma anotação encontrada em {lbl_dir}. Pulando {split}.")
        continue

    for label_file in label_files:
        try:
            img_file = label_file.replace(".txt", ".jpg")
            img_path = os.path.join(img_dir, img_file)
            lbl_path = os.path.join(lbl_dir, label_file)

            if not os.path.exists(img_path):
                continue

            shutil.copy(img_path, os.path.join(voc_img_dir, img_file))
            img = Image.open(img_path)
            w, h = img.size

            objects = []
            with open(lbl_path, "r") as f:
                for line in f:
                    try:
                        cls_id, x_center, y_center, width, height = map(float, line.strip().split())
                        cls_name = classes[int(cls_id)]
                        xmin = int((x_center - width / 2) * w)
                        ymin = int((y_center - height / 2) * h)
                        xmax = int((x_center + width / 2) * w)
                        ymax = int((y_center + height / 2) * h)

                        objects.append({
                            "name": cls_name,
                            "xmin": max(0, xmin),
                            "ymin": max(0, ymin),
                            "xmax": min(w, xmax),
                            "ymax": min(h, ymax)
                        })
                    except Exception as e:
                        print(f"Linha inválida em {lbl_path}: {line.strip()} ({e})")

            xml_content = create_voc_xml(img_file, w, h, objects)
            xml_path = os.path.join(voc_annot_dir, label_file.replace(".txt", ".xml"))
            with open(xml_path, "w") as f:
                f.write(xml_content)

        except Exception as e:
            print(f"Falha ao processar {label_file}: {e}")
