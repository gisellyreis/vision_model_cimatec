import os
import csv
import json

dataset_dir = "datasets/AFO"
output_dir = os.path.join(dataset_dir, "annotations_csv_json")
os.makedirs(output_dir, exist_ok=True)

splits = ["train", "test", "val"]

for split in splits:
    images_dir = os.path.join(dataset_dir, "images", split)
    labels_dir = os.path.join(dataset_dir, "labels", split)

    split_output_dir = os.path.join(output_dir, split)
    os.makedirs(split_output_dir, exist_ok=True)

    csv_path = os.path.join(split_output_dir, f"{split}_annotations.csv")
    json_path = os.path.join(split_output_dir, f"{split}_annotations.json")

    data_list = []

    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_path", "label_path", "class_id", "x_center", "y_center", "width", "height"])

        for label_file in os.listdir(labels_dir):
            if not label_file.endswith(".txt"):
                continue

            label_path = os.path.join(labels_dir, label_file)
            image_file = os.path.splitext(label_file)[0] + ".jpg"
            image_path = os.path.join(images_dir, image_file)

            with open(label_path, "r") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, x_center, y_center, width, height = parts
                    writer.writerow([image_path, label_path, class_id, x_center, y_center, width, height])

                    data_list.append({
                        "image_path": image_path,
                        "label_path": label_path,
                        "class_id": int(class_id),
                        "x_center": float(x_center),
                        "y_center": float(y_center),
                        "width": float(width),
                        "height": float(height)
                    })

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(data_list, json_file, indent=4, ensure_ascii=False)

