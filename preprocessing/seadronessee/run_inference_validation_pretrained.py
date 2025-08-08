from ultralytics import YOLO

model_name = "yolov5n.pt"
model = YOLO(model_name) 

# Run evaluation on the validation set
metrics = model.val(data='data.yaml', name='yolov5_mob', split='val', verbose=True)

# Print mAP, precision, recall, etc.
print("=== Evaluation Metrics ===")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"Precision: {metrics.box.p:}")
print(f"Recall: {metrics.box.r}")
