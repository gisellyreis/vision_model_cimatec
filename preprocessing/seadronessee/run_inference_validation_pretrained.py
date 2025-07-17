from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

# Run evaluation on the validation set
metrics = model.val(data='data.yaml', split='val', verbose=True)

# Print mAP, precision, recall, etc.
print("=== Evaluation Metrics ===")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"Precision: {metrics.box.p:}")
print(f"Recall: {metrics.box.r}")
