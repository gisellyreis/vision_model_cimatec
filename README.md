# How to Train the Models
All models below are configured to be trained on the SeaDronesSee dataset.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Training Models

### For YOLO

- YOLOv5
```bash
python train/yolo.py v5
```

- YOLOv8 (default)
```bash
python train/yolo.py v8
```

- YOLOv10
```bash
python train/yolo.py v10
```

**Output Location**: After training finishes, your trained models and metrics will be located in: runs/detect/yolo[version]_sds/


### For Faster R-CNN

- Download Weights
```bash
python fasterrcnn.py
```

- Start Training
```bash
python train/CNN.py
```

**Output Location**: After training finishes, your trained model will be located in: checkpoints/