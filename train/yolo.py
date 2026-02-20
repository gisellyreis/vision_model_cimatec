import os
import sys
import torch
import mlflow
import mlflow.pytorch
from ultralytics import YOLO, settings

settings.update({"mlflow": False})
os.environ["REPORTTO"] = "none"

mlflow.set_experiment("Detecção_Marítima_YOLO")

def train_model(model_path, run_name, data_path='datasets/data.yaml'):
    
    if not os.path.exists(data_path):
        print(f"❌ ERRO: Arquivo de configuração {data_path} não encontrado!")
        return

    device = 0
    print(f"🚀 Iniciando treino na GPU: {torch.cuda.get_device_name(0)}")

    torch.cuda.empty_cache()

    run = mlflow.start_run(run_name=run_name)
    run_id = run.info.run_id

    try:
       
        train_params = {
            "epochs": 100,
            "imgsz": 640,
            "batch": 64,
            "patience": 20,          
            "cos_lr": True,
            "mosaic": 0.5,           
            "mixup": 0.1,
            "close_mosaic": 2,      
            "lr0": 0.001,            
            "device": device,
            "label_smoothing": 0.1   
        }

        mlflow.log_params(train_params)
        
        mlflow.log_param("base_model", model_path)
        mlflow.log_param("dataset_config", data_path)

        model = YOLO(model_path)

        results = model.train(
            data=data_path,
            name=run_name,
            plots=True,
            workers=0,
            **train_params
        )

        if results is not None:
            metrics = {
                "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
                "precision": results.results_dict.get("metrics/precision(B)", 0),
                "recall": results.results_dict.get("metrics/recall(B)", 0),
                "inference_speed_ms": results.speed.get("inference", 0)
            }
            mlflow.log_metrics(metrics)
            print(f"✅ Métricas registradas com sucesso!")

        print("📦 Exportando para ONNX...")
        onnx_file = model.export(format="onnx", opset=10)
        
        mlflow.log_artifact(onnx_file, artifact_path="onnx_model")
        
        mlflow.pytorch.log_model(
            model.model, 
            artifact_path="yolo_model",
            registered_model_name="YOLO_Maritime_Detection"
        )

        print(f"✨ Treino finalizado! Run ID: {run_id}")

    except Exception as e:
        print(f"⚠️ O treino falhou! Erro: {e}")
        mlflow.set_tag("status", "failed")
        raise e

    finally:
        mlflow.end_run()

if __name__ == '__main__':
    version = sys.argv[1] if len(sys.argv) > 1 else 'v8'

    model_path = f"yolo{version}s.pt"
    run_name = f"yolo{version}_sds"

    train_model(model_path, run_name)