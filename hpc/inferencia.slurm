#!/bin/bash
#SBATCH --account=
#SBATCH --job-name=inferencia_yolo
#SBATCH --output=saida_inferencia_%j.txt
#SBATCH --error=erro_inferencia_%j.txt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

cd ../preprocessing/inference
python inferenciaYolov.py
