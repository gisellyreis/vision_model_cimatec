#!/bin/bash
#SBATCH --job-name=inferencia_yolo
#SBATCH --output=saida_inferencia_%j.txt
#SBATCH --error=erro_inferencia_%j.txt
#SBATCH --time=00:10:00
#SBATCH --partition=gpu           
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

source ~/teste_inferencia/venv/bin/activate
cd ~/teste_inferencia
python inferenciaYolov8.py
