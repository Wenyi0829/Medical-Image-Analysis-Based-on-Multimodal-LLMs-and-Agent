#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=qwen-inference
#SBATCH --output=/home/wshenah/project/logs/inference_%j.out
#SBATCH --error=/home/wshenah/project/logs/inference_%j.err

module load cuda12.2/toolkit/12.2.2
source activate qwenvl
pip install flash-attn --no-build-isolation --no-cache-dir

echo "Inference started at $(date)"
python /home/wshenah/project/scripts/inference2.py
echo "Inference finished at $(date)"
