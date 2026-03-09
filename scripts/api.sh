#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=qwen-api
#SBATCH --output=/home/wshenah/project/logs/api_%j.out
#SBATCH --error=/home/wshenah/project/logs/api_%j.err

module load cuda12.2/toolkit/12.2.2

eval "$(conda shell.bash hook)"
conda activate qwenvl


echo "=== API Server starting at $(date) ==="
echo "GPU Info: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

vllm serve /home/wshenah/project/models/Qwen3-VL-8B-Thinking \
  --host 0.0.0.0 \
  --port 22002 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --dtype bfloat16 \
  --enforce-eager \
  --mm-encoder-tp-mode data

echo "=== API Server exited at $(date) ==="