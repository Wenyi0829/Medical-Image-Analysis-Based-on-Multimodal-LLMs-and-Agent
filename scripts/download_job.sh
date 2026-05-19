#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=00:30:00
#SBATCH --job-name=qwen-hf-download
#SBATCH --output=/home/wshenah/project/logs/hf_download_%j.out

module load cuda12.2/toolkit/12.2.2
conda activate qwenvl

export HF_HOME=/home/wshenah/project/hf_cache
export HF_ENDPOINT=https://hf-mirror.com

echo "Pre-downloading model at $(date)"
huggingface-cli download Qwen/Qwen3-VL-8B-Thinking \
    --local-dir /home/wshenah/project/models/Qwen3-VL-8B-Thinking \
    --resume-download \
    --max-workers 1 
echo "Download finished at $(date)"