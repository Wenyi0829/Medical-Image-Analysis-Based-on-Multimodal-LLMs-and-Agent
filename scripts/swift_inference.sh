#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --job-name=infer
#SBATCH --output=/home/wshenah/project/logs/infer_%j.out
#SBATCH --error=/home/wshenah/project/logs/infer_%j.err

module load cuda12.2/toolkit/12.2.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwenvl

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export IMAGE_MAX_TOKEN_NUM=1024
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=16
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$HOME/.cache/huggingface


swift infer \
    --model "/home/wshenah/project/models/Qwen3-VL-8B-Thinking/" \
    --adapters "/home/wshenah/project/lora/v9-20260303-144202/checkpoint-50" \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --stream true \
    --max_length 4096
