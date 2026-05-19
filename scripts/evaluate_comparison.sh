#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --job-name=eval
#SBATCH --output=/home/wshenah/project/logs/eval_%j.out
#SBATCH --error=/home/wshenah/project/logs/eval_%j.out

module load cuda12.2/toolkit/12.2.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwenvl

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/wshenah/project/hf_cache

echo "Starting Evaluation"
echo "GPU Info: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python "/home/wshenah/project/scripts/evaluate_models.py" --sample_size 987 --eval_baseline

echo "Evaluation completed at $(date)"