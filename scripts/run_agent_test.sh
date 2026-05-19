#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=agent_eval
#SBATCH --output=/home/wshenah/project/logs/agent_eval_%j.out
#SBATCH --error=/home/wshenah/project/logs/agent_eval_%j.out

# 加载环境
module load cuda12.2/toolkit/12.2.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwenvl

# 显存优化配置
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/wshenah/project/hf_cache
export TOKENIZERS_PARALLELISM=false

echo "=========================================="
echo "Starting Medical VQA Agent Evaluation"
echo "Time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

# 运行评估脚本
# 注意：这里只加载 LoRA 模型，不加载 Baseline
python "/home/wshenah/project/scripts/evaluate_agent.py" \
    --sample_size 100 \
    --max_steps 3 \
    --output_dir "/home/wshenah/project/eval_results/agent_test_run"

echo "=========================================="
echo "Evaluation Completed at $(date)"
echo "=========================================="