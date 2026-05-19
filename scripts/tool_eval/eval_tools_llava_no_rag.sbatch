#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --time=00:28:00
#SBATCH --job-name=tool-eval
#SBATCH --output=/home/wshenah/project/logs/tool_eval_%j.out
#SBATCH --error=/home/wshenah/project/logs/tool_eval_%j.err

set -euo pipefail

DATASET_JSON="${DATASET_JSON:-/home/wshenah/LLaVA-Med/data/instruct/llava_med_qwen_format_fixed.json}"
BASE_MODEL="${BASE_MODEL:-/home/wshenah/project/models/Qwen3-VL-8B-Thinking}"
LORA_PATH="${LORA_PATH:-/home/wshenah/project/lora/v14-20260306-195347/checkpoint-1200}"

# Keep this small to reduce queue risk; adjust as needed.
LIMIT="${LIMIT:-30}"
MAX_STEPS="${MAX_STEPS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"

mkdir -p /home/wshenah/project/logs

echo "=== tool eval starting at $(date) ==="
echo "DATASET_JSON=${DATASET_JSON}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "LORA_PATH=${LORA_PATH}"
echo "LIMIT=${LIMIT}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"

eval "$(conda shell.bash hook)"
conda activate qwenvl

export HF_HOME=/home/wshenah/project/hf_cache
export PYTHONNOUSERSITE=1

python /home/wshenah/project/scripts/tool_eval/evaluate_tools_llava_med_no_rag.py \
  --dataset_json "${DATASET_JSON}" \
  --base_model "${BASE_MODEL}" \
  --lora_path "${LORA_PATH}" \
  --limit "${LIMIT}" \
  --max_steps "${MAX_STEPS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}"

echo "=== tool eval finished at $(date) ==="

