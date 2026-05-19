#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --job-name=rag-effect-short
#SBATCH --output=/home/wshenah/project/logs/rag_effect_%j.out
#SBATCH --error=/home/wshenah/project/logs/rag_effect_%j.err

set -euo pipefail

SAMPLE_SIZE="${SAMPLE_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-2}"
MAIN_METRIC="${MAIN_METRIC:-rouge_l_mean}"
BASE_MODEL="${BASE_MODEL:-/home/wshenah/project/models/Qwen3-VL-8B-Thinking/}"
LORA_PATH="${LORA_PATH:-/home/wshenah/project/lora/v14-20260306-195347/checkpoint-1200}"
VAL_DATASET="${VAL_DATASET:-/home/wshenah/project/lora/v14-20260306-195347/val_dataset.jsonl}"
RAG_INDEX_DIR="${RAG_INDEX_DIR:-/home/wshenah/project/rag_index/openfda_label}"
RAG_MANIFEST_PATH="${RAG_MANIFEST_PATH:-/home/wshenah/project/rag/corpus_manifest_openfda_label.json}"

mkdir -p /home/wshenah/project/logs
mkdir -p /home/wshenah/project/eval_results

module load cuda12.2/toolkit/12.2.2
eval "$(conda shell.bash hook)"
conda activate qwenvl
export PYTHONNOUSERSITE=1

echo "=== RAG effect short eval start $(date) ==="
echo "node=$(hostname)"
echo "sample_size=${SAMPLE_SIZE}, max_steps=${MAX_STEPS}, main_metric=${MAIN_METRIC}"

python /home/wshenah/project/scripts/evaluate_rag_effect_local.py \
  --sample_size "${SAMPLE_SIZE}" \
  --base_model "${BASE_MODEL}" \
  --lora_path "${LORA_PATH}" \
  --val_dataset "${VAL_DATASET}" \
  --max_steps "${MAX_STEPS}" \
  --main_metric "${MAIN_METRIC}" \
  --rag_index_dir "${RAG_INDEX_DIR}" \
  --rag_manifest_path "${RAG_MANIFEST_PATH}"

# Find newest report and plot it.
LATEST_REPORT="$(ls -1t /home/wshenah/project/eval_results/rag_effect_local_*/rag_effect_report.json | head -n 1)"
python /home/wshenah/project/scripts/plot_rag_effect.py --report "${LATEST_REPORT}"

echo "latest_report=${LATEST_REPORT}"
echo "latest_plot=$(dirname "${LATEST_REPORT}")/rag_effect_plot.png"
echo "=== RAG effect short eval end $(date) ==="

