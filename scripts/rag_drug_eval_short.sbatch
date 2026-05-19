#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=rag-drug-short
#SBATCH --output=/home/wshenah/project/logs/rag_drug_%j.out
#SBATCH --error=/home/wshenah/project/logs/rag_drug_%j.err

set -euo pipefail

MAX_STEPS="${MAX_STEPS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
QUERIES_FILE="${QUERIES_FILE:-/home/wshenah/project/rag_data/eval/drug_queries_50.jsonl}"
BASE_MODEL="${BASE_MODEL:-/home/wshenah/project/models/Qwen3-VL-8B-Thinking/}"
LORA_PATH="${LORA_PATH:-/home/wshenah/project/lora/v14-20260306-195347/checkpoint-1200}"
RAG_INDEX_DIR="${RAG_INDEX_DIR:-/home/wshenah/project/rag_index/openfda_label}"
RAG_MANIFEST_PATH="${RAG_MANIFEST_PATH:-/home/wshenah/project/rag/corpus_manifest_openfda_label.json}"

mkdir -p /home/wshenah/project/logs
mkdir -p /home/wshenah/project/eval_results

module load cuda12.2/toolkit/12.2.2
eval "$(conda shell.bash hook)"
conda activate qwenvl
export PYTHONNOUSERSITE=1

echo "=== RAG drug eval short start $(date) ==="
echo "node=$(hostname)"
echo "max_steps=${MAX_STEPS} max_new_tokens=${MAX_NEW_TOKENS}"
echo "queries_file=${QUERIES_FILE}"

python /home/wshenah/project/scripts/evaluate_rag_drug_queries_local.py \
  --base_model "${BASE_MODEL}" \
  --lora_path "${LORA_PATH}" \
  --max_steps "${MAX_STEPS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --queries_file "${QUERIES_FILE}" \
  --rag_index_dir "${RAG_INDEX_DIR}" \
  --rag_manifest_path "${RAG_MANIFEST_PATH}"

LATEST_REPORT="$(ls -1t /home/wshenah/project/eval_results/rag_drug_*/rag_drug_report.json | head -n 1)"
python /home/wshenah/project/scripts/plot_rag_drug_report.py --report "${LATEST_REPORT}"

echo "latest_report=${LATEST_REPORT}"
echo "latest_plot=$(dirname "${LATEST_REPORT}")/rag_drug_plot.png"
echo "=== RAG drug eval short end $(date) ==="

