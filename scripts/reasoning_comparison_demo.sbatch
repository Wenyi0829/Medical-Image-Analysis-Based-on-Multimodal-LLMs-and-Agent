#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=00:28:00
#SBATCH --job-name=reason-compare
#SBATCH --output=/home/wshenah/project/logs/reasoning_comparison_%j.out
#SBATCH --error=/home/wshenah/project/logs/reasoning_comparison_%j.err

set -euo pipefail

CASES_JSONL="${CASES_JSONL:-/home/wshenah/project/eval_results/defense_demo/defense_demo_cases.jsonl}"
OUT_JSON="${OUT_JSON:-/home/wshenah/project/eval_results/defense_demo/reasoning_comparison_latest.json}"
BASE_MODEL="${BASE_MODEL:-/home/wshenah/project/models/Qwen3-VL-8B-Thinking/}"
LORA_PATH="${LORA_PATH:-/home/wshenah/project/lora/v14-20260306-195347/checkpoint-1200}"
RAG_INDEX_DIR="${RAG_INDEX_DIR:-/home/wshenah/project/rag_index/openfda_label}"
RAG_MANIFEST="${RAG_MANIFEST_PATH:-/home/wshenah/project/rag/corpus_manifest_openfda_label.json}"
MAX_STEPS="${MAX_STEPS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
# 若排队压力大，可 export SKIP_BASELINE=1 以跳过基座对照（约省 4 次推理）
SKIP_BASELINE_FLAG=""
if [[ "${SKIP_BASELINE:-0}" == "1" ]]; then
  SKIP_BASELINE_FLAG="--skip_baseline"
fi

mkdir -p /home/wshenah/project/logs
mkdir -p "$(dirname "${OUT_JSON}")"

module load cuda12.2/toolkit/12.2.2
eval "$(conda shell.bash hook)"
conda activate qwenvl

export HF_HOME=/home/wshenah/project/hf_cache
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

echo "=== reasoning comparison $(date) ==="
echo "node=$(hostname)"
echo "CASES_JSONL=${CASES_JSONL}"
echo "OUT_JSON=${OUT_JSON}"
echo "time_limit=28min (adjust #SBATCH --time if runs timeout)"

python /home/wshenah/project/scripts/run_reasoning_comparison_demo.py \
  --cases_jsonl "${CASES_JSONL}" \
  --output_json "${OUT_JSON}" \
  --base_model "${BASE_MODEL}" \
  --lora_path "${LORA_PATH}" \
  --rag_index_dir "${RAG_INDEX_DIR}" \
  --rag_manifest_path "${RAG_MANIFEST}" \
  --max_steps "${MAX_STEPS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  ${SKIP_BASELINE_FLAG}

echo "=== done $(date) ==="
