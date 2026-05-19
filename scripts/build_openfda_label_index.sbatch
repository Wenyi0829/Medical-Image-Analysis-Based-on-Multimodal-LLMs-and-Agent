#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=openfda-rag
#SBATCH --output=/home/wshenah/project/logs/openfda_rag_%j.out
#SBATCH --error=/home/wshenah/project/logs/openfda_rag_%j.err

set -euo pipefail

mkdir -p /home/wshenah/project/logs /home/wshenah/project/rag_index/openfda_label

echo "=== OpenFDA RAG build starting at $(date) ==="
echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"

module load cuda12.2/toolkit/12.2.2 || true

eval "$(conda shell.bash hook)"
conda activate qwenvl

export PYTHONNOUSERSITE=1
export HF_HOME="/home/wshenah/project/hf_cache"

python /home/wshenah/project/scripts/build_openfda_label_index.py \
  --manifest /home/wshenah/project/rag/corpus_manifest_openfda_label.json \
  --index_dir /home/wshenah/project/rag_index/openfda_label \
  --embed_model sentence-transformers/all-MiniLM-L6-v2 \
  --top_k 3 \
  --rebuild \
  --chunk_size_words 800 \
  --chunk_overlap_words 120 \
  --min_chunk_words 30 \
  --min_text_chars 200

echo "=== OpenFDA RAG build finished at $(date) ==="

