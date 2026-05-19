#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --job-name=case-fig-dump
#SBATCH --output=/home/wshenah/project/logs/case_study_fig_%j.out
#SBATCH --error=/home/wshenah/project/logs/case_study_fig_%j.err

set -euo pipefail

module load cuda12.2/toolkit/12.2.2 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate qwenvl

export HF_HOME=/home/wshenah/project/hf_cache
export PYTHONNOUSERSITE=1

mkdir -p /home/wshenah/project/logs
mkdir -p /home/wshenah/project/eval_results/case_study_figure_dump

echo "=== dump_case_study_figures GPU $(date) host=$(hostname) ==="

python /home/wshenah/project/scripts/dump_case_study_figures.py \
  --case image \
  --out_dir /home/wshenah/project/eval_results/case_study_figure_dump

echo "=== done $(date) ==="
