#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --job-name=qwen-vllm
#SBATCH --output=/home/wshenah/project/logs/vllm_%j.out
#SBATCH --error=/home/wshenah/project/logs/vllm_%j.err

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/wshenah/project/models/Qwen3-VL-8B-Thinking}"
PORT="${PORT:-22002}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
DTYPE="${DTYPE:-bfloat16}"

# Optional reverse tunnel from compute -> login node.
# This avoids needing to ssh INTO the compute node (often blocked by PAM/Slurm policy).
ENABLE_REVERSE_TUNNEL="${ENABLE_REVERSE_TUNNEL:-1}"
LOGIN_HOST="${LOGIN_HOST:-${SLURM_SUBMIT_HOST:-}}"
# Choose a stable, non-privileged port based on job id if not provided.
LOGIN_PORT="${LOGIN_PORT:-$((30000 + (${SLURM_JOB_ID:-0} % 20000)))}"

mkdir -p /home/wshenah/project/logs

echo "=== vLLM starting at $(date) ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "HOSTNAME=$(hostname)"
echo "GPU Info: $(nvidia-smi --query-gpu=name --format=csv,noheader || true)"
echo "MODEL_PATH=${MODEL_PATH}"
echo "BIND=${HOST}:${PORT}"
if [[ "${ENABLE_REVERSE_TUNNEL}" == "1" && -n "${LOGIN_HOST}" ]]; then
  echo "REVERSE_TUNNEL=enabled"
  echo "LOGIN_HOST=${LOGIN_HOST}"
  echo "LOGIN_PORT=${LOGIN_PORT}"
else
  echo "REVERSE_TUNNEL=disabled"
fi

# Write node/port info for external scripts
INFO_FILE="/home/wshenah/project/logs/vllm_${SLURM_JOB_ID:-unknown}.info"
{
  echo "job_id=${SLURM_JOB_ID:-}"
  echo "node=$(hostname)"
  echo "host=${HOST}"
  echo "port=${PORT}"
  echo "reverse_tunnel=${ENABLE_REVERSE_TUNNEL}"
  echo "login_host=${LOGIN_HOST}"
  echo "login_port=${LOGIN_PORT}"
  echo "model_path=${MODEL_PATH}"
  echo "started_at=$(date -Iseconds)"
} > "${INFO_FILE}"
echo "Wrote info file: ${INFO_FILE}"

module load cuda12.2/toolkit/12.2.2

eval "$(conda shell.bash hook)"
conda activate qwenvl

# Avoid picking up user-site packages (~/.local) that can conflict
# with the conda env and break vLLM imports.
export PYTHONNOUSERSITE=1

echo "Python: $(which python)"
echo "Checking transformers/huggingface_hub imports..."
python - <<'PY'
import transformers
import huggingface_hub
print("transformers:", transformers.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
PY

if [[ "${ENABLE_REVERSE_TUNNEL}" == "1" && -n "${LOGIN_HOST}" ]]; then
  # Best-effort: keep a reverse tunnel open:
  #   (login) 127.0.0.1:${LOGIN_PORT}  ->  (compute) 127.0.0.1:${PORT}
  #
  # Assumes compute nodes can ssh to login nodes for internal traffic.
  # If this fails in your environment, set ENABLE_REVERSE_TUNNEL=0.
  ( ssh -N \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -R "127.0.0.1:${LOGIN_PORT}:127.0.0.1:${PORT}" \
      "${USER}@${LOGIN_HOST}" \
      >/home/wshenah/project/logs/vllm_${SLURM_JOB_ID:-unknown}.tunnel.out 2>&1 ) &
  echo "Started reverse tunnel process (compute -> ${LOGIN_HOST}:${LOGIN_PORT})"
fi

vllm serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype "${DTYPE}" \
  --enforce-eager \
  --mm-encoder-tp-mode data

echo "=== vLLM exited at $(date) ==="
