#!/usr/bin/env bash
set -euo pipefail

# One-command pipeline:
# - submit Slurm job (on slogin-01 if running locally)
# - poll until node allocated
# - set up ssh tunnel to localhost:${LOCAL_PORT}
# - wait until /health responds
#
# Usage:
#   ./vllm_pipeline.sh
#   ./vllm_pipeline.sh --local-port 22002 --remote-port 22002
#
# Requirements:
# - From your laptop: ssh access to slogin-01 and (via jump) to compute nodes.
# - `sbatch`/`squeue` available on slogin-01.

USER_NAME="${USER_NAME:-wshenah}"

# Auto-detect login node when running on the cluster.
# - If already on a host named like "slogin-XX", use that.
# - Otherwise default to "slogin-01" (override via --login-host or $LOGIN_HOST).
_host_short="$(hostname -s 2>/dev/null || hostname || true)"
if [[ -z "${LOGIN_HOST:-}" ]]; then
  if [[ "${_host_short}" == slogin-* ]]; then
    LOGIN_HOST="${_host_short}"
  else
    LOGIN_HOST="slogin-01"
  fi
fi
JOB_SCRIPT="${JOB_SCRIPT:-/home/wshenah/project/scripts/vllm_job.sbatch}"

LOCAL_PORT="22002"
REMOTE_PORT="22002"
# If compute-node SSH is blocked, we can rely on reverse tunnel created by the job:
#   (login) 127.0.0.1:${LOGIN_PORT} -> (compute) 127.0.0.1:${REMOTE_PORT}
USE_REVERSE_TUNNEL="${USE_REVERSE_TUNNEL:-1}"
LOGIN_PORT=""
# Jobs can easily queue longer than 3 minutes.
WAIT_RUNNING_SECS="1800"
WAIT_HEALTH_SECS="180"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local-port) LOCAL_PORT="${2:?}"; shift 2;;
    --remote-port) REMOTE_PORT="${2:?}"; shift 2;;
    --wait-running) WAIT_RUNNING_SECS="${2:?}"; shift 2;;
    --wait-health) WAIT_HEALTH_SECS="${2:?}"; shift 2;;
    --login-host) LOGIN_HOST="${2:?}"; shift 2;;
    --user) USER_NAME="${2:?}"; shift 2;;
    --job-script) JOB_SCRIPT="${2:?}"; shift 2;;
    -h|--help)
      cat <<'EOF'
Usage: vllm_pipeline.sh [options]

Options:
  --local-port PORT      Local port to bind (default: 22002)
  --remote-port PORT     Remote vLLM port on compute node (default: 22002)
  --wait-running SECS    Max seconds to wait for Slurm job running (default: 1200)
  --wait-health SECS     Max seconds to wait for /health (default: 180)
  --login-host HOST      Slurm login node (default: auto-detect slogin-XX else slogin-01)
  --user USER            Username (default: wshenah)
  --job-script PATH      Path to sbatch script on cluster
EOF
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

is_on_login_host() {
  # Compare short hostnames (e.g., slogin-01)
  [[ "$(hostname -s 2>/dev/null || hostname)" == "${LOGIN_HOST}" ]]
}

run_on_login() {
  # Run a command on the login host, either locally (if we are already there)
  # or via ssh from another machine.
  if is_on_login_host; then
    bash -lc "$*"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${USER_NAME}@${LOGIN_HOST}" "bash -lc $(printf '%q' "$*")"
  fi
}

submit_job() {
  # Load Slurm module if needed, then submit.
  # Force non-interactive output (avoid pagers).
  run_on_login \
    "export PAGER=cat LESS='-FRSX' && \
     (command -v sbatch >/dev/null 2>&1 || module load slurm >/dev/null 2>&1 || true) && \
     command -v sbatch >/dev/null 2>&1 || { echo \"ERROR: sbatch not found on ${LOGIN_HOST}. Try: module load slurm\" >&2; exit 127; } && \
     sbatch --parsable \"${JOB_SCRIPT}\" | head -n 1"
}

first_hostname_from_nodelist() {
  # Slurm nodelist can be a single host (dgx-35) or a compact range (dgx-[01-04]).
  # Prefer expanding via scontrol when available; fall back to best-effort parsing.
  local nodelist="$1"
  local host=""

  if [[ "${nodelist}" == *"["* || "${nodelist}" == *","* ]]; then
    host="$(run_on_login "export PAGER=cat LESS='-FRSX' && \
                         (command -v scontrol >/dev/null 2>&1 || module load slurm >/dev/null 2>&1 || true) && \
                         scontrol show hostnames ${nodelist} 2>/dev/null | head -n 1 || true")"
  fi

  if [[ -z "${host}" ]]; then
    # Best-effort: take up to first comma.
    host="${nodelist%%,*}"
  fi

  # Trim whitespace/newlines just in case.
  host="$(echo "${host}" | tr -d '[:space:]')"
  echo "${host}"
}

poll_node() {
  local job_id="$1"
  local deadline=$(( $(date +%s) + WAIT_RUNNING_SECS ))
  local node=""
  local last_print_ts=0
  local print_every=20

  while [[ $(date +%s) -lt "${deadline}" ]]; do
    # %T=state, %N=nodelist, %R=reason (queueing reason / node state)
    local out
    out="$(run_on_login "export PAGER=cat LESS='-FRSX' && \
                         (command -v squeue >/dev/null 2>&1 || module load slurm >/dev/null 2>&1 || true) && \
                         squeue -j ${job_id} -h -o '%T|%N|%R' 2>/dev/null | head -n 1 || true")"
    # If job finished quickly, squeue may return empty.
    if [[ -z "${out}" ]]; then
      sleep 2
      continue
    fi
    local state nodelist reason
    state="$(cut -d'|' -f1 <<<"${out}")"
    nodelist="$(cut -d'|' -f2 <<<"${out}")"
    reason="$(cut -d'|' -f3- <<<"${out}")"
    local now
    now="$(date +%s)"
    if (( now - last_print_ts >= print_every )); then
      # IMPORTANT: log to stderr so callers can safely capture stdout.
      echo "Job ${job_id}: state=${state} node=${nodelist} reason=${reason}" >&2
      last_print_ts="${now}"
    fi
    if [[ "${state}" == "RUNNING" && -n "${nodelist}" && "${nodelist}" != "(null)" && "${nodelist}" != "None" ]]; then
      node="$(first_hostname_from_nodelist "${nodelist}")"
      break
    fi
    sleep 2
  done

  if [[ -z "${node}" ]]; then
    echo "ERROR: timed out waiting for job ${job_id} to be RUNNING with allocated node." >&2
    echo "Tip: check on login node: squeue -j ${job_id}" >&2
    echo "Tip: show details: scontrol show job ${job_id}" >&2
    exit 1
  fi

  echo "${node}"
}

start_tunnel_bg() {
  local node="$1"
  local pidfile="/tmp/vllm_tunnel_${LOCAL_PORT}.pid"

  # If there's an old tunnel pidfile, try to clean up.
  if [[ -f "${pidfile}" ]]; then
    local oldpid
    oldpid="$(cat "${pidfile}" || true)"
    if [[ -n "${oldpid}" ]]; then
      kill "${oldpid}" >/dev/null 2>&1 || true
    fi
    rm -f "${pidfile}" || true
  fi

  # Run tunnel in background (-f) after auth.
  #
  # Preferred: forward to login node (job should establish a reverse tunnel from compute->login).
  if [[ "${USE_REVERSE_TUNNEL}" == "1" ]]; then
    if [[ -z "${LOGIN_PORT}" ]]; then
      echo "ERROR: USE_REVERSE_TUNNEL=1 but LOGIN_PORT is empty." >&2
      exit 1
    fi
    if is_on_login_host; then
      ssh -f -N \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -L "${LOCAL_PORT}:127.0.0.1:${LOGIN_PORT}" \
        "${USER_NAME}@${LOGIN_HOST}"
    else
      ssh -f -N \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -L "${LOCAL_PORT}:127.0.0.1:${LOGIN_PORT}" \
        "${USER_NAME}@${LOGIN_HOST}"
    fi
  else
    # Fallback: ssh directly to compute node (may be blocked by cluster policy).
    if is_on_login_host; then
      ssh -f -N \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
        "${USER_NAME}@${node}"
    else
      ssh -f -N \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -J "${USER_NAME}@${LOGIN_HOST}" \
        -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
        "${USER_NAME}@${node}"
    fi
  fi

  # Best-effort: record a pid (ssh -f detaches; $! is not reliable across platforms).
  # We store the newest matching ssh process pid.
  local pid
  pid="$(pgrep -n -f "ssh .* -L ${LOCAL_PORT}:127\\.0\\.0\\.1:${REMOTE_PORT} .*${USER_NAME}@${node}" || true)"
  if [[ -n "${pid}" ]]; then
    echo "${pid}" > "${pidfile}"
  fi

  echo "Tunnel established. Local endpoints:"
  echo "  http://127.0.0.1:${LOCAL_PORT}/docs"
  echo "  http://127.0.0.1:${LOCAL_PORT}/v1/models"
  echo "To stop tunnel (best-effort):"
  echo "  [[ -f ${pidfile} ]] && kill \$(cat ${pidfile})"
}

wait_health() {
  local deadline=$(( $(date +%s) + WAIT_HEALTH_SECS ))
  while [[ $(date +%s) -lt "${deadline}" ]]; do
    if curl -fsS "http://127.0.0.1:${LOCAL_PORT}/health" >/dev/null 2>&1; then
      echo "vLLM is healthy on http://127.0.0.1:${LOCAL_PORT}"
      return 0
    fi
    sleep 2
  done
  echo "WARNING: tunnel is up but /health didn't respond within ${WAIT_HEALTH_SECS}s." >&2
  echo "Try: curl -v http://127.0.0.1:${LOCAL_PORT}/health" >&2
}

is_local_port_free() {
  local port="$1"
  # Robust check: actually try binding the local port.
  # Exit 0 means bind succeeded (port free), exit 1 means in use.
  python - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    s.close()
sys.exit(0)
PY
}

choose_local_port() {
  # Auto-fallback only for default port workflow: 22002 -> 22003 -> 22004.
  # If user explicitly passed a different local port, keep it as-is.
  if [[ "${LOCAL_PORT}" != "22002" ]]; then
    return 0
  fi
  local candidate
  for candidate in 22002 22003 22004; do
    if is_local_port_free "${candidate}"; then
      if [[ "${candidate}" != "${LOCAL_PORT}" ]]; then
        echo "Local port ${LOCAL_PORT} is in use. Auto-switching to ${candidate}." >&2
      fi
      LOCAL_PORT="${candidate}"
      return 0
    fi
  done
  echo "ERROR: local ports 22002/22003/22004 are all in use. Please free one, or pass --local-port <PORT>." >&2
  exit 1
}

main() {
  choose_local_port
  echo "Submitting Slurm job on ${LOGIN_HOST} using ${JOB_SCRIPT} ..."
  local job_id
  job_id="$(submit_job)"
  job_id="$(echo "${job_id}" | tr -d '[:space:]')"
  if [[ -z "${job_id}" ]]; then
    echo "ERROR: failed to submit job (empty job_id). This usually means Slurm isn't available on ${LOGIN_HOST} (try: module load slurm) or sbatch returned an error." >&2
    exit 1
  fi
  echo "Submitted job_id=${job_id}"

  echo "Waiting for allocated compute node ..."
  local node
  node="$(poll_node "${job_id}")"
  echo "Allocated node: ${node}"

  # Match the port selection in vllm_job.sbatch when reverse tunnel is enabled.
  # (login) 127.0.0.1:${LOGIN_PORT} -> (compute) 127.0.0.1:${REMOTE_PORT}
  LOGIN_PORT="$((30000 + (job_id % 20000)))"

  if [[ "${USE_REVERSE_TUNNEL}" == "1" ]]; then
    echo "Starting ssh tunnel in background (localhost:${LOCAL_PORT} -> ${LOGIN_HOST}:127.0.0.1:${LOGIN_PORT}) ..."
  else
    echo "Starting ssh tunnel in background (localhost:${LOCAL_PORT} -> ${node}:127.0.0.1:${REMOTE_PORT}) ..."
  fi
  start_tunnel_bg "${node}"

  echo "Waiting for /health ..."
  wait_health

  echo "Done. You can now run clients with base_url=http://127.0.0.1:${LOCAL_PORT}/v1"
  echo
  echo "Web chat UI (copy/paste):"
  echo "  VLLM_PORT=${LOCAL_PORT} python /home/wshenah/project/scripts/api.py --web"
  echo "Then open: http://127.0.0.1:7860"
}

main

