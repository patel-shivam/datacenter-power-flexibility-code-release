#!/bin/bash
# capacity_bisection -- open-loop capacity for ONE model (1x H100), by bisection on arrival rate.
#
#   sbatch --array=0-8 slurm/capacity_bisection.sbatch
#
# Capacity is judged from the scheduler's WAITING list, not from unfinished requests --
# the latter is running+waiting and grows with offered load by Little's Law even when the
# queue is stable, which halves every capacity derived from it.
#SBATCH -J capacity_bisection
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH -t 08:00:00
#SBATCH -o slurm_logs/%x_%A_%a.out
#SBATCH -e slurm_logs/%x_%A_%a.err
# Cluster-specific SLURM settings: the partition (-p) and GPU type (--gres) below are
# placeholders; set them for your cluster here or on the sbatch command line (sbatch -p
# <partition> --gres=gpu:<type>:<n> -A <account> ...), which takes precedence over these lines.

set -euo pipefail
# The repository's root: the directory you submit from (SLURM_SUBMIT_DIR), or REPO_DIR.
REPO_DIR=${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO_DIR" || { echo "REPO_DIR $REPO_DIR not found"; exit 1; }
[ -f lib/env.py ] || { echo "REPO_DIR=$REPO_DIR is not the repository's root (no lib/env.py); set REPO_DIR"; exit 1; }
mkdir -p slurm_logs
PY=${PY:-python}   # the interpreter of an environment with requirements-gpu.txt installed
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# The stability rule this capacity is judged by has no absolute floor: a fixed tolerance of
# 0.5/s of queue growth passes growing queues and inflates capacities, so the job refuses an
# capacity_bisection.py whose rule lacks "no absolute floor".
if ! grep -q "no absolute floor" capacity_bisection.py; then
  echo "REFUSING: $(pwd)/capacity_bisection.py lacks the no-absolute-floor stability rule."
  exit 1
fi

MODELS=(llama32-1b qwen3-1.7b llama32-3b qwen3-4b mistral-7b llama31-8b qwen3-8b qwen3-14b mistral-24b)
MODEL=${MODEL:-${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}}
WORKLOAD=${WORKLOAD:-sharegpt}

CAP_ARG=()
if [ -n "${MAX_NUM_SEQS:-}" ]; then CAP_ARG=(--max-num-seqs "$MAX_NUM_SEQS"); fi
# SMOKE=1: a two-step shakeout, not a measurement; tagged "smoke", which every reader of results/
# (capacity lookups included) skips, so it can never supersede a measured capacity.
SMOKE_ARG=()
if [ -n "${SMOKE:-}" ]; then
  SMOKE_ARG=(--smoke --out-tag "smoke_${MODEL}__${WORKLOAD}__cap${MAX_NUM_SEQS:-grid}"); echo "=== smoke run"
fi
echo "=== capacity_bisection  model=$MODEL  workload=$WORKLOAD  arrival=${ARRIVAL:-poisson}"
$PY capacity_bisection.py \
    --model "$MODEL" --workload "$WORKLOAD" --arrival "${ARRIVAL:-poisson}" \
    --probe-s "${PROBE_S:-90}" --warmup-s "${WARMUP_S:-25}" --iters "${ITERS:-6}" \
    "${CAP_ARG[@]}" "${SMOKE_ARG[@]}"
