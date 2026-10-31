#!/bin/bash
# poisson_load -- one replica under Poisson arrivals: per-request residence, occupancy, power and
# natural drains with oracle residuals (the single-GPU model figures). One model per array
# task, every workload on one engine load, the replica at its fleet role cap (64 for the
# 8B-class, 128 for the small models) so the cell matches the capacities and fleets.
#
#   sbatch --array=0-3 --export=ALL,REPO_DIR=<checkout> slurm/poisson_load.sbatch
#   sbatch --array=0 --export=ALL,REPO_DIR=<checkout>,SMOKE=1,WORKLOADS=sharegpt:humaneval ...
#
# COLONS in WORKLOADS (SLURM's --export splits on commas).
#SBATCH -J poisson_load
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH -t 08:00:00
#SBATCH -o slurm_logs/%x_%A_%a.out
#SBATCH -e slurm_logs/%x_%A_%a.err
# Cluster-specific SLURM settings: the partition (-p) and GPU type (--gres) below are
# placeholders; set them for your cluster here or on the sbatch command line (sbatch -p
# <partition> --gres=gpu:<type>:<n> -A <account> ...), which takes precedence over these lines.

set -uo pipefail
# The repository's root: the directory you submit from (SLURM_SUBMIT_DIR), or REPO_DIR.
REPO_DIR=${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO_DIR" || { echo "REPO_DIR $REPO_DIR not found"; exit 1; }
[ -f lib/env.py ] || { echo "REPO_DIR=$REPO_DIR is not the repository's root (no lib/env.py); set REPO_DIR"; exit 1; }
mkdir -p slurm_logs
PY=${PY:-python}   # the interpreter of an environment with requirements-gpu.txt installed
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# On a V100 (submit with --gres=gpu:v100:1): Triton attention and the native sampler,
# as in concurrency_sweep_v100.sbatch.
if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -q V100; then
  export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-TRITON_ATTN} VLLM_USE_FLASHINFER_SAMPLER=0
  echo "=== V100: TRITON_ATTN, native sampler"
fi

# The rate grid is placed from a MEASURED capacity_bisection capacity when one exists; the script
# refuses a cell with neither a measurement nor the cell's own t0/t1. Every phase is
# judged by the waiting-list rule regardless.
if ! grep -q "no absolute floor" capacity_bisection.py; then
  echo "REFUSING: $REPO_DIR/capacity_bisection.py lacks the no-absolute-floor stability rule."
  exit 1
fi

MODELS=(llama32-1b qwen3-4b llama31-8b qwen3-8b)
CAPS=(128 128 64 64)
I=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODEL:-${MODELS[$I]}}
CAP=${CAP:-${CAPS[$I]}}
WORKLOADS=$(printf '%s' "${WORKLOADS:-sharegpt:humaneval:fixed_128_128:fixed_1024_1024:arxiv_summarization}" | tr ':' ',')
EXTRA=()
# SMOKE=1: a two-phase shakeout, tagged "smoke" unless OUT_TAG says otherwise; every reader skips it
if [ -n "${SMOKE:-}" ]; then EXTRA+=(--smoke); OUT_TAG=${OUT_TAG:-smoke_${MODEL}__$(printf '%s' "$WORKLOADS" | tr ',' '-')}; fi
[ -n "${FRACTIONS:-}" ] && EXTRA+=(--fractions "$(printf '%s' "$FRACTIONS" | tr ':' ',')")
[ -n "${RATES:-}" ] && EXTRA+=(--rates "$(printf '%s' "$RATES" | tr ':' ',')")

# Opt-in: read a big checkpoint into the page cache in parallel first (prefetch_weights.sh).
if [ -n "${PREFETCH:-}" ]; then bash slurm/prefetch_weights.sh "$MODEL" || echo "prefetch failed; loading cold"; fi
echo "=== poisson_load  model=$MODEL cap=$CAP workloads=$WORKLOADS extra=${EXTRA[*]:-none}  $(date -Is)"
$PY poisson_load.py \
    --model "$MODEL" --workload "$WORKLOADS" --max-num-seqs "$CAP" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
    --warmup-s "${WARMUP_S:-30}" --measure-s "${MEASURE_S:-120}" --max-measure-s "${MAX_MEASURE_S:-480}" \
    --n-drains "${N_DRAINS:-3}" --pool-size 4096 \
    "${EXTRA[@]}" \
    ${OUT_TAG:+--out-tag "$OUT_TAG"}
echo "=== poisson_load $MODEL exit=$?  $(date -Is)"
