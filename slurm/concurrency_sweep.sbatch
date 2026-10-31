#!/bin/bash
# concurrency_sweep -- concurrency sweep for ONE model across several workloads (1x H100).
#
# Submit as an array, one model per task, so the nine models run in parallel across
# whatever GPUs are free:
#
#   sbatch --array=0-8 slurm/concurrency_sweep.sbatch
#   sbatch --array=0-8 --export=ALL,WORKLOADS=sharegpt:splitwise_code slurm/concurrency_sweep.sbatch
#
# NOTE the COLONS. SLURM's --export splits its argument on commas, so a comma-separated
# value silently becomes several variables: --export=ALL,WORKLOADS=a,b sets WORKLOADS=a
# and a variable named b, and the job then sweeps only workload a. This script therefore
# takes a colon-separated list and translates it.
#
# The time limit is deliberately generous. Model load dominates the first minutes and is
# dominated in turn by Lustre: a cold read of one 4 GB shard of Llama-3.1-8B was measured
# at 8m52s. That is also why every workload is swept on a SINGLE engine load rather than
# one job per (model, workload).
#SBATCH -J concurrency_sweep
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH -t 10:00:00
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
# A live Hub lookup hangs on compute nodes; everything must already be in the cache.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

MODELS=(llama32-1b qwen3-1.7b llama32-3b qwen3-4b mistral-7b llama31-8b qwen3-8b qwen3-14b mistral-24b)
MODEL=${MODEL:-${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}}
# colon-separated in, comma-separated out (see the --export note above)
WORKLOADS=$(printf '%s' "${WORKLOADS:-fixed_128_128:sharegpt}" | tr ':' ',')

# Optional concurrency override (CONCURRENCY). Empty means "use the model's own grid".
#
# Why it exists: a level above the KV ceiling is not a measurement of P(b) at that b.
# The engine cannot hold that many sequences, so the sweep plateaus and the fit absorbs a
# MEMORY limit as a feature of the power curve. Pre-flight with
# feasible.kv_concurrency_ceiling and keep every level under 0.9*c_kv -- on the 1:1
# ladder that binds only for qwen3-8b and qwen3-4b at fixed_1024_1024 (c_kv 262 and 299),
# which drop their top level. Since --concurrency applies to the whole job, a model whose
# grid must be truncated for ONE workload gets that workload in its own submission.
# COLON-separated in, comma-separated out -- the same translation WORKLOADS gets above,
# and for the same reason. Passing --export=ALL,CONCURRENCY=1,2,4 sets CONCURRENCY=1 and
# then variables named 2 and 4, so the sweep silently runs a SINGLE level instead of the
# grid.
CONC_ARG=()
if [ -n "${CONCURRENCY:-}" ]; then
  CONC=$(printf '%s' "$CONCURRENCY" | tr ':' ',')
  CONC_ARG=(--concurrency "$CONC")
  echo "=== concurrency override: $CONC"
fi

# Opt-in: read a big checkpoint into the page cache in parallel first (prefetch_weights.sh).
if [ -n "${PREFETCH:-}" ]; then bash slurm/prefetch_weights.sh "$MODEL" || echo "prefetch failed; loading cold"; fi

# Opt-in: KV_GUARD=1 drops the grid levels the measured KV pool cannot hold for a workload
# (concurrency_sweep --kv-guard: c > 0.9 * pool / (mean prompt + mean output), from the engine's own pool).
# Off by default.
GUARD_ARG=()
if [ -n "${KV_GUARD:-}" ]; then GUARD_ARG=(--kv-guard); echo "=== KV guard on"; fi
# SMOKE=1: a three-level shakeout of the whole path (engine, sampler, outputs), not a measurement;
# its run directory is tagged "smoke", which every reader of results/ skips.
SMOKE_ARG=()
if [ -n "${SMOKE:-}" ]; then
  SMOKE_ARG=(--smoke --out-tag "smoke_${MODEL}__$(printf '%s' "$WORKLOADS" | tr ',' '-')"); echo "=== smoke run"
fi
echo "=== concurrency_sweep  model=$MODEL  workloads=$WORKLOADS"
$PY concurrency_sweep.py \
    --model "$MODEL" --workload "$WORKLOADS" \
    "${CONC_ARG[@]}" "${GUARD_ARG[@]}" "${SMOKE_ARG[@]}" \
    --warmup-s "${WARMUP_S:-20}" --measure-s "${MEASURE_S:-60}" --idle-s 20 \
    --slo-tbt-ms "${SLO_TBT_MS:-50}" --pool-size 4096

echo "=== done $MODEL"
