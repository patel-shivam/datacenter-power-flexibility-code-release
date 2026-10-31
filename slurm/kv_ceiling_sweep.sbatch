#!/bin/bash
# kv_ceiling_sweep -- the KV-ceiling sweep: concurrency_sweep's closed-loop sweep driven PAST the memory limit on a
# long-context workload, with max_num_seqs raised above every level so the request cap
# cannot bind first. The observed plateau of the mean resident count (with KV usage at
# 100% and a waiting list) is the measured ceiling; the analysis compares it with
# floor(K / l_bar) (decode-weighted context) and the peak-aware K / (P + D).
#
#   sbatch --export=ALL,REPO_DIR=<checkout>,MODEL=llama31-8b,WORKLOAD=fixed_1024_1024,CONCURRENCY=119:178:238:267:297:327:371:446:594,MAX_NUM_SEQS=640 \
#          slurm/kv_ceiling_sweep.sbatch
#
# COLONS in CONCURRENCY (SLURM's --export splits on commas). Windows are long because a
# 1024-token decode at b~240 lives ~36 s on the H100 and the window must hold >= 2 of them.
#SBATCH -J kv_ceiling_sweep
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH -t 04:00:00
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
# On a V100 (submit with --gres=gpu:v100:1) vLLM needs the Triton attention backend
# (CUDA graphs; FlexAttention has none on sm_70) and the native sampler (no FlashInfer
# JIT path) -- the settings of concurrency_sweep_v100.sbatch.
if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -q V100; then
  export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-TRITON_ATTN} VLLM_USE_FLASHINFER_SAMPLER=0
  echo "=== V100: TRITON_ATTN, native sampler"
fi

MODEL=${MODEL:?MODEL}
WORKLOAD=${WORKLOAD:?WORKLOAD}
# CONCURRENCY=auto: place the grid from the KV pool this model has MEASURED on this hardware
# (round(K / l_bar) x 0.4 .. 2.0, max_num_seqs 2.2x -- the rule behind every explicit grid),
# for a model whose pool is only known once a job has built it.
if [ "${CONCURRENCY:-}" = "auto" ]; then
  LV=$($PY analyze/kv_ceiling_levels.py --model "$MODEL" --workload "$WORKLOAD") \
    || { echo "REFUSING: no measured KV pool for $MODEL on this hardware ($LV)"; exit 1; }
  read -r CONCURRENCY MAX_NUM_SEQS <<< "$LV"
  echo "=== kv_ceiling_sweep levels from the measured KV pool: $CONCURRENCY (max_num_seqs $MAX_NUM_SEQS)"
fi
CONC=$(printf '%s' "${CONCURRENCY:?CONCURRENCY}" | tr ':' ',')
MAX_NUM_SEQS=${MAX_NUM_SEQS:?MAX_NUM_SEQS}
# Opt-in: read a big checkpoint into the page cache in parallel first (prefetch_weights.sh).
if [ -n "${PREFETCH:-}" ]; then bash slurm/prefetch_weights.sh "$MODEL" || echo "prefetch failed; loading cold"; fi
if ! grep -q "max-num-seqs" concurrency_sweep.py; then
  echo "REFUSING: $REPO_DIR/concurrency_sweep.py has no --max-num-seqs."
  exit 1
fi

echo "=== kv_ceiling_sweep kv ceiling  model=$MODEL workload=$WORKLOAD levels=$CONC max_num_seqs=$MAX_NUM_SEQS  $(date -Is)"
$PY concurrency_sweep.py \
    --model "$MODEL" --workload "$WORKLOAD" --concurrency "$CONC" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
    --warmup-s "${WARMUP_S:-60}" --measure-s "${MEASURE_S:-150}" --idle-s 20 \
    --slo-tbt-ms 50 --pool-size 4096 \
    --out-tag "kvceil_${MODEL}__${WORKLOAD}"
echo "=== kv_ceiling_sweep $MODEL $WORKLOAD exit=$?  $(date -Is)"
