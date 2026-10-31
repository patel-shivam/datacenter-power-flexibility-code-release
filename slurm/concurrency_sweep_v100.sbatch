#!/bin/bash
# concurrency_sweep on ONE V100-32 (SLURM cluster, GPU partition): the single-GPU P(b) sweep across
# workloads on a third architecture (Volta, sm_70), as an ablation of the concave per-replica
# power curve measured on the H100 (and found nearly flat on the Turing RTX 8000).
#
#   sbatch --array=0-3 slurm/concurrency_sweep_v100.sbatch
#   sbatch --array=0 --export=ALL,SMOKE=1,WORKLOADS=sharegpt -t 0:45:00 \
#          slurm/concurrency_sweep_v100.sbatch        # shakeout first
#
# Differences from concurrency_sweep.sbatch, all deliberate:
#   * --gres=gpu:v100:1. The 16 GB V100s cannot hold an 8B in fp16 (16 GB of weights).
#   * dtype: lib/hardware.resolve_dtype serves float16 on sm_70 (no bfloat16 there), and
#     lib/env puts the results under results/<hardware slug>/ so nothing lands next to
#     the H100 numbers.
#   * --kv-guard: a 32 GB card holds far fewer resident sequences than an 80 GB one, so
#     the top of each model's grid is a KV limit, not a P(b) level; the guard drops the
#     levels above 0.9 * pool / (mean prefill + mean decode) per workload and records it.
#   * longer probes (MEASURE_S 90, WARMUP_S 30): a V100 decodes about three times more
#     slowly than an H100, and a level must outlast a few residences at the cap.
#   * --max-num-batched-tokens 2048 pinned and recorded (long-prefill workloads depend
#     on it; it is vLLM 0.11's default, so H100 comparability is unchanged).
# Lists are COLON-separated for the same --export reason as concurrency_sweep.sbatch.
#SBATCH -J concurrency_sweep_v100
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH -t 4:30:00
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
# vLLM 0.11 samples top-k/top-p through FlashInfer's JIT kernels, which refuse sm_70
# ("FlashInfer requires GPUs with sm75 or higher", raised after the engine has loaded). The
# native PyTorch sampler is a few microseconds per step on a forward pass of tens of
# milliseconds, so power is unaffected; the H100 runs used the FlashInfer sampler. INFO
# logging records which attention backend vLLM picks on Volta.
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOGGING_LEVEL=INFO
# Left to itself vLLM 0.11 picks FlexAttention on sm_70 (FlashAttention needs sm_80,
# FlashInfer sm_75): no CUDA graphs, and decoding is so slow that one sequence draws 58 W
# against 53 W idle -- the GPU sits idle behind host overhead. Triton attention keeps CUDA
# graphs; it is what vLLM used on the Turing RTX 8000.
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}

echo "=== node $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader 2>/dev/null | head -1)"

MODELS=(llama32-1b llama31-8b qwen3-4b qwen3-8b)
MODEL=${MODEL:-${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}}
WORKLOADS=$(printf '%s' "${WORKLOADS:-fixed_128_128:sharegpt:fixed_1024_1024:arxiv_summarization}" | tr ':' ',')

EXTRA=()
if [ -n "${CONCURRENCY:-}" ]; then
  EXTRA+=(--concurrency "$(printf '%s' "$CONCURRENCY" | tr ':' ',')")
fi
if [ -n "${SMOKE:-}" ]; then   # a shakeout, tagged "smoke": every reader of results/ skips it
  EXTRA+=(--smoke --out-tag "smoke_${MODEL}__$(printf '%s' "$WORKLOADS" | tr ',' '-')")
fi

echo "=== concurrency_sweep (V100)  model=$MODEL  workloads=$WORKLOADS  extra=${EXTRA[*]:-none}"
$PY concurrency_sweep.py \
    --model "$MODEL" --workload "$WORKLOADS" --kv-guard \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
    "${EXTRA[@]}" \
    --warmup-s "${WARMUP_S:-30}" --measure-s "${MEASURE_S:-90}" --idle-s 20 \
    --slo-tbt-ms "${SLO_TBT_MS:-50}" --pool-size 4096

echo "=== done $MODEL"
