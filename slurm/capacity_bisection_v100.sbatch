#!/bin/bash
# Open-loop capacity (capacity_bisection.py) on a V100-32, one model@cap per array task, for
# the single-GPU model figures' V100 column (capacity parity, occupancy probes, power from
# rate). Triton attention + native sampler as every V100 run.
#
#   sbatch --array=0-3 --export=ALL,REPO_DIR=<checkout>,WORKLOAD=sharegpt,SPEC=llama31-8b@64:llama32-1b@128:qwen3-8b@64:qwen3-4b@128 \
#          slurm/capacity_bisection_v100.sbatch
#SBATCH -J capacity_bisection_v100
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
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
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOGGING_LEVEL=INFO
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}

if ! grep -q "no absolute floor" capacity_bisection.py; then
  echo "REFUSING: $REPO_DIR/capacity_bisection.py lacks the no-absolute-floor stability rule."
  exit 1
fi

SPEC=${SPEC:-llama31-8b@64:llama32-1b@128:qwen3-8b@64:qwen3-4b@128}
WORKLOAD=${WORKLOAD:-sharegpt}
IFS=':' read -ra ITEMS <<< "$SPEC"
I=${SLURM_ARRAY_TASK_ID:-0}
if [ "$I" -ge "${#ITEMS[@]}" ]; then
  echo "REFUSING: array index $I but SPEC has only ${#ITEMS[@]} entries."; exit 1
fi
MODEL=${ITEMS[$I]%@*}
CAP=${ITEMS[$I]#*@}

# The V100 is ~3x slower than the H100, so a probe must be longer to outlast a request:
# 1024-token outputs live ~1-2 min at the cap, so those probes are 300 s after 90 s warm-up.
case "$WORKLOAD" in
  fixed_1024_1024|fixed_6144_1024|fixed_2048_2048)
    PROBE_S=${PROBE_S:-300}; WARMUP_S=${WARMUP_S:-90} ;;
  arxiv_summarization)
    PROBE_S=${PROBE_S:-240}; WARMUP_S=${WARMUP_S:-60} ;;
  *)
    PROBE_S=${PROBE_S:-150}; WARMUP_S=${WARMUP_S:-40} ;;
esac

echo "=== capacity bisection (V100): $MODEL @ max_num_seqs=$CAP on $WORKLOAD  probe $PROBE_S s  $(date -Is)"
$PY capacity_bisection.py \
    --model "$MODEL" --workload "$WORKLOAD" --arrival poisson \
    --max-num-seqs "$CAP" --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
    --iters "${ITERS:-6}" --probe-s "$PROBE_S" --warmup-s "$WARMUP_S" \
    ${LO:+--lo "$LO"} --pool-size 4096 \
    --out-tag "paircap_${MODEL}__${WORKLOAD}__cap${CAP}"
echo "=== capacity bisection (V100) $MODEL cap $CAP exit=$?  $(date -Is)"
