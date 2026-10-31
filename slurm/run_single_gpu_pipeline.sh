#!/bin/bash
# The single-GPU experiment set for ONE model on ONE GPU of a workstation (no SLURM), run back
# to back on that card: closed-loop sweeps (concurrency_sweep), capacities at the model's fleet-role cap
# (capacity_bisection), Poisson phases with natural drains (poisson_load) and KV-ceiling sweeps (kv_ceiling_sweep = concurrency_sweep past
# the memory limit) on every task. This is how the RTX 8000 and RTX 5090 cells of the appendix
# were measured (one physical card per model: cards of one type differ by several percent in
# throughput and tens of watts, so a calibration, a capacity and a Poisson phase belong together).
#
#   bash slurm/run_single_gpu_pipeline.sh <model-slug> <gpu-index> [cap]
#   e.g.  nohup bash slurm/run_single_gpu_pipeline.sh qwen3-8b 0 > logs/sg_qwen3-8b.log 2>&1 &
#
# Results land in results/<hardware-slug>/ (lib/hardware.py names the card from NVML), never
# beside another card's. The windows below are the H100 defaults of the SLURM launchers; a
# slower card needs longer ones -- a level or probe must outlast several request lifetimes
# (docs/EXPERIMENTS.md): the RTX 8000 runs used 240/600 s sweeps on ShareGPT-class tasks and
# 120/300 s on 1024-token tasks. Set WARMUP_SHORT/MEASURE_SHORT/WARMUP_LONG/MEASURE_LONG and the
# PROBE_*/WARMUP_* variables accordingly. Every step's exit code is appended to $LOG/sg_<model>.status.
set -uo pipefail
MODEL=${1:?model slug, e.g. qwen3-8b}
GPU=${2:?gpu index}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_DIR"
PY=${PY:-python}                     # an environment with requirements-gpu.txt installed
LOG=${LOG:-$REPO_DIR/logs}
mkdir -p "$LOG"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID     # NVML indexes by PCI order (lib/hardware.nvml_index)
# vLLM 0.11's defaults break on some cards: the FlashInfer sampler needs sm_75+ (and was off on the
# Blackwell RTX 5090 too); FlexAttention on sm_70 has no CUDA graphs (use Triton attention).
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$NAME" in
  *V100*)      export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-TRITON_ATTN} VLLM_USE_FLASHINFER_SAMPLER=0 ;;
  *5090*|*RTX*) export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0} ;;
esac
echo "=== $(date -Is) $MODEL on GPU $GPU ($NAME)"

# the fleet-role caps: 128 for the small models, 64 for the 7B-24B class (the capacities the fleets use)
declare -A CAP=( [llama32-1b]=128 [qwen3-1.7b]=128 [llama32-3b]=128 [qwen3-4b]=128
                 [mistral-7b]=64 [llama31-8b]=64 [qwen3-8b]=64 [qwen3-14b]=64 [mistral-24b]=64 )
C=${3:-${CAP[$MODEL]:?unknown model $MODEL; pass the cap as the third argument}}
WARMUP_SHORT=${WARMUP_SHORT:-20}; MEASURE_SHORT=${MEASURE_SHORT:-60}      # ShareGPT-class tasks
WARMUP_LONG=${WARMUP_LONG:-30};   MEASURE_LONG=${MEASURE_LONG:-90}        # 1024-token tasks
STATUS="$LOG/sg_${MODEL}.status"
: > "$STATUS"

step() {  # name, then the command; a failing step is logged and the pipeline moves on
  local name=$1; shift
  echo "=== $(date -Is) START $name" | tee -a "$STATUS"
  "$@" > "$LOG/sg_${MODEL}__${name}.log" 2>&1
  local rc=$?
  echo "=== $(date -Is) END   $name rc=$rc" | tee -a "$STATUS"
}
sweep() {  # workloads(colon-separated) warmup measure
  step "concurrency_sweep_$1" $PY concurrency_sweep.py --model "$MODEL" --workload "$(echo "$1" | tr ':' ',')" \
       --kv-guard --max-num-batched-tokens 2048 --warmup-s "$2" --measure-s "$3" --idle-s 20 --slo-tbt-ms 50 --pool-size 4096
}
cap() {    # workload probe warmup  (open-loop capacity at the role cap; judged on the waiting list, no floor)
  step "capacity_bisection_$1" $PY capacity_bisection.py --model "$MODEL" --workload "$1" --arrival poisson --max-num-seqs "$C" \
       --max-num-batched-tokens 2048 --iters 6 --probe-s "$2" --warmup-s "$3" --pool-size 4096 \
       --out-tag "paircap_${MODEL}__$1__cap$C"
}
poisson() {  # workloads(colon-separated): phases at 0.15-0.75 of the MEASURED capacity, three drains each
  step "poisson_load_$(echo "$1" | tr ':' '-')" $PY poisson_load.py --model "$MODEL" --workload "$(echo "$1" | tr ':' ',')" \
       --max-num-seqs "$C" --max-num-batched-tokens 2048 --warmup-s 30 --measure-s 120 --max-measure-s 480 --n-drains 3 --pool-size 4096
}
kv() {     # workload: levels at 0.4 .. 2x round(K / l_bar) from this card's MEASURED KV pool (analyze/kv_ceiling_levels.py)
  local lv
  lv=$($PY analyze/kv_ceiling_levels.py --model "$MODEL" --workload "$1") || { echo "no measured KV pool for $MODEL on $1 yet" | tee -a "$STATUS"; return; }
  read -r conc mx <<< "$lv"
  step "kv_ceiling_sweep_$1" $PY concurrency_sweep.py --model "$MODEL" --workload "$1" --concurrency "$conc" \
       --max-num-seqs "$mx" --max-num-batched-tokens 2048 --warmup-s 90 --measure-s 240 --idle-s 20 --slo-tbt-ms 50 --pool-size 4096 \
       --out-tag "kvceil_${MODEL}__$1"
}

# 1. closed-loop sweeps: P(b), the step-time line, the KV pool
sweep sharegpt:humaneval:fixed_128_128:fixed_256_256:fixed_512_512 "$WARMUP_SHORT" "$MEASURE_SHORT"
sweep fixed_1024_1024:fixed_2048_2048:arxiv_summarization "$WARMUP_LONG" "$MEASURE_LONG"
# 2. capacities at the role cap (probe / warm-up lengths rise with the request lifetime)
cap sharegpt "${PROBE_SHORT:-90}" "${WARMUP_PROBE_SHORT:-25}"
cap humaneval "${PROBE_SHORT:-90}" "${WARMUP_PROBE_SHORT:-25}"
cap fixed_128_128 "${PROBE_SHORT:-90}" "${WARMUP_PROBE_SHORT:-25}"
cap fixed_256_256 "${PROBE_MID:-120}" "${WARMUP_PROBE_MID:-40}"
cap fixed_512_512 "${PROBE_MID:-180}" "${WARMUP_PROBE_MID:-60}"
cap fixed_1024_1024 "${PROBE_LONG:-300}" "${WARMUP_PROBE_LONG:-90}"
cap fixed_2048_2048 "${PROBE_XLONG:-420}" "${WARMUP_PROBE_XLONG:-120}"
cap arxiv_summarization "${PROBE_ARXIV:-240}" "${WARMUP_PROBE_ARXIV:-60}"
# 3. Poisson phases + drains (placed from the measured capacities above)
poisson sharegpt:humaneval:fixed_128_128:fixed_1024_1024:arxiv_summarization
poisson fixed_256_256:fixed_512_512:fixed_2048_2048
# 4. KV ceilings on the long-context tasks
kv arxiv_summarization
kv fixed_1024_1024
kv fixed_2048_2048
echo "=== $(date -Is) PIPELINE DONE $MODEL on GPU $GPU" | tee -a "$STATUS"
