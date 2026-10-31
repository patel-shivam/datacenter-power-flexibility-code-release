#!/bin/bash
# fleet_sweep_4gpu -- knob authority over a WIDE range of offered load, H100, llama pair.
#
#   sbatch --array=0-3 --export=ALL,REPO_DIR=<checkout> slurm/fleet_sweep_4gpu.sbatch
#
# WHY. The band-vs-load figures need both workloads measured with ONE protocol over
# u = 0.1 .. 0.9, so they compare like with like.
#
# DESIGN:
#   loads   u = 0.1 .. 0.9 of the MEASURED fleet capacity 3 cL + cS (capacity_bisection, via
#           capacity_lookup, cross-checked against capacity.json and lib/feasible), never the
#           analytic one.
#   f-grid  11 points, 0 .. 1: a coarse grid misses band ends.
#   probes  150 s: fleet_routing_sweep drains before every probe and averages its last half, so the window
#           opens at 75 s -- several request lifetimes even for arXiv's 2048-token tail.
#   policy  pack, all flexible (x000-000-100), 3L+1S, caps 64 / 128: the canonical
#           configuration.
# Four array tasks, a workload's loads split in two so each fits a short slot:
#   0 arXiv u 0.1-0.5   1 arXiv u 0.6-0.9   2 ShareGPT u 0.1-0.5   3 ShareGPT u 0.6-0.9
# About 2.5-3 h each: 4-5 loads x 11 f x (150 s + drain) + model load.
#
# OTHER PAIRS through LG / SM, protocol unchanged; the tag carries the pair. The qwen pair on
# ShareGPT:
#   LG=qwen3-8b SM=qwen3-4b REPO_DIR=<checkout> sbatch --array=2 --export=ALL \
#     slurm/fleet_sweep_4gpu.sbatch
#   LG=qwen3-8b SM=qwen3-4b FGRID_TAG=win FGRID=0,0.1,0.2,0.3,0.34,0.4,0.43,0.5,0.6,0.7,0.8,0.9,1.0 \
#     REPO_DIR=<checkout> sbatch --array=3 --export=ALL slurm/fleet_sweep_4gpu.sbatch
# At u = 0.8 and 0.9 the qwen pair's measured-capacity window is f 0.24-0.49 and 0.33-0.44,
# so 2 and 1 points of the 11-point grid fall inside it and the strict band at u = 0.9 could
# only read "none". FGRID adds 0.34 and 0.43 (inside the window at every load of task 3) to
# the same 11; the band on the 11 alone stays comparable with the llama sweep. Set FGRID in
# the environment with --export=ALL: sbatch's --export=A=x,B=y splits values on commas.
#
# OTHER WORKLOADS: WL overrides the task's workload, e.g. the 1:1 ladder (fixed_128_128,
# fixed_256_256, fixed_512_512), whose capacities at caps 64/128 exist; the tag carries the
# workload. Tasks 2 and 3 give u = 0.1-0.5 and 0.6-0.9:
#   WL=fixed_256_256 REPO_DIR=<checkout> sbatch --array=2 --export=ALL \
#     slurm/fleet_sweep_4gpu.sbatch
#
# IN-WINDOW PROBES AT LOADS ALREADY SWEPT. At the llama sweep's high loads the 0.1-step grid
# puts under two usable fractions inside the capacity window (ShareGPT u 0.9: 1; arXiv u 0.8:
# 2, one of which started with a backlog; arXiv u 0.9: 1), so those loads have no band. USET
# replaces the task's loads; FGRID then holds fractions inside the window only, since the 11
# standard ones come from the task's standard run:
#   USET=0.9 FGRID_TAG=winonly FGRID=0.47,0.5,0.53,0.56 REPO_DIR=<checkout> \
#     sbatch -t 01:30:00 --array=3 --export=ALL slurm/fleet_sweep_4gpu.sbatch
#SBATCH -J fleet_sweep_4gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:4
#SBATCH -t 04:30:00
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
# The per-replica NVML binding. Without it four replicas sample one card.
if ! grep -q "device_index=cfg.gpu_index" lib/fleet.py; then
  echo "REFUSING: lib/fleet.py lacks the per-replica NVML binding."; exit 1
fi
if ! grep -q "def lookup" analyze/capacity_lookup.py; then
  echo "REFUSING: no analyze/capacity_lookup.py to read capacities with."; exit 1
fi
mkdir -p slurm_logs
PY=${PY:-python}   # the interpreter of an environment with requirements-gpu.txt installed
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FLEET_POWER_WINDOW_S=${FLEET_POWER_WINDOW_S:-0.5}
export FLEET_ADMISSION_HEADROOM_SEQS=${FLEET_ADMISSION_HEADROOM_SEQS:-16}
export FLEET_READY_TIMEOUT_S=${FLEET_READY_TIMEOUT_S:-2700}
R=results
LG=${LG:-llama31-8b}; SM=${SM:-llama32-1b}; CL=64; CS=128

#       0                    1                    2        3
WLS=(   arxiv_summarization  arxiv_summarization  sharegpt sharegpt )
USETS=( 0.1,0.2,0.3,0.4,0.5  0.6,0.7,0.8,0.9      0.1,0.2,0.3,0.4,0.5 0.6,0.7,0.8,0.9 )
I=${SLURM_ARRAY_TASK_ID:-0}
if [ "$I" -ge "${#WLS[@]}" ]; then
  echo "REFUSING: array index $I but only ${#WLS[@]} tasks are defined."; exit 1
fi
WL=${WL:-${WLS[$I]}}; US=${USET:-${USETS[$I]}}

LOADS=$($PY - "$LG" "$SM" "$CL" "$CS" "$WL" "$US" <<'PYEOF'
import json
import sys
from pathlib import Path
sys.path.insert(0, ".")
from analyze import capacity_lookup as CL
from lib import feasible
large, small, cl, cs, wl = (sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                            sys.argv[5])
us = [float(x) for x in sys.argv[6].split(",")]
R = Path("results")
caps = {}
for m, c in ((large, cl), (small, cs)):
    got = CL.lookup(R, m, wl, c)
    if got is None:
        sys.stderr.write(f"MISSING measured capacity: {m}@{c} on {wl}\n"); raise SystemExit(1)
    recorded = json.loads((R / "capacity_bisection" / got["source"] / "capacity.json")
                          .read_text()).get("capacity_reqs_s")
    clamp = feasible.load_measured(R, m, wl, c)
    if got["rule"] != CL.AS_RECORDED and got["reqs_s"] != recorded:
        sys.stderr.write(f"STALE: {m}@{c} on {wl} ({got['source']}) re-judges to "
                         f"{got['reqs_s']} against {recorded} recorded\n"); raise SystemExit(1)
    if clamp is None or abs(clamp - got["reqs_s"]) > 1e-9:
        sys.stderr.write(f"INCONSISTENT: {m}@{c} on {wl}: loads would use {got['reqs_s']} "
                         f"but fleet_routing_sweep records feasibility with {clamp}\n"); raise SystemExit(1)
    caps[m] = got
    sys.stderr.write(f"  {m}@{c} = {got['reqs_s']:.3f} req/s ({got['source']}; "
                     f"{'re-judged from its probes, unchanged' if got['rule'] != CL.AS_RECORDED else 'as recorded'})\n")
fleet = 3 * caps[large]["reqs_s"] + caps[small]["reqs_s"]
sys.stderr.write(f"  fleet 3L+1S = {fleet:.2f} req/s -> loads at u={us}\n")
print(",".join(f"{u * fleet:.2f}" for u in us))
PYEOF
) || { echo "REFUSING: no usable measured $WL capacity for $LG@$CL / $SM@$CS."; exit 1; }

U0=${US%%,*}; U1=${US##*,}
TAG="${LG}-${SM}__${WL}__3L1S__c${CL}-${CS}__pack__poisson__x000-000-100__wide_u${U0}-${U1}${FGRID_TAG:+__f${FGRID_TAG}}"
FGRID=${FGRID:-0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
echo "################ fleet_sweep_4gpu task $I: $TAG  loads $LOADS  $(date -Is)"
$PY fleet_routing_sweep.py \
    --n-large 3 --n-small 1 --large "$LG" --small "$SM" \
    --workload "$WL" --arrival poisson \
    --concurrency-large "$CL" --concurrency-small "$CS" \
    --policy pack --class-split 0.0,0.0,1.0 \
    --loads "$LOADS" \
    --f-grid "$FGRID" \
    --point-s "${POINT_S:-150}" \
    --out-tag "$TAG" \
    --residence-report $R/power_curves_report.json ${DRY_RUN:+--dry-run}
echo "################ fleet_sweep_4gpu task $I exit=$?  $(date -Is)"
