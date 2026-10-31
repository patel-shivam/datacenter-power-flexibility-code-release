#!/bin/bash
# fleet_sweep_8gpu -- knob authority over the whole load range on EIGHT H100s (one full node).
#
#   NL=5 NS=3 REPO_DIR=<checkout> sbatch --array=0-1 --export=ALL \
#      slurm/fleet_sweep_8gpu.sbatch
#
# WHY. fleet_sweep_4gpu's open-loop protocol, unchanged, on eight GPUs in the RTX 8000's compositions
# (5L+3S, 4L+4S, 6L+2S), so the 4- and 8-GPU H100 fleets and the 8-GPU RTX fleet compare like
# with like.
#
# DESIGN (fleet_sweep_4gpu's): loads u = 0.1 .. 0.9 of the MEASURED fleet capacity NL*C_L + NS*C_S
# (capacity_lookup, the same three-way check); 11 values of f, 0 .. 1; 150 s probes, drained
# before each; pack, all traffic flexible; caps 64 / 128. Two array tasks, as fleet_sweep_4gpu:
#   0  u 0.1-0.5      1  u 0.6-0.9
# At high load the capacity window is ~0.1-0.25 wide in f: dry-run first and add fractions
# inside it with FGRID (+ FGRID_TAG) so every load has at least two -- see fleet_sweep_4gpu's header.
# Variables go through the environment with --export=ALL (FGRID holds commas).
#
# AT THIS SCALE, one driver process routes for eight replicas, and on ShareGPT the top load is
# ~250-330 req/s, 2-3x the four-GPU fleet's 121 req/s. The 1B is CPU-bound in this harness, so
# check the first run's high-load probes for queues that grow inside capacity before trusting
# the rest.
#SBATCH -J fleet_sweep_8gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:8
#SBATCH -t 05:00:00
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
NL=${NL:-5}; NS=${NS:-3}
WL=${WL:-sharegpt}
if [ $((NL + NS)) -ne 8 ]; then
  echo "REFUSING: NL + NS = $((NL + NS)); this launcher is for one eight-GPU node."; exit 1
fi
#       0                    1
USETS=( 0.1,0.2,0.3,0.4,0.5  0.6,0.7,0.8,0.9 )
I=${SLURM_ARRAY_TASK_ID:-0}
if [ "$I" -ge "${#USETS[@]}" ]; then
  echo "REFUSING: array index $I but only ${#USETS[@]} tasks are defined."; exit 1
fi
US=${USET:-${USETS[$I]}}
if [ -z "${DRY_RUN:-}" ]; then
  NG=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU")
  if [ "$NG" -lt 8 ]; then
    echo "REFUSING: only $NG GPUs visible; this launcher needs a full eight-GPU node."; exit 1
  fi
fi

LOADS=$($PY - "$LG" "$SM" "$CL" "$CS" "$WL" "$US" "$NL" "$NS" <<'PYEOF'
import json
import sys
from pathlib import Path
sys.path.insert(0, ".")
from analyze import capacity_lookup as CL
from lib import feasible
large, small, cl, cs, wl = (sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                            sys.argv[5])
us = [float(x) for x in sys.argv[6].split(",")]
nl, ns = int(sys.argv[7]), int(sys.argv[8])
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
    sys.stderr.write(f"  {m}@{c} = {got['reqs_s']:.3f} req/s ({got['source']})\n")
fleet = nl * caps[large]["reqs_s"] + ns * caps[small]["reqs_s"]
ustar = ns * caps[small]["reqs_s"] / fleet
sys.stderr.write(f"  fleet {nl}L+{ns}S = {fleet:.2f} req/s, u* = {ustar:.3f} -> loads at u={us}\n")
print(",".join(f"{u * fleet:.2f}" for u in us))
PYEOF
) || { echo "REFUSING: no usable measured $WL capacity for $LG@$CL / $SM@$CS."; exit 1; }
U0=${US%%,*}; U1=${US##*,}
TAG="${LG}-${SM}__${WL}__${NL}L${NS}S__c${CL}-${CS}__pack__poisson__x000-000-100__wide_u${U0}-${U1}${FGRID_TAG:+__f${FGRID_TAG}}"
FGRID=${FGRID:-0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
echo "################ fleet_sweep_8gpu task $I: $TAG  loads $LOADS  $(date -Is)"
$PY fleet_routing_sweep.py \
    --n-large "$NL" --n-small "$NS" --large "$LG" --small "$SM" \
    --workload "$WL" --arrival poisson \
    --concurrency-large "$CL" --concurrency-small "$CS" \
    --policy pack --class-split 0.0,0.0,1.0 \
    --loads "$LOADS" \
    --f-grid "$FGRID" \
    --point-s "${POINT_S:-150}" \
    --out-tag "$TAG" \
    --residence-report $R/power_curves_report.json ${DRY_RUN:+--dry-run}
echo "################ fleet_sweep_8gpu task $I exit=$?  $(date -Is)"
