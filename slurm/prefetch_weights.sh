#!/bin/bash
# Read one model's weight shards into this node's page cache, many files at once, before
# vLLM loads them.
#
#   bash slurm/prefetch_weights.sh <model-slug> [parallel readers, default 16]
#
# Why: vLLM reads a checkpoint shard by shard, and a cold serial read from Lustre is slow --
# Mistral-24B's 44 GiB took 768 s to load (~61 MB/s), which for the 131 GiB of
# Llama-3.1-70B is ~40 min of four idle H100s per job. Thirty shards read in parallel spread
# over many storage targets; the page cache (host RAM far exceeds a checkpoint) then serves
# every rank's reads. Opt-in from the launchers with PREFETCH=1. Run from the repo root.
set -uo pipefail
M=${1:?usage: prefetch_weights.sh <model-slug> [parallel]}
NPAR=${2:-16}
PY=${PY:-python}   # the interpreter of an environment with requirements-gpu.txt installed
SNAP=$($PY -c "
import sys
sys.path.insert(0, '.')
from lib import env  # noqa: F401  (HF_HOME)
from lib import models
from huggingface_hub import snapshot_download
print(snapshot_download(models.get(sys.argv[1]).model_id, local_files_only=True))
" "$M") || { echo "prefetch: cannot resolve the cached snapshot of $M"; exit 1; }
N=$(find -L "$SNAP" -name '*.safetensors' | wc -l)
[ "$N" -gt 0 ] || { echo "prefetch: no safetensors under $SNAP"; exit 1; }
T0=$(date +%s)
find -L "$SNAP" -name '*.safetensors' -print0 \
  | xargs -0 -P "$NPAR" -n 1 sh -c 'dd if="$1" of=/dev/null bs=16M status=none' _
RC=$?
DT=$(( $(date +%s) - T0 ))
GIB=$(du -scL "$SNAP"/*.safetensors 2>/dev/null | tail -1 | awk '{printf "%.1f", $1/1048576}')
echo "=== prefetch $M: $N shards, $GIB GiB in ${DT} s ($NPAR readers, rc=$RC)"
exit $RC
