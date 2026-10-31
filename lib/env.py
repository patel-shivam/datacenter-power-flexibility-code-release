"""Environment bootstrap. **Import this first, before anything imports vllm or torch.**

Every setting here has to be in place at vllm import time, not at engine construction
time, so the contract is that each experiment script's first project import is

    from lib import env  # noqa: F401  (side effects at import)

or equivalently ``import lib.env``. Importing it twice is harmless.

Why each knob is set
--------------------
``HF_HOME``/``HF_HUB_CACHE``  Optionally point at a shared Hugging Face cache (set
    ``DCE_HF_CACHE``) rather than ``~/.cache``; the models here are 2-131 GB each.

``VLLM_ENABLE_V1_MULTIPROCESSING=0``  Runs the V1 engine core **in-process** instead of
    forking a separate EngineCore process. This is a deliberate accuracy/throughput
    trade and it caveats every capacity number this suite produces:

      * With it ON (vLLM's default), the engine core runs in its own process and
        overlaps with the client, and measured single-replica capacity is ~1.24x
        higher.
      * With it OFF, the scheduler lives in *our* process, so we can read
        ``scheduler.waiting`` directly every step. That is the only way to measure
        the true queue-stability boundary: ``get_num_unfinished_requests()`` counts
        running+waiting, which grows with offered load by Little's Law even when the
        queue is perfectly stable, and reading it as "queue depth" halves every
        capacity number.

    Accuracy of the queue signal wins. The 1.24x is a known, quantified offset --
    not an unknown -- and it applies uniformly to every model here.

``CUDA_HOME``/``PATH``  flashinfer JIT-compiles sampling kernels on first use and needs
    ``nvcc`` plus the env's own ``bin/`` (ninja) on PATH at runtime.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# lib/env.py -> the repository's root (lib/, analyze/, data/, results/ live beside each other)
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EXPERIMENTS_DIR
RESULTS_DIR = EXPERIMENTS_DIR / "results"
TRACES_DIR = EXPERIMENTS_DIR / "data" / "traces"          # the public length traces
DERIVED_TRACES_DIR = EXPERIMENTS_DIR / "data" / "traces"  # the traces prepare_*.py derive

# A shared Hugging Face cache (weights are 2-131 GB per model). Unset: the default ~/.cache.
HF_CACHE = os.environ.get("DCE_HF_CACHE")
# The CUDA toolkit flashinfer JIT-compiles against. Set CUDA_HOME if it is not /usr/local/cuda.
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")

# Derive bin/ from the running interpreter rather than from PROJECT_ROOT: sys.executable
# always points at the environment actually running, wherever this tree lives.
ENV_BIN = Path(sys.executable).resolve().parent


def bootstrap() -> None:
    # CUDA and NVML order the same cards differently. CUDA defaults to FASTEST_FIRST;
    # NVML always enumerates by PCI bus id and ignores CUDA_VISIBLE_DEVICES entirely.
    # hardware.nvml_index() bridges the two by assuming CUDA_VISIBLE_DEVICES=k means
    # NVML index k, which is only true under PCI ordering -- so pin it. lib/fleet.py
    # already sets this inside each worker; doing it here covers the single-GPU path
    # (concurrency_sweep/capacity_bisection) too, which on a node without cgroup isolation is exactly where a
    # mismatch would sample a neighbour's card and look like a busier replica.
    # No-op on a SLURM cluster whose cgroup hands the job one GPU at index 0.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if HF_CACHE:
        os.environ.setdefault("HF_HOME", HF_CACHE)
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(HF_CACHE, "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_CACHE, "transformers"))
    # See module docstring: in-process engine core, so scheduler.waiting is readable.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if os.path.isdir(CUDA_HOME):
        os.environ.setdefault("CUDA_HOME", CUDA_HOME)
        os.environ.setdefault("CUDA_PATH", CUDA_HOME)
    # Quieter logs; the per-step INFO lines are ~5 MB/hour/replica in slurm output.
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    path = os.environ.get("PATH", "")
    cuda_bin = os.path.join(os.environ.get("CUDA_HOME", CUDA_HOME), "bin")
    # Only prepend a CUDA bin/ that exists: on hosts where the toolkit lives
    # elsewhere (module systems, other clusters) a stale absolute path here
    # shadows nothing useful and hides a real PATH problem behind a silent miss.
    for prepend in (cuda_bin, str(ENV_BIN)):
        if os.path.isdir(prepend) and prepend not in path.split(os.pathsep):
            path = f"{prepend}{os.pathsep}{path}"
    os.environ["PATH"] = path

    # Only the repository's root goes on sys.path, never lib/. Putting lib/ there makes
    # every module in it shadow a top-level package of the same name: a lib module named
    # `requests` would be imported by huggingface_hub in place of the real package, which
    # then fails with "'requests' is not a package". Importing as `from lib import x`
    # keeps the names namespaced and makes that class of collision impossible.
    if str(EXPERIMENTS_DIR) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS_DIR))


bootstrap()  # at import time, by design


def run_dir(experiment: str, tag: str, timestamp: str | None = None) -> Path:
    """``results/[<hw>/]<experiment>/<timestamp>__<tag>/``, created.

    One directory per run, never reused, so a re-run can never half-overwrite an
    earlier one and leave a directory that is a mix of two configurations.

    ``<hw>`` is inserted for any accelerator other than the baseline H100 (see
    ``lib/hardware.py``). Results from two different cards are not comparable and
    must never be globbed together by accident; putting the hardware in the path
    makes that structural rather than a convention someone has to remember. The
    baseline keeps a flat layout so paths written against ``results/`` still resolve.
    """
    import time

    from lib import hardware

    ts = timestamp or time.strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in tag)
    base = results_root()
    d = base / experiment / f"{ts}__{safe}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_root() -> Path:
    """Where THIS accelerator's results live: ``results/`` or ``results/<hw>/``.

    ``run_dir`` above namespaces every WRITE. Reads have to use the same root or the
    two diverge: a capacity lookup rooted at the flat tree (``Path(__file__).parent /
    "results"``) on another card finds the H100's capacity instead of the one ``capacity_bisection``
    measured on that card -- on the RTX 8000, a number measured on a card with roughly
    twice the KV pool -- and no missing-capacity check fires, because a match WAS found.

    On the baseline H100 this returns ``RESULTS_DIR`` itself, so paths, globs and
    analysis scripts written against the flat layout resolve as they are.
    """
    from lib import hardware

    return RESULTS_DIR / (hardware.results_namespace() or "")
