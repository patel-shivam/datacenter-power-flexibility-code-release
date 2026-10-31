"""Which accelerator this is running on, and what that implies.

Every number this suite produces -- power, throughput, capacity, the fitted
``P(b)`` -- is a property of *a model on a particular GPU*, not of the model
alone. Pooling runs from different accelerators silently averages two different
physical systems, and the result looks like a noisy version of either one rather
than like an error. So hardware is made explicit in three places:

1. **Results are namespaced on disk** by :func:`results_namespace`, so a run on
   a new card cannot land in the same directory tree as the H100 runs.
2. **dtype is resolved per platform** by :func:`resolve_dtype`. Turing (sm_75)
   has no bfloat16, and vLLM refuses to start rather than silently downcasting.
3. **Figures carry the hardware tag** -- see ``analyze/plot_authority_compare.py``,
   which derives it from where each run's results live and refuses to plot runs
   from two different cards on the same axes.

The baseline
------------
``BASELINE_SLUG`` is the card whose results sit directly in ``results/``. Runs on
it keep a flat layout so paths, globs and analysis scripts written against
``results/`` resolve unchanged; anything else is nested one level deeper. The
asymmetry is deliberate -- it keeps the large body of baseline results where the
analysis expects them.

Detection goes through NVML rather than torch so that importing this module
never initializes a CUDA context.
"""

from __future__ import annotations

import functools
import os
import re

BASELINE_SLUG = "h100-80gb"

# Compute capability at which bfloat16 becomes available in hardware (Ampere).
_BF16_MIN_MAJOR = 8


def _slugify(name: str, mem_gib: float) -> str:
    """Marketing name -> stable short slug.

    ``'NVIDIA H100 80GB HBM3'``   -> ``'h100-80gb'``
    ``'Quadro RTX 8000'``         -> ``'rtx-8000-48gb'``
    ``'NVIDIA A100-SXM4-80GB'``   -> ``'a100-80gb'``

    Vendor and packaging words are dropped; the product line is not. ``rtx`` in
    particular has to stay -- without it a Quadro RTX 8000 slugs to bare
    ``8000``, which names nothing.
    """
    n = name.lower()
    n = re.sub(r"\b(nvidia|geforce|quadro|tesla|sxm\d*|hbm\d*|pcie)\b", " ", n)
    n = re.sub(r"\b\d+\s*gb\b", " ", n)          # drop an embedded capacity
    n = re.sub(r"[^a-z0-9]+", "-", n).strip("-")
    return f"{n}-{int(round(mem_gib))}gb" if n else f"gpu-{int(round(mem_gib))}gb"


INSTANCE_ENV = "DCE_INSTANCE_TAG"


def set_instance(tensor_parallel_size: int) -> str:
    """Declare how many GPUs ONE model instance in this process spans; returns the tag.

    A model sharded over N cards (``ModelSpec.tensor_parallel_size``) is a different
    serving platform from one card: its power is N cards' worth, its step time is set by
    the interconnect as well as by HBM. Its results therefore get their own namespace --
    ``h100-80gb-tp4`` -- by the same mechanism that keeps an RTX 8000 run out of the H100
    tree, rather than by a naming convention someone has to remember. The tag goes into
    the environment so worker processes and every later ``slug()`` call agree.

    N <= 1 clears the tag: every one-card run keeps its card's plain slug.
    """
    n = int(tensor_parallel_size or 1)
    if n <= 1:
        os.environ.pop(INSTANCE_ENV, None)
        return ""
    tag = f"tp{n}"
    os.environ[INSTANCE_ENV] = tag
    return tag


def instance_tag() -> str:
    t = os.environ.get(INSTANCE_ENV, "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-")


def current() -> dict:
    """``{name, slug, cc_major, cc_minor, mem_gib, index}`` for the sampled GPU.

    Returns ``{}`` when NVML is unavailable (CPU-only analysis hosts), which
    callers treat as "assume the baseline and change nothing".

    Under a multi-GPU instance (``set_instance``) the slug carries the instance tag
    (``h100-80gb-tp4``) and ``card_slug`` keeps the card's own; without one this is the
    cached record itself, unchanged.
    """
    d = _device()
    tag = instance_tag()
    if not d or not tag:
        return d
    m = re.fullmatch(r"tp(\d+)", tag)
    return {**d, "slug": f"{d['slug']}-{tag}", "card_slug": d["slug"], "instance": tag,
            "instance_gpus": int(m.group(1)) if m else None}


@functools.lru_cache(maxsize=1)
def _device() -> dict:
    """The NVML facts behind ``current()``, read once per process."""
    try:
        import pynvml
    except Exception:
        return {}
    try:
        pynvml.nvmlInit()
        idx = nvml_index()
        h = pynvml.nvmlDeviceGetHandleByIndex(idx)
        name = pynvml.nvmlDeviceGetName(h)
        name = name.decode() if isinstance(name, bytes) else name
        mem = pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024 ** 3)
        try:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
        except Exception:
            major, minor = 0, 0
        return {"name": name, "slug": _slugify(name, mem), "cc_major": major,
                "cc_minor": minor, "mem_gib": round(mem, 1), "index": idx}
    except Exception:
        return {}


def nvml_index() -> int:
    """Physical NVML index of the GPU this process actually uses.

    NVML enumerates every card on the node and **ignores CUDA_VISIBLE_DEVICES**,
    so a process pinned to GPU 3 that samples NVML index 0 reads a different
    card -- silently, and with plausible-looking numbers. So the mapping is derived
    here once rather than passed around by hand: when CUDA_VISIBLE_DEVICES names
    a single device, that *is* the physical index to sample.

    ``DCE_NVML_INDEX`` overrides, for the case where the two genuinely differ.
    """
    override = os.environ.get("DCE_NVML_INDEX")
    if override is not None and override.strip().isdigit():
        return int(override.strip())
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if vis and "," not in vis and vis.isdigit():
        return int(vis)
    return 0


def nvml_indices(n: int) -> list[int]:
    """Physical NVML indices of the ``n`` GPUs one tensor-parallel instance runs on.

    vLLM places rank k on CUDA device k of the visible set, so the instance is the first
    ``n`` entries of CUDA_VISIBLE_DEVICES -- read here as NVML indices, which holds because
    ``env.py`` sets CUDA_DEVICE_ORDER=PCI_BUS_ID (NVML's own order). No list means the
    process sees every card, numbered from 0. ``DCE_NVML_INDICES`` overrides.

    The mapping is an assumption until the GPUs are loaded; ``PowerSampler`` checks the
    bus ids are distinct and ``power_sampler.assert_instance_resident`` checks that every
    sampled card actually holds its share of the weights.
    """
    n = int(n)
    override = os.environ.get("DCE_NVML_INDICES", "").strip()
    vis = override or os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if vis:
        parts = [p.strip() for p in vis.split(",") if p.strip()]
        if not all(p.isdigit() for p in parts):
            raise RuntimeError(f"cannot map CUDA_VISIBLE_DEVICES={vis!r} (UUIDs / MIG) to NVML "
                               f"indices; set DCE_NVML_INDICES")
        idx = [int(p) for p in parts]
    else:
        idx = list(range(n))
    if len(idx) < n:
        raise RuntimeError(f"a {n}-GPU instance needs {n} visible GPUs; got {idx}")
    return idx[:n]


def slug() -> str:
    # Without NVML (a CPU analysis host) the card is assumed to be the baseline, and a
    # declared multi-GPU instance still gets its tag -- else its runs resolve to results/.
    tag = instance_tag()
    return current().get("slug") or (f"{BASELINE_SLUG}-{tag}" if tag else BASELINE_SLUG)


def is_baseline() -> bool:
    return slug() == BASELINE_SLUG


def results_namespace() -> str:
    """Sub-directory under ``results/`` for this hardware (``''`` on baseline)."""
    return "" if is_baseline() else slug()


def supports_bf16() -> bool:
    cc = current().get("cc_major")
    return True if cc is None else cc >= _BF16_MIN_MAJOR


def resolve_dtype(requested: str) -> str:
    """The dtype to actually serve with, given what this card supports.

    bfloat16 needs compute capability >= 8.0. On Turing, vLLM raises
    ``Bfloat16 is only supported on GPUs with compute capability of at least
    8.0`` rather than downcasting, so the substitution has to happen here.
    fp16 is the same two bytes per parameter, so no memory or KV arithmetic
    changes -- only the exponent/mantissa split does.
    """
    if requested == "bfloat16" and not supports_bf16():
        return "float16"
    return requested


def describe() -> str:
    d = current()
    if not d:
        return "no NVML (CPU host)"
    return (f"{d['name']} ({d['mem_gib']:.0f} GiB, sm_{d['cc_major']}{d['cc_minor']}, "
            f"nvml index {d['index']}) -> slug {d['slug']}")
