"""vLLM engine construction, sizing, and the live scheduler read.

One rule governs this module: **every number that describes the engine's state must
come from the scheduler, not from a client-side proxy.** The proxies are all wrong in
ways that look plausible:

  * ``get_num_unfinished_requests()`` counts running **plus** waiting. In-flight
    concurrency grows with offered load by Little's Law even when the queue is
    perfectly stable, so reading this as "queue depth" makes a healthy system look
    saturated and halves any capacity measured from it.
  * The number of requests *submitted minus completed* is the same quantity by
    another route, with the same defect.
  * KV "usage" inferred from token counts ignores block granularity and prefix
    sharing.

``read_state`` therefore reaches into ``scheduler.running`` / ``scheduler.waiting`` /
``kv_cache_manager.usage`` directly. That is only possible because ``env.py`` sets
``VLLM_ENABLE_V1_MULTIPROCESSING=0`` so the engine core lives in this process; the
documented cost of that choice is ~1.24x lower measured capacity, applied uniformly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from lib import env  # noqa: F401  (before vllm; bootstraps env vars)
from lib import hardware, models


@dataclass
class EngineState:
    """One scheduler-truth snapshot. ``num_running`` is *the* concurrency variable:
    it is the number of sequences the model is actually doing forward passes for,
    which is the coordinate the power model is built in."""

    num_running: int
    num_waiting: int
    kv_usage: float          # fraction of the KV block pool in use, 0..1
    ok: bool = True          # False => the scheduler could not be read; see read_state


def build_engine(spec: "models.ModelSpec | str", *, max_num_seqs: int,
                 max_num_batched_tokens: int | None = None,
                 gpu_memory_utilization: float | None = None,
                 max_model_len: int | None = None,
                 enable_chunked_prefill: bool = True,
                 enforce_eager: bool = False):
    """Build an ``LLMEngine`` for one model at one concurrency cap.

    ``max_num_seqs`` is the engine's hard ceiling on the running batch -- the
    "max parallelism cap" this suite sweeps. It is distinct from the router-side
    ``pack_cap`` (see ``router.py``), which decides when to *stop sending* to a replica
    that could still accept more.

    ``max_num_batched_tokens`` caps prefill chunks + decode continuations per step.
    Left at vLLM's default unless overridden: setting it too low starves prefill
    admission under a high arrival rate and throttles throughput below what a
    concurrency-held benchmark would report, which then reads as a capacity ceiling
    that is really a scheduler budget.
    """
    from vllm import EngineArgs, LLMEngine  # type: ignore

    if isinstance(spec, str):
        spec = models.get(spec)
    kwargs = {}
    if max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    tp = int(getattr(spec, "tensor_parallel_size", 1) or 1)
    if tp > 1:
        # ONE instance over tp cards. The model runs in tp worker processes (vLLM's "mp"
        # executor); the scheduler stays in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0),
        # so read_state still reads scheduler truth. Spawn, not fork: the power sampler's
        # thread is already polling NVML when the engine is built, and a fork taken while
        # it holds a lock hands the child a lock nobody will release.
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        kwargs["tensor_parallel_size"] = tp
        kwargs["distributed_executor_backend"] = "mp"
    args = EngineArgs(
        model=spec.model_id,
        # Resolved per accelerator: bfloat16 needs sm_80+, and on older cards
        # vLLM raises rather than downcasting. Same 2 bytes/param either way, so
        # no memory or KV arithmetic changes. No-op on the baseline H100.
        dtype=hardware.resolve_dtype(spec.dtype),
        gpu_memory_utilization=(gpu_memory_utilization
                                if gpu_memory_utilization is not None
                                else spec.gpu_memory_utilization),
        max_model_len=max_model_len if max_model_len is not None else spec.max_model_len,
        max_num_seqs=max_num_seqs,
        enable_chunked_prefill=enable_chunked_prefill,
        enforce_eager=enforce_eager,
        disable_log_stats=True,
        **kwargs,
    )
    return LLMEngine.from_engine_args(args)


def read_state(engine) -> EngineState:
    """Live scheduler truth. Cheap enough to call every step (~microseconds)."""
    try:
        sched = engine.engine_core.engine_core.scheduler
        return EngineState(len(sched.running), len(sched.waiting),
                           float(sched.kv_cache_manager.usage))
    except Exception:
        pass
    # Fallback for a vLLM layout change. get_metrics() is a public snapshot; it is
    # slower and coarser, so the direct path above is strongly preferred.
    run = wait = 0
    usage = 0.0
    seen = False
    try:
        for m in engine.get_metrics():
            if m.name == "vllm:num_requests_running":
                run = int(getattr(m, "value", 0))
                seen = True
            elif m.name == "vllm:num_requests_waiting":
                wait = int(getattr(m, "value", 0))
                seen = True
            elif m.name == "vllm:kv_cache_usage_perc":
                usage = float(getattr(m, "value", 0.0))
    except Exception:
        pass
    # If neither path read the scheduler, say so instead of returning zeros. An
    # unreadable queue that reports 0 is indistinguishable from an empty one, and an
    # empty queue is exactly what "healthy" looks like -- so the failure mode of this
    # probe would be to certify every load as sustainable and every replica as idle.
    # The rule: a sensor's failure value must not resemble its most favourable
    # reading. Callers must check .ok before believing a zero.
    return EngineState(run, wait, usage, ok=seen)


def kv_sizing(engine) -> dict:
    """Static size of the preallocated KV block pool.

    ``kv_pool_tokens`` is the hard ceiling on (sum of context lengths) across all
    resident sequences, so it -- not ``max_num_seqs`` -- is what binds concurrency at
    long context. The two ceilings cross: at 128-token contexts a 24B model runs out
    of ``max_num_seqs`` first; at 4k contexts it runs out of KV first.
    """
    import torch  # type: ignore

    vc = engine.vllm_config
    cc, mc = vc.cache_config, vc.model_config
    n_layers = mc.get_num_layers(vc.parallel_config)
    n_kv_heads = mc.get_num_kv_heads(vc.parallel_config)
    head_size = mc.get_head_size()
    dtype_bytes = torch.tensor([], dtype=mc.dtype).element_size()
    kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_size * dtype_bytes
    n_blocks = cc.num_gpu_blocks or 0
    pool_tokens = n_blocks * cc.block_size
    out = {
        "num_gpu_blocks": int(n_blocks),
        "block_size": int(cc.block_size),
        "kv_bytes_per_token": int(kv_bytes_per_token),
        "kv_pool_tokens": int(pool_tokens),
        "kv_pool_gib": float(pool_tokens * kv_bytes_per_token / (1024 ** 3)),
        "max_model_len": int(mc.max_model_len),
        "num_layers": int(n_layers),
        "num_kv_heads": int(n_kv_heads),
        "head_size": int(head_size),
    }
    # Tensor parallel: every rank holds the same blocks for its share of the KV heads, so
    # the pool in TOKENS is the instance's, while the head count and bytes above are one
    # card's. The instance totals are added beside them rather than replacing them.
    tp = int(getattr(vc.parallel_config, "tensor_parallel_size", 1) or 1)
    if tp > 1:
        out.update({"tensor_parallel_size": tp,
                    "kv_bytes_per_token_instance": int(kv_bytes_per_token * tp),
                    "kv_pool_gib_instance": float(pool_tokens * kv_bytes_per_token * tp
                                                  / (1024 ** 3))})
    return out


def tokenizer_info(engine) -> tuple[int, set[int]]:
    tok = engine.get_tokenizer()
    return len(tok), set(tok.all_special_ids)


def teardown(engine) -> None:
    """Release the device between back-to-back builds in one process.

    A model holds ``gpu_memory_utilization`` of the card, so a second build without
    this fails to find room for its KV pool -- and the failure surfaces as an
    unhelpful "No available memory for the cache blocks" far from its cause.
    """
    import gc

    import torch  # type: ignore

    tp = 1
    try:
        tp = int(engine.vllm_config.parallel_config.tensor_parallel_size)
    except Exception:
        pass
    if tp > 1:
        # The cards belong to the worker processes: stop them, or they keep their memory
        # until this process exits. This process holds no CUDA context of its own, and
        # torch.cuda calls here would create one.
        try:
            engine.engine_core.shutdown()
        except Exception:
            pass
        try:
            del engine
        except Exception:
            pass
        gc.collect()
        return
    try:
        del engine
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
