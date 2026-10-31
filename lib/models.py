"""The model registry: which models this suite characterizes, and how to load them.

Nine models across three families and a ~20x parameter range, all served on a single
H100-80GB. The point of the range is that a heterogeneous fleet's power knob is
*model choice*, so the study needs a ladder of power/throughput operating points, not
just a big one and a small one.

Comparability rules (deliberate, and they matter)
-------------------------------------------------
1. **Same ``gpu_memory_utilization`` for every model.** Each model gets the same
   fraction of the GPU; what differs is how much is left for KV after weights. That
   is a real property of the model, not a nuisance to be tuned away.
2. **Same ``max_model_len`` for every model.** The families disagree wildly on their
   native context (Llama-3.1 128k, Qwen3 32k+), and letting each use its own would
   make the KV block accounting incomparable.
3. **Prompts are random token ids of an exact length, decode forced to an exact
   length** (see ``request_gen.py``). No chat templates, no tokenizer differences, no
   EOS variance. Two models running "the same workload" therefore run *literally* the
   same token counts, which is what makes a cross-model power comparison meaningful.
   It also means the instruct/base distinction is irrelevant here: nothing is ever
   conditioned on the prompt's content.
4. **Reasoning / "thinking" mode is off for every model.** Qwen3 ships with thinking
   enabled by default *through its chat template*, and a model that emits a long
   ``<think>`` block before answering has a decode length set by the model rather than
   by the experiment -- which would silently break the equal-token-counts property
   above and make decode length model-dependent.

   On the current path this is already guaranteed rather than requested: a raw
   ``prompt_token_ids`` prompt never goes through a chat template, so nothing can turn
   thinking on. ``THINKING_DISABLED_KWARGS`` carries the explicit switch for any future
   path that *does* template (open-ended generation, where decode length is not forced),
   so the guarantee survives that change instead of quietly lapsing.

``max_concurrency`` grids are per-model because a 1B and a 24B do not have remotely
the same useful concurrency range; the small models are given a longer tail so their
saturation point is actually inside the swept range.

One instance across several GPUs
--------------------------------
``tensor_parallel_size > 1`` marks a model that does not fit one card and is served as
ONE instance sharded over that many H100s (vLLM tensor parallelism). It is still a
single replica -- one scheduler, one KV pool, one batch -- so every single-GPU
experiment applies unchanged, with two differences that are structural rather than
conventions: its power is the SUM over the instance's cards (``PowerSampler.for_model``),
and its results live under their own namespace (``results/h100-80gb-tp4/``, see
``hardware.set_instance``), so they can never be globbed together with one-card runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    slug: str                     # filesystem-safe short name
    family: str
    params_b: float
    concurrency_grid: tuple[int, ...]
    # Serving defaults. Uniform across models by design (see module docstring).
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    dtype: str = "bfloat16"
    notes: str = ""
    # GPUs ONE instance is sharded over (see the module docstring). 1 for every model that
    # fits a single card, which is every model the fleet experiments use.
    tensor_parallel_size: int = 1

    @property
    def weights_gib(self) -> float:
        """bf16 weight footprint, the floor under everything else on the device."""
        return self.params_b * 2.0 * 1e9 / (1024 ** 3)

    @property
    def weights_gib_per_gpu(self) -> float:
        """The weight footprint each card of the instance holds."""
        return self.weights_gib / max(int(self.tensor_parallel_size), 1)


# Concurrency ladders. Powers of two plus a few in-fills near the knee, capped where
# the model can no longer hold that many sequences' KV in what is left after weights.
_GRID_TINY = (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512)
_GRID_SMALL = (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384)
_GRID_MID = (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256)
_GRID_LARGE = (1, 2, 4, 8, 16, 32, 64, 96, 128, 192)
_GRID_XL = (1, 2, 4, 8, 16, 32, 48, 64, 96)


MODELS: dict[str, ModelSpec] = {
    m.slug: m
    for m in [
        # --- Llama. Note there is no "Llama 3.1 1B/3B": Meta shipped 1B and 3B under
        # Llama 3.2 and 8B under Llama 3.1. Both are used here.
        ModelSpec("meta-llama/Llama-3.2-1B", "llama32-1b", "llama", 1.24, _GRID_TINY),
        ModelSpec("meta-llama/Llama-3.2-3B", "llama32-3b", "llama", 3.21, _GRID_SMALL),
        ModelSpec("meta-llama/Llama-3.1-8B", "llama31-8b", "llama", 8.03, _GRID_MID),
        # --- Qwen3
        ModelSpec("Qwen/Qwen3-1.7B", "qwen3-1.7b", "qwen3", 1.72, _GRID_TINY),
        ModelSpec("Qwen/Qwen3-4B", "qwen3-4b", "qwen3", 4.02, _GRID_SMALL),
        ModelSpec("Qwen/Qwen3-8B", "qwen3-8b", "qwen3", 8.19, _GRID_MID),
        ModelSpec("Qwen/Qwen3-14B", "qwen3-14b", "qwen3", 14.8, _GRID_LARGE),
        # --- Mistral
        ModelSpec("mistralai/Mistral-7B-Instruct-v0.2", "mistral-7b", "mistral", 7.24,
                  _GRID_MID),
        ModelSpec("mistralai/Mistral-Small-24B-Instruct-2501", "mistral-24b", "mistral", 23.6,
                  _GRID_XL,
                  notes="47 GiB of bf16 weights leaves ~25 GiB for KV at util=0.90; "
                        "the concurrency ceiling here is KV, not compute."),
        # --- One instance over four H100s (appendix). The Instruct checkpoint: the
        # architecture, shapes and kernels are the base model's, and nothing here
        # conditions on content (rule 3).
        ModelSpec("meta-llama/Llama-3.1-70B-Instruct", "llama31-70b", "llama", 70.55,
                  _GRID_MID, tensor_parallel_size=4,
                  notes="131 GiB of bf16 weights, ~33 GiB per card at TP=4; 80 layers x 8 KV "
                        "heads x 128 = 320 KiB of KV per token for the instance (80 KiB per "
                        "card). Power is the sum over the four cards."),
    ]
}

ALL_SLUGS = tuple(MODELS)

# Pass to a chat-templated generate() call to keep reasoning off. A no-op on the
# raw-token-id path used by every experiment here, which bypasses templating entirely.
THINKING_DISABLED_KWARGS = {"chat_template_kwargs": {"enable_thinking": False}}


def get(slug_or_id: str) -> ModelSpec:
    if slug_or_id in MODELS:
        return MODELS[slug_or_id]
    for m in MODELS.values():
        if m.model_id == slug_or_id:
            return m
    raise KeyError(f"unknown model {slug_or_id!r}; known: {', '.join(ALL_SLUGS)}")


def by_family(family: str) -> list[ModelSpec]:
    return [m for m in MODELS.values() if m.family == family]


def ordered_by_size() -> list[ModelSpec]:
    return sorted(MODELS.values(), key=lambda m: m.params_b)


def single_gpu() -> list[ModelSpec]:
    """The models that fit one card -- the fleet's candidates."""
    return [m for m in MODELS.values() if m.tensor_parallel_size == 1]


def _selfcheck() -> None:
    assert len(MODELS) == 10, len(MODELS)
    assert len(single_gpu()) == 9, [m.slug for m in single_gpu()]
    assert get("meta-llama/Llama-3.1-8B").slug == "llama31-8b"
    assert get("llama31-8b").model_id == "meta-llama/Llama-3.1-8B"
    assert get("llama31-70b").tensor_parallel_size == 4
    sizes = [m.params_b for m in ordered_by_size()]
    assert sizes == sorted(sizes)
    # Each card's share of the weights must leave room for KV on an 80 GiB card at the
    # stated util -- the whole footprint for a one-card model.
    for m in MODELS.values():
        budget = 80.0 * m.gpu_memory_utilization
        assert m.weights_gib_per_gpu < budget, (m.slug, m.weights_gib_per_gpu, budget)
        assert m.tensor_parallel_size > 1 or m.weights_gib == m.weights_gib_per_gpu
    print(f"models selfcheck OK: {len(MODELS)} models, "
          f"{sizes[0]:.1f}B -> {sizes[-1]:.1f}B")
    for m in ordered_by_size():
        tp = f"  TP={m.tensor_parallel_size} ({m.weights_gib_per_gpu:.1f} GiB/card)" \
            if m.tensor_parallel_size > 1 else ""
        print(f"  {m.slug:14s} {m.params_b:5.2f}B  weights {m.weights_gib:5.1f} GiB  "
              f"concurrency {m.concurrency_grid[0]}..{m.concurrency_grid[-1]} "
              f"({len(m.concurrency_grid)} levels){tp}")


if __name__ == "__main__":  # python -m lib.<module>
    _selfcheck()
