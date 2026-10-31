"""Task types: the (prefill, decode) length distributions the fleet is measured on.

Why the workload is defined as a *length distribution* and not as text
---------------------------------------------------------------------
Every request this suite issues is a random token-id prompt of an exact length with
the decode length forced exactly (``ignore_eos``, ``min_tokens == max_tokens``). The
content is never used. What is replayed from a real workload is therefore its
**shape distribution**, not its text.

That is a deliberate trade and it is what makes the cross-model comparison valid:

  * Nine models across three families have three different tokenizers. The *same*
    ShareGPT conversation is 180 tokens to one and 210 to another. If each model
    tokenized the text itself, "the same workload" would silently be a different
    number of tokens per model, and any power/throughput comparison between them
    would be confounded by tokenizer efficiency.
  * Forcing the decode length removes EOS variance, which is otherwise a large and
    model-dependent source of spread -- an instruct model stops early, a base model
    rambles to the cap.

So lengths are derived **once**, with one reference tokenizer, and then replayed
verbatim on every model. What is lost is any effect of prompt *content* on power
(essentially nil at fixed shape -- power is set by batch geometry) and any study of
early-EOS behaviour (out of scope; it would be a workload difference, not a power one).

Task types
----------
Real traces, checked in under ``data/traces/`` (from vidur's processed traces of the
Azure LLM inference traces) plus ShareGPT prepared by ``prepare_traces.py``:

  ``sharegpt``              chat. The field's default benchmark. Medium prefill,
                            medium decode, very heavy right tail on both.
  ``splitwise_conv``        Azure production conversation. Carries real arrival
                            timestamps, so it can drive arrival replay as well as
                            shape sampling.
  ``splitwise_code``        Azure production code completion. Long prefill, tiny
                            decode -- the prefill-dominated corner.
  ``arxiv_summarization``   Long-document summarization. The longest prefills here;
                            the regime where KV, not compute, sets concurrency.
  ``azure_conv_2024``       Azure production conversation, one week of 2024 (lengths only,
                            from ``prepare_azure_trace.py``): the requests whose timing the
                            datacenter simulation replays. Long prompt, short output.
  ``azure_code_2024``       Azure production code completion, 2024. Long prompt, tiny output.

Controlled synthetic shapes, for ablations where a distribution would confound:

  ``fixed_128_128``         short and balanced; the first rung of the 1:1 context
                            ladder.
  ``fixed_512_512``         balanced, longer.
  ``fixed_2048_128``        prefill-heavy control against ``splitwise_code``.
  ``fixed_128_1024``        decode-heavy control; the regime where a replica sits in
                            long decode and concurrency is set by ``max_num_seqs``.

The fixed shapes are not decoration. Prefill and decode work are perfectly collinear
in any single trace at a fixed mean, which is exactly the degeneracy that keeps a model
fitted on one trace from separating a per-step cost from a per-token one. Sweeping the
fixed shapes breaks it.
"""

from __future__ import annotations

import functools
import math
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lib.env import DERIVED_TRACES_DIR, TRACES_DIR

# name -> (directory, filename). Checked-in traces (TRACES_DIR) and those derived by the
# prepare_*.py scripts (DERIVED_TRACES_DIR) both live under data/traces/.
_TRACE_FILES: dict[str, tuple[Path, str]] = {
    "splitwise_conv": (TRACES_DIR, "splitwise_conv.csv"),
    "splitwise_code": (TRACES_DIR, "splitwise_code.csv"),
    "arxiv_summarization": (TRACES_DIR,
                            "arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv"),
    "sharegpt": (DERIVED_TRACES_DIR, "sharegpt_llama31.csv"),
    # prepare_humaneval_trace.py: 164 problems, prompt = signature + docstring, output =
    # canonical solution, both counted under the Llama-3.1 tokenizer like sharegpt.
    "humaneval": (DERIVED_TRACES_DIR, "humaneval_llama31.csv"),
    # prepare_azure_trace.py: 100,000 requests sampled uniformly from each week of the Azure
    # LLM inference trace 2024 -- the production token counts, replayed as lengths.
    "azure_conv_2024": (DERIVED_TRACES_DIR, "azure_conv_2024.csv"),
    "azure_code_2024": (DERIVED_TRACES_DIR, "azure_code_2024.csv"),
}


@functools.lru_cache(maxsize=8)
def load_trace(name: str) -> pd.DataFrame:
    """Load a length trace with columns ``num_prefill_tokens``/``num_decode_tokens``."""
    if name not in _TRACE_FILES:
        raise ValueError(f"unknown trace {name!r}; known: {', '.join(sorted(_TRACE_FILES))}")
    directory, fname = _TRACE_FILES[name]
    path = directory / fname
    if not path.exists():
        hint = ("  run `prepare_traces.py` to build it"
                if directory == DERIVED_TRACES_DIR else "")
        raise FileNotFoundError(f"trace {name!r} not found at {path}{hint}")
    df = pd.read_csv(path)
    return df[(df["num_prefill_tokens"] > 0) & (df["num_decode_tokens"] > 0)].reset_index(drop=True)


def _logu_mean(a: float, b: float) -> float:
    return (b - a) / math.log(b / a)


def _logu_second(a: float, b: float) -> float:
    return (b * b - a * a) / (2.0 * math.log(b / a))


@dataclass(frozen=True)
class Workload:
    """A named (prefill, decode) length distribution, with optional clipping.

    ``max_prefill``/``max_decode`` clip rather than drop, so clipping changes the mean
    without changing the number of requests or reweighting the body of the
    distribution. ``max_prefill`` also has to respect the engine's ``max_model_len``:
    a request longer than the context window is rejected, not truncated, by vLLM.
    """

    name: str
    kind: str                       # "trace" | "fixed"
    trace: str | None = None
    prefill: int = 0                # kind == "fixed"
    decode: int = 0
    max_prefill: int | None = None  # kind == "trace"; also the UPPER bound for loguniform
    max_decode: int | None = None
    description: str = ""
    # kind == "loguniform": independent log-uniform draws on [min_*, max_*]. Exists for
    # fitting a per-token-batch step-time model, which needs token batches that DIFFER
    # from each other. A trace has one narrow length distribution and a fixed shape has
    # none at all, so under either the prefill chunk is nearly always the same size and
    # the prefill-attention regressor collapses onto the token-count regressor (measured
    # r = +0.861, condition number 8.9e5). Log-uniform spreads both lengths over an order
    # of magnitude, which is what makes the coefficients separately identifiable.
    min_prefill: int = 0
    min_decode: int = 0

    # --- sampling ---------------------------------------------------------------
    def sample(self, n: int, rng: random.Random) -> list[tuple[int, int]]:
        if self.kind == "fixed":
            return [(self.prefill, self.decode)] * n
        if self.kind == "loguniform":
            import math as _m

            def draw(lo, hi):
                return int(round(_m.exp(rng.uniform(_m.log(lo), _m.log(hi)))))
            return [(max(draw(self.min_prefill, self.max_prefill), 1),
                     max(draw(self.min_decode, self.max_decode), 1)) for _ in range(n)]
        df = load_trace(self.trace)
        p = df["num_prefill_tokens"].to_numpy()
        d = df["num_decode_tokens"].to_numpy()
        idx = [rng.randrange(len(p)) for _ in range(n)]
        out = []
        for i in idx:
            pi, di = int(p[i]), int(d[i])
            if self.max_prefill is not None:
                pi = min(pi, self.max_prefill)
            if self.max_decode is not None:
                di = min(di, self.max_decode)
            out.append((max(pi, 1), max(di, 1)))
        return out

    # --- moments ----------------------------------------------------------------
    def _clipped(self, col: str, cap: int | None):
        s = load_trace(self.trace)[col]
        return s.clip(upper=cap) if cap is not None else s

    @property
    def mean_prefill(self) -> float:
        if self.kind == "fixed":
            return float(self.prefill)
        if self.kind == "loguniform":
            return _logu_mean(self.min_prefill, self.max_prefill)
        return float(self._clipped("num_prefill_tokens", self.max_prefill).mean())

    @property
    def mean_decode(self) -> float:
        if self.kind == "fixed":
            return float(self.decode)
        if self.kind == "loguniform":
            return _logu_mean(self.min_decode, self.max_decode)
        return float(self._clipped("num_decode_tokens", self.max_decode).mean())

    @property
    def second_moment_decode(self) -> float:
        """E[D^2]. Drain time and actuation bandwidth need it, not just the mean: the
        in-flight population is length-biased, so expected remaining decode work goes
        as E[D^2]/E[D]. A fixed shape has Var(D)=0 and is therefore a best case."""
        if self.kind == "fixed":
            return float(self.decode) ** 2
        if self.kind == "loguniform":
            return _logu_second(self.min_decode, self.max_decode)
        return float((self._clipped("num_decode_tokens", self.max_decode) ** 2).mean())

    @property
    def decode_weighted_context(self) -> float:
        """Mean context of a RESIDENT sequence -- what the KV pool actually holds.

        This is not ``E[P] + E[D]/2``. That weights every request equally, but the KV
        pool holds whatever is resident at an instant, and a request occupies the pool
        for as many steps as it decodes. A request that emits 1000 tokens puts its
        context into 1000 steps; one that emits 10 puts it into 10. So the average has to
        be weighted by decode length:

            c_bar = E[ P*D + D(D+1)/2 ] / E[D]

        On a heavy tail the two differ materially -- 311 against 242 on sharegpt (1.29x),
        and 2357 against 2720 on arxiv, where the naive figure is 15% HIGH because
        arxiv's long prompts come with short completions. Using the wrong one is worth
        20.6% MAPE where the same quantity is the attention term of a per-token-batch
        step-time model.

        Used for the KV ceiling: a replica holds at most ``kv_pool_tokens / c_bar``
        sequences, whatever ``max_num_seqs`` says.

        Why dividing one expectation by another is legitimate here
        ----------------------------------------------------------
        It looks like the ``E[X/Y] != E[X]/E[Y]`` trap and is not. ``c_bar`` is defined as
        a ratio of two TIME AVERAGES over the same arrival stream, not as the expectation
        of a per-request ratio:

            c_bar = (time-average total KV tokens held) / (time-average number resident)

        Both are sums over the same requests, so the ratio of the sums converges to the
        ratio of the means by the strong law of large numbers. In queueing terms the
        numerator is **Brumelle's H = lambda*G** -- the generalisation of Little's Law to a
        reward accrued at a rate depending on time-in-system, here the reward being
        "tokens of KV held per step" -- and the denominator is **Little's Law** itself,
        L = lambda*E[D]. Dividing, lambda cancels:

            H / L  =  lambda*E[P*D + D(D+1)/2] / (lambda*E[D])

        This is the standard form of the renewal-reward theorem, whose statement *is* a
        ratio of expectations: long-run average reward rate = E[reward per cycle] /
        E[cycle length]. See Ross, *Stochastic Processes* (renewal-reward); Wolff,
        *Stochastic Modeling and the Theory of Queues* (H = lambda*G); Brumelle (1971),
        "On the relation between customer and time averages in queues". Equivalently it is
        the Palm inversion formula relating a time-stationary average to a per-arrival one.

        Note what is NOT assumed: ``E[P*D]`` is taken from the JOINT sample, never as
        ``E[P]*E[D]``. That matters -- corr(P, D) is -0.47 on arxiv, and assuming
        independence is what makes the naive ``E[P] + E[D]/2`` wrong there.

        Checked numerically as well as derived: an explicit step-by-step occupancy
        simulation, accumulating both time averages directly, agrees with this closed form
        to **0.2%** on sharegpt, arxiv and fixed_6144_1024.

        Three things it assumes, in order of how much they cost
        ------------------------------------------------------
        1. **It is a MEAN, and KV is an INSTANTANEOUS constraint.** ``kv_pool / c_bar`` is
           the batch size at which the *average* KV usage reaches the pool; the sum of
           contexts fluctuates around that mean, so a real engine must sit below it or
           preempt. The ceiling is therefore **optimistic by construction**, which
           predicts the sign of the residual: 193 predicted against ~180 measured on
           arxiv, 68 against 66 on fixed_6144_1024. Both high, as they should be.
        2. **Stationarity.** It is a steady-state statement. It says nothing about a
           transient or a diverging queue -- which is exactly why fleet probes are
           filtered on their waiting-list slope before anything is scored against it.
        3. **Prefill residence is ignored** -- a request also holds KV while its prefill
           chunks run. Measured cost of leaving it out: **<= 0.1%**, because decode steps
           outnumber prefill chunks by two orders of magnitude (262 against 1.26 on arxiv).
        """
        if self.kind == "fixed":
            P, D = float(self.prefill), float(self.decode)
            return P + (D + 1.0) / 2.0
        # loguniform and trace: take it from the sampled pairs, which already apply the
        # same clipping the workload serves with. Deterministic seed so the model does
        # not move between calls.
        import random as _random
        pairs = self.sample(200_000, _random.Random(0))
        sd = sum(d for _, d in pairs)
        if sd <= 0:
            return self.mean_prefill
        num = sum(p * d + d * (d + 1) / 2.0 for p, d in pairs)
        return float(num / sd)

    @property
    def cv_decode(self) -> float:
        """Coefficient of variation of the decode length. 0 for fixed shapes."""
        m = self.mean_decode
        var = max(self.second_moment_decode - m * m, 0.0)
        return (var ** 0.5) / m if m > 0 else 0.0

    def descriptor(self) -> dict:
        d = {"name": self.name, "kind": self.kind}
        if self.kind == "loguniform":
            d.update(min_prefill=self.min_prefill, max_prefill=self.max_prefill,
                     min_decode=self.min_decode, max_decode=self.max_decode)
        elif self.kind == "fixed":
            d.update(prefill=self.prefill, decode=self.decode)
        else:
            d.update(trace=self.trace, max_prefill=self.max_prefill, max_decode=self.max_decode)
        return d

    def matches(self, descriptor: dict | None) -> bool:
        """Guard against using a capacity/power number measured at a different shape."""
        return descriptor is not None and descriptor == self.descriptor()

    def label(self) -> str:
        if self.kind == "fixed":
            return f"{self.name}(P={self.prefill},D={self.decode})"
        return f"{self.name}(trace={self.trace})"


# max_prefill is clipped to 4096 for the long-prefill traces: it keeps a single request
# inside the uniform 8192-token max_model_len with room for its decode, and it bounds
# the KV a single sequence can pin so a concurrency sweep is not silently truncated by
# one outlier request.
WORKLOADS: dict[str, Workload] = {
    w.name: w
    for w in [
        Workload("sharegpt", "trace", trace="sharegpt", max_prefill=4096, max_decode=2048,
                 description="ShareGPT chat; the field's default serving benchmark"),
        Workload("splitwise_conv", "trace", trace="splitwise_conv",
                 max_prefill=4096, max_decode=2048,
                 description="Azure production conversation trace (has real arrivals)"),
        Workload("splitwise_code", "trace", trace="splitwise_code",
                 max_prefill=4096, max_decode=2048,
                 description="Azure production code completion; long prefill, tiny decode"),
        Workload("arxiv_summarization", "trace", trace="arxiv_summarization",
                 max_prefill=4096, max_decode=2048,
                 description="Long-document summarization; the KV-bound corner"),
        # HumanEval code generation: short prompts (~130 tokens) and short outputs (the
        # canonical solution, ~60 tokens). 164 problems sampled with replacement, so a
        # 4096-request pool repeats each ~25 times; lengths only, as everywhere here.
        Workload("humaneval", "trace", trace="humaneval", max_prefill=4096, max_decode=2048,
                 description="HumanEval code completion; short prompt, short output"),
        # The Azure LLM inference trace 2024 (one week each; 27.3 M / 16.8 M requests): the
        # lengths of the requests whose TIMING drives the datacenter simulation. Long prompts
        # and short outputs, the opposite of sharegpt. Clipped at 7040 / 1024 instead of the
        # 4096 the other long-prefill traces use: 7040 + 1024 = 8064 stays inside max_model_len
        # 8192, and it changes 0.64 % of conversation prompts (means -0.2 % / -0.1 %) and 8.3 %
        # of code prompts (means -1.9 % / -1.9 %), where 4096 would clip 8.8 % / 18.5 %.
        Workload("azure_conv_2024", "trace", trace="azure_conv_2024",
                 max_prefill=7040, max_decode=1024,
                 description="Azure production conversation, 2024 week; long prompt, short output"),
        Workload("azure_code_2024", "trace", trace="azure_code_2024",
                 max_prefill=7040, max_decode=1024,
                 description="Azure production code completion, 2024 week; long prompt, tiny output"),
        Workload("fixed_128_128", "fixed", prefill=128, decode=128,
                 description="1:1 ladder rung 1; short and balanced"),
        # The 1:1 CONTEXT LADDER: 128/128, 256/256, 512/512, 1024/1024. P:D is pinned at
        # 1:1 so only the SCALE varies, which is what makes it the controlled axis
        # against arxiv (prefill-dominated) and sharegpt (decode-dominated, heavy-tailed).
        #
        # Every prefill here is <= 1024, below vLLM's 2048-token default chunk budget, so
        # each prompt is a SINGLE chunk and the token budget never binds. That is the
        # point: the model's 19% capacity shortfall on arxiv traces to exactly that
        # budget clamping step time, and this ladder is the arm where it provably
        # cannot. Deviations along it are the residence law's own.
        #
        # The ladder also stays REQUEST-capped the whole way up, which is what makes it a
        # clean control. Against the 8B's measured 455,872-token pool the decode-weighted
        # context and resulting c_kv are 192/2368, 384/1186, 768/593 and 1536/297 -- so
        # even the top rung holds 297 sequences against the caps we run (64, 128), and
        # 0.9*c_kv = 267 keeps a cap of 256 legal too. The ladder therefore does not drift
        # into the KV-bound regime partway up; fixed_6144_1024 below (c_kv = 68) is the
        # shape for that, deliberately separate.
        Workload("fixed_256_256", "fixed", prefill=256, decode=256,
                 description="1:1 ladder rung 2; single-chunk prefill"),
        Workload("fixed_512_512", "fixed", prefill=512, decode=512,
                 description="1:1 ladder rung 3; balanced, longer"),
        Workload("fixed_1024_1024", "fixed", prefill=1024, decode=1024,
                 description="1:1 ladder rung 4; largest prefill still inside one chunk"),
        Workload("fixed_2048_128", "fixed", prefill=2048, decode=128,
                 description="prefill-heavy control"),
        # 1:1 ladder rung 5 for the KV-ceiling figure: resident context 2048 + 1024.5 =
        # 3072.5 tokens, so the 8B's ~456k-token pool holds ~148 sequences -- the KV limit
        # binds below a 256 cap where 1024/1024 (297) only just does. Two prefill chunks.
        Workload("fixed_2048_2048", "fixed", prefill=2048, decode=2048,
                 description="1:1 ladder rung 5; KV binds at ~148 seqs on the 8B"),
        Workload("fixed_128_1024", "fixed", prefill=128, decode=1024,
                 description="decode-heavy control"),
        # The only shape here that makes the KV POOL the binding constraint rather than
        # the request cap. Every other workload leaves headroom: at the 8B's measured
        # 457,712-token pool, sharegpt holds ~1,900 sequences against a cap of 64 (30x)
        # and even arxiv_summarization holds ~168 (2.6x), so none of them tests what
        # happens when KV binds.
        #
        # 6144 + 1024 = 7168 tokens fits inside max_model_len 8192 with room for the
        # chunked-prefill scheduler to work. Mean resident context is 6144 + 1024/2 =
        # 6656, so the pool holds ~69 sequences -- binding at cap 128 and above, and
        # marginal at 64. For a hard bind at 64 combine it with a reduced
        # --gpu-memory-utilization, which shrinks the pool without touching the workload.
        #
        # The decode length matters as much as the prefill: KV pressure is context TIMES
        # residence, and a sequence with a 28-token decode (splitwise_code) leaves almost
        # immediately however long its prompt was. 1024 decode tokens keep it resident.
        Workload("fixed_6144_1024", "fixed", prefill=6144, decode=1024,
                 description="long-context, KV-bound: ~69 seqs fit the 8B's pool"),
        # For fitting a per-token-batch step-time model ONLY -- not a serving workload and
        # not meant to resemble one. Prefill spans 6.6 doublings and decode 6, independently,
        # so a token batch mixes chunk sizes and context lengths instead of repeating one
        # composition. 6144+1024 stays inside max_model_len 8192 at the top corner.
        Workload("mixed_lengths", "loguniform",
                 min_prefill=64, max_prefill=6144, min_decode=16, max_decode=1024,
                 description="log-uniform P and D; diverse batch compositions for "
                             "identifying per-token-batch step-time coefficients"),
    ]
}

ALL_WORKLOADS = tuple(WORKLOADS)
# The default sweep set: two real traces spanning the prefill/decode plane, plus one
# fixed shape.
DEFAULT_SWEEP = ("sharegpt", "splitwise_code", "fixed_128_128")


def get(name: str) -> Workload:
    if name not in WORKLOADS:
        raise KeyError(f"unknown workload {name!r}; known: {', '.join(ALL_WORKLOADS)}")
    return WORKLOADS[name]


def _selfcheck() -> None:
    rng = random.Random(0)
    f = get("fixed_128_128")
    assert f.sample(5, rng) == [(128, 128)] * 5
    assert f.mean_decode == 128.0 and f.cv_decode == 0.0
    assert f.matches(f.descriptor())
    assert not f.matches(get("fixed_512_512").descriptor())
    # every request any workload can produce fits the uniform 8192-token max_model_len
    # (vLLM rejects, rather than truncates, a request longer than the context window)
    for w in WORKLOADS.values():
        top = (w.prefill + w.decode) if w.kind == "fixed" else (w.max_prefill or 0) + (w.max_decode or 0)
        assert 0 < top <= 8192, (w.name, top)
    print("workloads selfcheck OK")
    print(f"  {'workload':22s} {'meanP':>7} {'meanD':>7} {'cv(D)':>6}  rows")
    for name in ALL_WORKLOADS:
        w = get(name)
        try:
            n = len(load_trace(w.trace)) if w.kind == "trace" else 0
            print(f"  {name:22s} {w.mean_prefill:7.0f} {w.mean_decode:7.0f} "
                  f"{w.cv_decode:6.2f}  {n or '-'}")
        except FileNotFoundError as e:
            print(f"  {name:22s} MISSING -- {e}")


if __name__ == "__main__":  # python -m lib.<module>
    _selfcheck()
