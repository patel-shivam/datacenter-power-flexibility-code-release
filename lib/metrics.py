"""Latency, throughput, and SLO accounting.

Definitions, stated because papers disagree about them
------------------------------------------------------
``TTFT``  time to first token: submission -> first output token. Dominated by queueing
          plus prefill.
``TBT``   time between tokens (a.k.a. inter-token latency): the mean gap between
          consecutive output tokens *within one request*, over that request's decode.
          Reported per request and then aggregated, so a long request does not
          out-vote a short one.
``E2E``   submission -> last token.

TBT is the SLO that matters for the packing cap. It degrades smoothly with concurrency,
which is exactly what makes it usable as the thing ``pack_cap`` is calibrated against:
the cap is the largest batch at which the model still meets its TBT target.

Percentiles are computed on the empirical distribution with no interpolation -- p99 of
150 samples is the 149th value, and it is reported alongside n so that a p99 computed
from too few samples is visible as such rather than quietly believed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass
class RequestRecord:
    """Per-request timing. Times are ``time.monotonic()`` seconds."""

    request_id: str
    replica_idx: int
    role: str
    req_class: str
    num_prefill_tokens: int
    num_decode_tokens: int
    t_submit: float
    t_first_token: float = math.nan
    t_last_token: float = math.nan
    tokens_emitted: int = 0

    @property
    def ttft_ms(self) -> float:
        return (self.t_first_token - self.t_submit) * 1e3

    @property
    def e2e_ms(self) -> float:
        return (self.t_last_token - self.t_submit) * 1e3

    @property
    def tbt_ms(self) -> float:
        """Mean gap between output tokens. Needs >= 2 tokens to be defined."""
        if self.tokens_emitted < 2 or not math.isfinite(self.t_first_token):
            return math.nan
        return ((self.t_last_token - self.t_first_token) / (self.tokens_emitted - 1)) * 1e3

    @property
    def complete(self) -> bool:
        return math.isfinite(self.t_last_token)


def percentiles(values, qs=(50, 90, 95, 99)) -> dict:
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return {f"p{q}": math.nan for q in qs} | {"mean": math.nan, "n": 0}
    out = {}
    for q in qs:
        k = min(len(vals) - 1, max(0, int(math.ceil(q / 100.0 * len(vals))) - 1))
        out[f"p{q}"] = vals[k]
    out["mean"] = sum(vals) / len(vals)
    out["n"] = len(vals)
    return out


@dataclass
class SLO:
    """Latency targets. ``None`` disables that component."""

    ttft_ms: float | None = None
    tbt_ms: float | None = None

    def met(self, r: RequestRecord) -> bool:
        if self.ttft_ms is not None and not (r.ttft_ms <= self.ttft_ms):
            return False
        if self.tbt_ms is not None:
            t = r.tbt_ms
            if math.isnan(t) or t > self.tbt_ms:
                return False
        return True


def summarize(records: list[RequestRecord], window_s: float | None = None,
              slo: SLO | None = None) -> dict:
    """Aggregate a set of completed requests."""
    done = [r for r in records if r.complete]
    out = {
        "n_submitted": len(records),
        "n_completed": len(done),
        "ttft_ms": percentiles([r.ttft_ms for r in done]),
        "tbt_ms": percentiles([r.tbt_ms for r in done]),
        "e2e_ms": percentiles([r.e2e_ms for r in done]),
    }
    prefill = sum(r.num_prefill_tokens for r in done)
    decode = sum(r.tokens_emitted for r in done)
    out["prefill_tokens"] = prefill
    out["decode_tokens"] = decode
    if window_s and window_s > 0:
        out["throughput_reqs_s"] = len(done) / window_s
        out["throughput_decode_tok_s"] = decode / window_s
        out["throughput_total_tok_s"] = (prefill + decode) / window_s
    if slo is not None and done:
        ok = sum(1 for r in done if slo.met(r))
        out["slo_attainment"] = ok / len(done)
        out["slo_target"] = {"ttft_ms": slo.ttft_ms, "tbt_ms": slo.tbt_ms}
        # Goodput: completions that also met the SLO. Throughput alone can be raised by
        # admitting more work than the SLO tolerates, so it is not on its own a measure
        # of useful capacity.
        if window_s and window_s > 0:
            out["goodput_reqs_s"] = ok / window_s
    return out


class Stopwatch:
    """Steady-state window helper: discard a warmup prefix, then time the rest.

    Every throughput and power number in this suite is taken over a window that starts
    *after* the engine has reached steady state. Measuring from t=0 mixes in the ramp
    from an empty queue, which at high concurrency takes tens of seconds and biases
    throughput down and power down together -- consistently enough to look like a real
    operating point rather than a transient.
    """

    def __init__(self, warmup_s: float, measure_s: float):
        self.warmup_s = warmup_s
        self.measure_s = measure_s
        self.t0 = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    @property
    def in_warmup(self) -> bool:
        return self.elapsed < self.warmup_s

    @property
    def in_window(self) -> bool:
        return self.warmup_s <= self.elapsed < self.warmup_s + self.measure_s

    @property
    def done(self) -> bool:
        return self.elapsed >= self.warmup_s + self.measure_s

    @property
    def window_start(self) -> float:
        return self.t0 + self.warmup_s

    @property
    def window_end(self) -> float:
        return self.t0 + self.warmup_s + self.measure_s


def _selfcheck() -> None:
    r = RequestRecord("a", 0, "large", "flexible", 128, 10, t_submit=0.0)
    r.t_first_token, r.t_last_token, r.tokens_emitted = 0.05, 0.95, 10
    assert abs(r.ttft_ms - 50.0) < 1e-6
    assert abs(r.tbt_ms - 100.0) < 1e-6, r.tbt_ms   # 0.9 s over 9 gaps
    assert abs(r.e2e_ms - 950.0) < 1e-6

    p = percentiles([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert p["p50"] == 5 and p["p90"] == 9 and p["p99"] == 10 and p["n"] == 10, p
    assert math.isnan(percentiles([])["p50"])

    recs = []
    for i in range(100):
        x = RequestRecord(str(i), 0, "large", "flexible", 128, 10, 0.0)
        x.t_first_token = 0.01 * (1 + i / 100)
        x.t_last_token = x.t_first_token + 0.9
        x.tokens_emitted = 10
        recs.append(x)
    s = summarize(recs, window_s=10.0, slo=SLO(ttft_ms=15.0, tbt_ms=200.0))
    assert s["n_completed"] == 100
    assert abs(s["throughput_reqs_s"] - 10.0) < 1e-9
    assert abs(s["throughput_decode_tok_s"] - 100.0) < 1e-9
    assert 0.4 < s["slo_attainment"] < 0.6, s["slo_attainment"]
    assert abs(s["goodput_reqs_s"] - s["slo_attainment"] * 10.0) < 1e-9

    incomplete = RequestRecord("z", 0, "large", "flexible", 8, 4, 0.0)
    assert not incomplete.complete
    assert summarize([incomplete])["n_completed"] == 0

    print("metrics selfcheck OK")
    print(f"  ttft p50/p99 = {s['ttft_ms']['p50']:.2f}/{s['ttft_ms']['p99']:.2f} ms, "
          f"SLO attainment {s['slo_attainment']:.0%}, "
          f"goodput {s['goodput_reqs_s']:.1f} req/s of {s['throughput_reqs_s']:.1f} req/s")


if __name__ == "__main__":  # python -m lib.<module>
    _selfcheck()
