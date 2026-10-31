"""Arrival processes. The whole stream is materialized before the timed window opens.

Why precompute: anything the submitter does inside the loop is stolen from the loop it
is measuring. Costs of this shape (an O(vocab) prompt build per arrival; a submitter
that starves the engine it is driving) present as a capacity ceiling rather than as
overhead. So a ``Stream`` is a finished list of ``(t_offset, request_index)``
and the timed loop only compares clocks.

Processes
---------
``poisson``   Exponential inter-arrivals, CV = 1. The memoryless baseline, and what
              "offered load lambda" conventionally means.

``bursty``    On/off Markov-modulated Poisson. Alternates between an ON state
              (rate ``lam / duty``) and a silent OFF state, both with exponentially
              distributed dwell times. The mean rate is exactly ``lam`` by
              construction; ``duty`` sets how concentrated the arrivals are and
              ``burst_s`` sets the timescale over which they concentrate.

              This is the right burst model for the question being asked. Real serving
              traces are bursty in a *correlated* way -- requests cluster in time,
              which transiently drives concurrency far above its mean -- and a
              heavy-tailed-but-independent process (gamma with CV > 1) reproduces the
              variance without the clustering. Since the whole point is what happens
              to power and latency when concurrency spikes, the correlation is the
              part that matters.

``gamma``     Gamma inter-arrivals with a direct CV knob, independent draws. Useful as
              the control that isolates "variance" from "clustering": run it at the
              same CV as a bursty arm and any difference is due to correlation alone.

``replay``    The trace's own ``arrived_at`` timestamps, rescaled to a target mean
              rate. Only ``splitwise_conv``/``splitwise_code`` carry them.

A bursty stream needs a long window before it *looks* like its nominal rate
--------------------------------------------------------------------------
The realized arrival rate of an on/off process is an average over however many bursts
fit in the window, and that count is small. Measured here at rate=50/s, duty=0.25,
burst_s=4 s (so ~25 ON periods in 400 s), over 40 seeds:

    horizon     mean realized rate      sd        range
      400 s        48.1 /s            9.21    26.4 .. 66.5
     4000 s        50.1 /s            3.35    43.4 .. 56.7
    40000 s        49.8 /s            1.31    45.9 .. 52.2

The process is unbiased -- it converges to the nominal rate -- but a *single* 400 s
draw lands anywhere from half to a third above the load that was asked for. So:

  * A short measurement phase under bursty arrivals has an offered load that is only
    nominally the requested one. Every experiment here records ``stream.mean_rate``
    and reports capacity/utilization against the **realized** rate, never the nominal.
  * If an arm needs a specific offered load, either lengthen the window until the
    variance is acceptable, or use ``poisson`` (whose realized rate concentrates far
    faster) and put the burstiness somewhere it is being measured on purpose.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class Stream:
    """Materialized arrival plan. ``offsets[i]`` is when request ``i`` is submitted,
    in seconds from the start of the timed window; ``offsets`` is non-decreasing."""

    offsets: list[float]
    process: str
    params: dict

    def __len__(self) -> int:
        return len(self.offsets)

    @property
    def horizon_s(self) -> float:
        """The window the process was defined over -- NOT the last arrival time.

        These differ whenever the stream ends in a quiet stretch, which is exactly
        what a bursty process does: dividing by the last arrival would then report a
        rate measured only over the busy part and overstate the offered load (by 56%
        on a duty=0.25 stream)."""
        h = self.params.get("horizon_s")
        return float(h) if h else (self.offsets[-1] if self.offsets else 0.0)

    @property
    def mean_rate(self) -> float:
        return len(self.offsets) / self.horizon_s if self.horizon_s > 0 else 0.0

    def index_of_cv(self) -> float:
        """CV of the realized inter-arrival times -- report it, do not assume it."""
        if len(self.offsets) < 3:
            return 0.0
        gaps = [b - a for a, b in zip(self.offsets, self.offsets[1:])]
        m = sum(gaps) / len(gaps)
        if m <= 0:
            return 0.0
        var = sum((g - m) ** 2 for g in gaps) / len(gaps)
        return math.sqrt(var) / m

    def peak_rate(self, window_s: float = 1.0) -> float:
        """Highest arrival rate over any ``window_s`` window -- the burst amplitude
        that actually reaches the engine, as opposed to the nominal mean."""
        if not self.offsets:
            return 0.0
        best, j = 0, 0
        for i, t in enumerate(self.offsets):
            while self.offsets[j] < t - window_s:
                j += 1
            best = max(best, i - j + 1)
        return best / window_s


def poisson(rate: float, horizon_s: float, rng: random.Random) -> Stream:
    offsets, t = [], 0.0
    while True:
        t += rng.expovariate(rate)
        if t > horizon_s:
            break
        offsets.append(t)
    return Stream(offsets, "poisson", {"rate": rate, "horizon_s": horizon_s})


def bursty(rate: float, horizon_s: float, rng: random.Random, *,
           duty: float = 0.3, burst_s: float = 5.0) -> Stream:
    """On/off MMPP. Mean rate is ``rate``; ON-state rate is ``rate / duty``.

    ``duty`` in (0, 1]: fraction of time the source is ON. duty=1.0 degenerates to
    Poisson. ``burst_s`` is the mean ON dwell; the mean OFF dwell follows from the
    duty cycle as ``burst_s * (1 - duty) / duty``.
    """
    if not 0 < duty <= 1:
        raise ValueError("duty must be in (0, 1]")
    if duty == 1.0:
        s = poisson(rate, horizon_s, rng)
        return Stream(s.offsets, "bursty", {"rate": rate, "duty": 1.0,
                                            "burst_s": burst_s, "horizon_s": horizon_s})
    on_rate = rate / duty
    mean_off = burst_s * (1 - duty) / duty
    offsets, t, on = [], 0.0, rng.random() < duty
    while t < horizon_s:
        dwell = rng.expovariate(1.0 / (burst_s if on else mean_off))
        end = min(t + dwell, horizon_s)
        if on:
            u = t
            while True:
                u += rng.expovariate(on_rate)
                if u >= end:
                    break
                offsets.append(u)
        t = end
        on = not on
    return Stream(offsets, "bursty",
                  {"rate": rate, "duty": duty, "burst_s": burst_s,
                   "on_rate": on_rate, "horizon_s": horizon_s})


def gamma(rate: float, horizon_s: float, rng: random.Random, *, cv: float = 2.0) -> Stream:
    """Gamma inter-arrivals with coefficient of variation ``cv`` (cv=1 => Poisson).

    Shape k = 1/cv^2, scale = 1/(rate*k), so the mean gap is 1/rate for any cv."""
    k = 1.0 / (cv * cv)
    scale = 1.0 / (rate * k)
    offsets, t = [], 0.0
    while True:
        t += rng.gammavariate(k, scale)
        if t > horizon_s:
            break
        offsets.append(t)
    return Stream(offsets, "gamma", {"rate": rate, "cv": cv, "horizon_s": horizon_s})


def replay(trace_name: str, rate: float, horizon_s: float, rng: random.Random) -> Stream:
    """Replay a trace's real arrival timestamps, time-scaled to mean rate ``rate``.

    Rescaling preserves the *shape* of the arrival process (its burst structure and
    correlation) while putting it at a comparable offered load, which is the only way
    to place it on the same axis as the synthetic processes.
    """
    from lib.workloads import load_trace

    df = load_trace(trace_name)
    if "arrived_at" not in df.columns:
        raise ValueError(f"trace {trace_name!r} carries no arrival timestamps")
    t = df["arrived_at"].to_numpy()
    t = t - t.min()
    native_rate = len(t) / t.max() if t.max() > 0 else rate
    scale = native_rate / rate           # >1 stretches time to slow arrivals down
    start = rng.uniform(0, max(t.max() - horizon_s * scale, 0.0))
    sel = t[(t >= start) & (t < start + horizon_s * scale)]
    offsets = sorted(float((x - start) / scale) for x in sel)
    return Stream(offsets, "replay",
                  {"trace": trace_name, "rate": rate, "horizon_s": horizon_s,
                   "native_rate": float(native_rate), "time_scale": float(scale)})


def build(process: str, rate: float, horizon_s: float, rng: random.Random, **kw) -> Stream:
    if process == "poisson":
        return poisson(rate, horizon_s, rng)
    if process == "bursty":
        return bursty(rate, horizon_s, rng, duty=kw.get("duty", 0.3),
                      burst_s=kw.get("burst_s", 5.0))
    if process == "gamma":
        return gamma(rate, horizon_s, rng, cv=kw.get("cv", 2.0))
    if process == "replay":
        return replay(kw["trace"], rate, horizon_s, rng)
    raise ValueError(f"unknown arrival process {process!r}")


def _selfcheck() -> None:
    rng = random.Random(0)
    p = poisson(50, 200, rng)
    assert abs(p.mean_rate - 50) / 50 < 0.1, p.mean_rate
    assert abs(p.index_of_cv() - 1.0) < 0.15, p.index_of_cv()

    # A single bursty draw has huge rate variance (see module docstring), so test the
    # generator, not one sample path: average the realized rate over many seeds.
    b = bursty(50, 400, rng, duty=0.25, burst_s=4.0)
    many = [bursty(50, 400, random.Random(s), duty=0.25, burst_s=4.0).mean_rate
            for s in range(40)]
    assert abs(sum(many) / len(many) - 50) / 50 < 0.1, sum(many) / len(many)
    assert b.index_of_cv() > p.index_of_cv(), (b.index_of_cv(), p.index_of_cv())
    assert b.peak_rate(1.0) > p.peak_rate(1.0), (b.peak_rate(1.0), p.peak_rate(1.0))

    g = gamma(50, 400, rng, cv=3.0)
    assert abs(g.mean_rate - 50) / 50 < 0.25, g.mean_rate
    assert g.index_of_cv() > 2.0, g.index_of_cv()

    assert bursty(50, 200, rng, duty=1.0).process == "bursty"
    print("arrivals selfcheck OK")
    print(f"  bursty realized rate over 40 seeds: {sum(many) / len(many):.1f}/s "
          f"(nominal 50) -- single draws range "
          f"{min(many):.0f}..{max(many):.0f}/s, hence the caveat in the docstring")
    for name, s in [("poisson", p), ("bursty d=0.25", b), ("gamma cv=3", g)]:
        print(f"  {name:14s} n={len(s):5d}  mean={s.mean_rate:6.1f}/s  "
              f"cv={s.index_of_cv():4.2f}  peak(1s)={s.peak_rate(1.0):6.0f}/s")


if __name__ == "__main__":  # python -m lib.<module>
    _selfcheck()
