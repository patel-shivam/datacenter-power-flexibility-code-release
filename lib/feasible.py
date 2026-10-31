"""The routing fractions that keep BOTH pools inside their own serving capacity.

Why this file exists
--------------------
A routing fraction that is fine at one offered load can overload a pool at another, and
nothing in the fraction itself says so. The single small replica sustains 69 req/s. At
an offered 66 req/s with an 80% flexible share, ``f`` = 0.95 sends 50 req/s to it, which
is fine. At 119 req/s the same ``f`` sends 90 req/s to a replica that serves 69, and its
waiting list grows for as long as the load lasts.

This file computes the band of feasible fractions against MEASURED per-replica capacity
rather than an analytic throughput model -- see ``capacity_bisection.py``, which bisects on
the slope of the scheduler's waiting list.

Units: requests, not tokens, and why that is a choice
----------------------------------------------------
The same inequalities can be written in **decode tokens per second**. This file works
in **requests per second**. Both are correct; they fail differently.

Tokens are the more fundamental coordinate — a token is roughly a fixed amount of work,
a request is not, and on a heavy-tailed shape like sharegpt one request can be two orders
of magnitude more work than another. A token-denominated band therefore transfers across
workloads, but it needs a token capacity and an ``E[D]`` to convert back to the arrival
rate the driver actually controls, and that conversion is where the analytic capacity
goes 1.7-2x wrong.

Requests are used here because that is what ``capacity_bisection.py`` **measures**: it
bisects the offered request rate until the scheduler's waiting list stops draining, on a
specific (model, workload, ``max_num_seqs``). No conversion, no ``E[D]``, no step-time
model. The cost is that the number is **shape-locked** — 21.75 req/s for an 8B at cap 64
is a fact about sharegpt's length distribution and means nothing on another one.

``load_measured`` therefore keys on (model, workload, ``max_num_seqs``) and returns None
rather than the closest match, so a missing measurement is reported as missing instead
of substituted. That is what keeps the shape-locked number from being reused where it
does not apply. If the class mix ever becomes shape-heterogeneous — flexible traffic
drawn from a different distribution than pinned traffic — this file has to move to
tokens, because a single req/s capacity would no longer describe either pool.

The packing caveat
------------------
``cap_large_pool = n_large * per_replica`` assumes every replica in the pool is available
to serve. Under ``--policy pack`` that is false by construction: the packer fills one
replica to ``pack_cap`` before lighting the next, precisely so the rest can idle and save
power. A pool with one lit replica serves one replica's worth.

So the band below is **optimistic at the low-f end** -- it says how much the large pool
could take if fully lit, not how much it can take right now. Under packing at high load
the two constraints can cross (``f_lo > f_hi``), meaning no single routing fraction is
feasible: the load needs replicas lit that the power target wants dark. That is a real
conflict between packing and capacity, not a solver artifact, and ``band()`` reports it
rather than silently returning an empty range.
"""

from __future__ import annotations

import json
from pathlib import Path

from lib import hardware


def load_measured(results_dir: str | Path, model: str, workload: str,
                  max_num_seqs: int) -> float | None:
    """Per-replica sustainable req/s, from an capacity_bisection bisection. None if not measured.

    Never falls back to an analytic estimate. A missing measurement should stop the run
    and be measured, not be silently replaced by a number that reads 1.7-2x high.
    """
    root = Path(results_dir) / "capacity_bisection"
    if not root.is_dir():
        return None
    best = None
    for d in sorted(root.iterdir()):
        f = d / "capacity.json"
        if not f.exists() or "smoke" in d.name.lower():   # shakeout runs are never a measurement
            continue
        try:
            j = json.loads(f.read_text())
        except ValueError:
            continue
        # HARDWARE IS PART OF THE KEY, even though it is not part of the signature.
        #
        # A capacity is a fact about a model on a workload at a cap ON A PARTICULAR
        # CARD. Nothing in the three named fields says which, so without this check a
        # lookup run on the RTX 8000 would return an H100 bisection -- and a caller's
        # check for a missing capacity would not fire, because a match WAS found.
        # env.results_root points the lookup at the right directory; this makes the
        # wrong directory fail loudly instead of silently matching, which is the half
        # that survives someone passing a path by hand.
        #
        # Absent means baseline, the same convention analyze/capacity_lookup.py uses:
        # H100 capacity.json files written without a `hardware` field must keep
        # matching on an H100 and stop matching anywhere else.
        hw = (j.get("hardware") or {}).get("slug") or hardware.BASELINE_SLUG
        if hw != hardware.slug():
            continue
        if (j.get("model_slug") == model
                and (j.get("workload") or {}).get("name") == workload
                and int(j.get("max_num_seqs", -1)) == int(max_num_seqs)):
            best = float(j["capacity_reqs_s"])       # later runs win
    return best


def kv_pool_tokens(results_dir: str | Path, model: str) -> float | None:
    """Tokens of KV a replica of this model can hold, from any capacity_bisection bisection.

    The KV pool is a hard ceiling on the SUM of context lengths across resident
    sequences, so a replica runs out of KV long before it runs out of sequence slots
    once the context is long. ``max_num_seqs`` is an intention; this is the constraint.

    It is a property of (model, gpu_memory_utilization, max_model_len) and NOT of the
    workload or the concurrency cap -- the pool is carved at engine build time -- so any
    capacity.json for this model reports the same figure, and they are checked against
    each other here rather than trusting whichever is found first.

    Returns None rather than guessing. A missing pool means "do not apply a KV ceiling":
    the model then runs without one.
    """
    root = Path(results_dir) / "capacity_bisection"
    if not root.is_dir():
        return None
    seen: set[float] = set()
    for d in sorted(root.iterdir()):
        f = d / "capacity.json"
        if not f.exists() or "smoke" in d.name.lower():   # shakeout runs are never a measurement
            continue
        try:
            j = json.loads(f.read_text())
        except ValueError:
            continue
        if j.get("model_slug") != model:
            continue
        tok = (j.get("kv") or {}).get("kv_pool_tokens")
        if tok:
            seen.add(float(tok))
    if not seen:
        return None
    lo, hi = min(seen), max(seen)
    # A few tenths of a percent between runs is the allocator rounding to whole blocks,
    # not a different engine configuration -- llama31-8b reports 455872 / 457664 / 457712
    # across three bisections, a 0.4% spread. Only shout when the gap is big enough to
    # mean the engine was actually built differently, and take the smallest either way
    # since that is the pool that binds soonest.
    if hi > lo * 1.02:
        print(f"  [warn] {model}: capacity_bisection reports KV pools spanning {lo:.0f}-{hi:.0f} "
              f"tokens ({hi/lo - 1:.0%}); the engine was built differently between runs. "
              f"Using the smallest.")
    return lo


def kv_concurrency_ceiling(results_dir: str | Path, model: str,
                           mean_context_tokens: float) -> float | None:
    """How many resident sequences of this shape the KV pool holds. None if unknown.

    ``mean_context_tokens`` must be the DECODE-WEIGHTED mean context
    (``Workload.decode_weighted_context``), not ``E[P] + E[D]/2``: the pool holds what is
    resident, and residence is proportional to decode length.
    """
    pool = kv_pool_tokens(results_dir, model)
    if not pool or mean_context_tokens <= 0:
        return None
    return pool / float(mean_context_tokens)


def band(lam: float, split, cap_large_pool: float, cap_small_pool: float) -> dict:
    """Feasible ``[f_lo, f_hi]`` for the flexible->small fraction, in requests/s.

        large load = large_only + (1-f)*flexible <= cap_large_pool   -> f >= f_lo
        small load = small_only +     f*flexible <= cap_small_pool   -> f <= f_hi

    ``f_lo > f_hi`` means no single fraction serves this load: the fleet is
    under-provisioned for this (rate, split), and the run will build a queue whatever the
    controller does.
    """
    flexible = split.flexible * lam
    large_only = split.large_only * lam
    small_only = split.small_only * lam
    if flexible <= 0:
        return {"f_lo": 0.0, "f_hi": 1.0, "feasible": True, "flexible_reqs_s": 0.0}
    f_lo_raw = (large_only + flexible - cap_large_pool) / flexible
    f_hi_raw = (cap_small_pool - small_only) / flexible
    f_lo, f_hi = max(0.0, f_lo_raw), min(1.0, f_hi_raw)
    return {
        "f_lo": f_lo, "f_hi": f_hi,
        "f_lo_raw": f_lo_raw, "f_hi_raw": f_hi_raw,
        "feasible": f_lo <= f_hi,
        "flexible_reqs_s": flexible,
        "cap_large_pool_reqs_s": cap_large_pool,
        "cap_small_pool_reqs_s": cap_small_pool,
    }


def report(b: dict, lam: float) -> None:
    """Print the band, and shout if the operating point is infeasible for any routing."""
    if not b["feasible"]:
        print("")
        print("  " + "!" * 76)
        print("  NO FEASIBLE ROUTING FRACTION AT THIS OFFERED LOAD")
        print(f"    {lam:.1f} req/s needs f >= {b['f_lo_raw']:.3f} to keep the large pool "
              f"inside {b['cap_large_pool_reqs_s']:.1f} req/s,")
        print(f"    and f <= {b['f_hi_raw']:.3f} to keep the small pool inside "
              f"{b['cap_small_pool_reqs_s']:.1f} req/s. The queue will grow whatever the")
        print("    controller does. Lower the offered rate or add a small replica.")
        print("  " + "!" * 76)
        print("")
        return
    tight = []
    if b["f_hi"] < 1.0:
        tight.append(f"small pool caps f at {b['f_hi']:.3f}")
    if b["f_lo"] > 0.0:
        tight.append(f"large pool floors f at {b['f_lo']:.3f}")
    note = "; ".join(tight) if tight else "neither pool binds"
    print(f"  feasible routing band: f in [{b['f_lo']:.3f}, {b['f_hi']:.3f}]  ({note})")


def _selfcheck() -> None:
    class _Split:
        def __init__(self, lo, so, fx):
            self.large_only, self.small_only, self.flexible = lo, so, fx

    # Measured fleet: 3 x 21.75 large, 1 x 69.0 small, 80% flexible.
    CL, CS = 3 * 21.75, 69.0
    s = _Split(0.2, 0.0, 0.8)

    # 66 req/s -- a load both pools can serve.
    b = band(66.1, s, CL, CS)
    assert b["feasible"], b
    assert b["f_hi"] == 1.0, b            # 0.8*66.1 = 52.9 < 69, small pool never binds
    # But the LARGE pool already binds, slightly: 3 x 21.75 = 65.25 req/s against 66.1
    # offered, so f must stay above ~0.016 or the large pool is over its own capacity
    # even before the power target is considered. There is therefore not much of a
    # margin at this load -- the fleet is at roughly 101% of the large pool.
    assert 0.0 < b["f_lo"] < 0.03, b

    # 119 req/s -- where f = 0.95 overloads the small replica (see module docstring).
    b = band(118.9, s, CL, CS)
    assert b["feasible"], b
    assert abs(b["f_hi"] - 69.0 / (0.8 * 118.9)) < 1e-9, b
    assert b["f_hi"] < 0.73, b            # f above ~0.725 overloads the small replica

    # Packing makes the low end bind: one lit large replica serves one replica's worth.
    b = band(118.9, s, cap_large_pool=21.75, cap_small_pool=CS)
    assert not b["feasible"], b           # f_lo > f_hi -- no routing works
    assert b["f_lo_raw"] > b["f_hi_raw"], b

    # All-flexible split, small pool binding.
    b = band(100.0, _Split(0.0, 0.0, 1.0), CL, CS)
    assert abs(b["f_hi"] - 0.69) < 1e-9, b

    # No flexible traffic: the knob does nothing, so nothing is infeasible.
    b = band(50.0, _Split(0.5, 0.5, 0.0), CL, CS)
    assert b["feasible"] and (b["f_lo"], b["f_hi"]) == (0.0, 1.0), b

    # --- THE HARDWARE KEY MUST FIRE, not merely exist.
    #
    # A silently-inert guard passes every test that only checks the code runs. So
    # this asserts the two directions against the REAL capacity files: an H100 lookup
    # resolves, and the identical lookup on any other card does not.
    res = Path(__file__).resolve().parent.parent / "results"
    if (res / "capacity_bisection").is_dir():
        real = load_measured(res, "llama31-8b", "sharegpt", 64)
        if real is not None:                     # skip where nothing is bisected yet
            _true_slug = hardware.slug
            try:
                hardware.slug = lambda: "rtx-8000-48gb"
                foreign = load_measured(res, "llama31-8b", "sharegpt", 64)
            finally:
                hardware.slug = _true_slug
            assert foreign is None, (
                f"hardware key did not fire: an H100 bisection ({real} req/s) was "
                f"returned for a lookup on rtx-8000-48gb")
            assert load_measured(res, "llama31-8b", "sharegpt", 64) == real, \
                "restoring hardware.slug did not restore the H100 lookup"
            print(f"          hardware key fires: {real} req/s on the baseline, "
                  f"None on rtx-8000-48gb")

    print("feasible: selfcheck ok -- open at 66 req/s, caps f at 0.725 at 119 req/s,")
    print("          and reports infeasible when packing leaves one large replica lit")


if __name__ == "__main__":
    _selfcheck()
