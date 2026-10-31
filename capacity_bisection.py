#!/usr/bin/env python
"""capacity_bisection -- open-loop capacity: the arrival rate at which the queue stops draining.

Capacity here means the largest **open-loop** arrival rate a single replica can sustain
without its queue growing without bound. That is not the same as the throughput a
closed-loop benchmark reports, and the difference is not small: holding concurrency
constant and resubmitting on completion measures what the engine can retire when work is
always available, which overstates what a live arrival stream sustains by roughly 1.6x.

Measuring it correctly turns on one detail
------------------------------------------
Stability must be judged from the scheduler's **waiting** list, not from the number of
unfinished requests. ``get_num_unfinished_requests()`` is running + waiting, and by
Little's Law the running part grows in proportion to the offered rate even when the
queue is perfectly stable. Reading that as "queue depth" makes a healthy system look
saturated and halves every capacity number derived from it. This suite reads
``scheduler.waiting`` directly, which is only possible because the engine core runs
in-process (see ``lib/env.py``).

The test at each candidate rate is a **trend**, not a level: fit a line to the waiting
depth over the measurement window and require the slope to be flat. A deep-but-flat
queue is stable; a shallow-but-rising one is not.

Bisection, not a sweep, because the boundary is what is wanted and a sweep spends most
of its samples far from it.

    python capacity_bisection.py \\
        --model llama31-8b --workload sharegpt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import env  # noqa: F401
from lib import arrivals
from lib import engine as eng
from lib import hardware, metrics, models
from lib import power_sampler as ps
from lib import request_gen as reqlib
from lib import workloads
from lib.env import run_dir


def reset_engine(engine, timeout_s: float = 300.0) -> dict:
    """Abort everything in flight and step until the engine is empty.

    **Probes must not inherit each other's backlog.** A diverging probe ends with a deep
    queue; if that queue is still resident when the next probe starts, the next probe
    spends its window draining it and reports a *negative* waiting-list slope -- so an
    unstable rate is scored "stable" because someone else's backlog was leaving, and the
    bisection converges on a number that means nothing.

    Draining naturally is not an option after divergence -- by definition the queue does
    not drain at that rate -- so requests are aborted rather than completed.
    """
    t0 = time.monotonic()
    try:
        sched = engine.engine_core.engine_core.scheduler
        ids = [r.request_id for r in list(sched.running)] + \
              [r.request_id for r in list(sched.waiting)]
        # vLLM v1's LLMEngine.abort_request takes a LIST of ids; passing one id string
        # iterates its characters and aborts nothing. Without a working abort a reset only
        # waits for the requests to finish by themselves -- fine on an H100 inside 60 s,
        # not on a V100 with 1024-token outputs or a 1,000-deep waiting list.
        try:
            engine.abort_request(ids)
        except TypeError:
            for rid in ids:
                try:
                    engine.abort_request(rid)
                except Exception:
                    pass
    except Exception:
        pass
    while engine.has_unfinished_requests() and time.monotonic() - t0 < timeout_s:
        engine.step()
    st = eng.read_state(engine)
    if st.num_running or st.num_waiting:
        # The next probe would inherit this backlog and judge it as its own. Say so
        # loudly; the probe record carries reset_residual_* so the verdict can be
        # discounted afterwards. (A 60 s timeout was not enough on a V100 after a
        # diverging probe left ~200 waiting; it also made the next probe's request
        # ids collide with the aborted ones, which crashed the run.)
        print(f"  WARNING: engine not empty {timeout_s:.0f} s after the reset: "
              f"{st.num_running} running, {st.num_waiting} waiting", flush=True)
    return {"drain_s": time.monotonic() - t0, "residual_running": st.num_running,
            "residual_waiting": st.num_waiting}


_PROBE_SERIAL = 0   # request ids must stay unique across probes: an aborted request
                    # can outlive reset_engine's timeout, and vLLM refuses a duplicate id


def probe_rate(engine, specs, rate: float, seconds: float, warmup_s: float,
               sampler: ps.PowerSampler, rng: random.Random,
               arrival: str, duty: float, burst_s: float) -> dict:
    """Offer ``rate`` req/s for ``seconds`` and decide whether the queue is stable.

    The engine is left empty on exit, so the next probe starts from a clean queue."""
    global _PROBE_SERIAL
    _PROBE_SERIAL += 1
    tag = f"p{_PROBE_SERIAL}q"
    stream = arrivals.build(arrival, rate, seconds, rng, duty=duty, burst_s=burst_s)
    reset_engine(engine)
    live: dict[str, dict] = {}
    waiting_t: list[float] = []
    waiting_v: list[int] = []
    running_v: list[int] = []
    done: list[metrics.RequestRecord] = []
    t0 = time.monotonic()
    i = 0
    seq = 0
    last_probe = 0.0

    while True:
        now = time.monotonic()
        el = now - t0
        if el >= seconds and i >= len(stream.offsets) and not engine.has_unfinished_requests():
            break
        if el >= seconds + 0.5 and i >= len(stream.offsets):
            break
        while i < len(stream.offsets) and stream.offsets[i] <= el:
            spec = specs[seq % len(specs)]
            rid = f"{tag}{seq}"
            seq += 1
            engine.add_request(rid, {"prompt_token_ids": spec.prompt_token_ids},
                               reqlib.sampling_params(spec))
            live[rid] = {"t": now, "P": spec.num_prefill_tokens, "first": None, "n": 0}
            i += 1
        outs = engine.step()
        now = time.monotonic()
        for o in outs:
            rec = live.get(o.request_id)
            if rec is None:
                continue
            n = len(o.outputs[0].token_ids) if o.outputs else 0
            if n > 0 and rec["first"] is None:
                rec["first"] = now
            rec["n"] = max(rec["n"], n)
            if o.finished:
                r = metrics.RequestRecord(o.request_id, 0, "solo", "flexible",
                                          rec["P"], rec["n"], rec["t"])
                r.t_first_token = rec["first"] or now
                r.t_last_token = now
                r.tokens_emitted = rec["n"]
                if el > warmup_s:
                    done.append(r)
                live.pop(o.request_id, None)
        if now - last_probe > 0.1:
            st = eng.read_state(engine)
            if el > warmup_s:
                waiting_t.append(el)
                waiting_v.append(st.num_waiting)
                running_v.append(st.num_running)
            last_probe = now

    # Trend test on the WAITING list, with the slope's own standard error.
    slope = slope_se = float("nan")
    if len(waiting_t) > 10:
        x = np.array(waiting_t)
        y = np.array(waiting_v, float)
        A = np.vstack([np.ones(len(x)), x]).T
        (intercept, slope), *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - (intercept + slope * x)
        dof = max(len(x) - 2, 1)
        sxx = float(np.sum((x - x.mean()) ** 2))
        slope_se = float(np.sqrt(np.sum(resid ** 2) / dof / sxx)) if sxx > 0 else float("nan")
    reset = reset_engine(engine)
    win = (t0 + warmup_s, t0 + seconds)
    pw = sampler.average_over(*win) or {}
    summ = metrics.summarize(done, window_s=max(seconds - warmup_s, 1e-9))
    stable = is_stable(slope, slope_se, rate)
    return {
        "rate_reqs_s": rate, "realized_rate": stream.mean_rate,
        "arrival_cv": stream.index_of_cv(),
        "waiting_slope_per_s": slope, "waiting_slope_se": slope_se,
        "stability_threshold": stability_threshold(rate, slope_se),
        "waiting_mean": float(np.mean(waiting_v)) if waiting_v else 0.0,
        "waiting_final": waiting_v[-1] if waiting_v else 0,
        "running_mean": float(np.mean(running_v)) if running_v else 0.0,
        "stable": stable,
        "avg_power_w": pw.get("avg_power_w"),
        "throughput_reqs_s": summ.get("throughput_reqs_s", 0.0),
        "throughput_decode_tok_s": summ.get("throughput_decode_tok_s", 0.0),
        "ttft_ms_p50": summ["ttft_ms"]["p50"], "ttft_ms_p99": summ["ttft_ms"]["p99"],
        "tbt_ms_p50": summ["tbt_ms"]["p50"], "tbt_ms_p99": summ["tbt_ms"]["p99"],
        "n_completed": summ["n_completed"],
        **{f"reset_{k}": v for k, v in reset.items()},
    }


# Stable if the waiting list is not trending up. Two conditions, because either alone
# misclassifies:
#
#   * **material**: growth must be small relative to the offered rate. 5% of the rate
#     would permit 11 req/s of permanent queue growth at 224 req/s; 1% is the right
#     order: a queue growing at 1% of the arrival rate is diverging, just slowly.
#   * **significant**: growth must exceed its own noise. Without this, a long quiet probe
#     whose waiting list wanders by a couple of requests can be called unstable on a
#     slope that is indistinguishable from zero.
#
# A probe is unstable only when the growth is both real and large enough to matter.
#
# NO ABSOLUTE FLOOR. A floor such as max(0.01 * rate, 0.5) is 1% of the rate only at
# 50 req/s: below that it tolerates MORE than 1%, and below 0.5 req/s it cannot fail at
# all, because a queue cannot grow faster than requests arrive -- an RTX arxiv bisection
# (capacity ~0.2 req/s) would pass every rate up to ~0.7. Noise is what `significant` is
# for; a floor only hides growing queues.
STABILITY_RULE = "waiting slope < max(0.01 * rate, 2 * slope_se); no absolute floor"


def stability_threshold(rate: float, slope_se: float) -> float:
    material = 0.01 * rate
    significant = 2.0 * slope_se if np.isfinite(slope_se) else 0.0
    return float(max(material, significant))


def is_stable(slope: float, slope_se: float, rate: float) -> bool:
    """A NaN slope (too few samples) is never stable."""
    return bool(np.isfinite(slope) and slope < stability_threshold(rate, slope_se))


def monotone_capacity(probes: list[dict]) -> tuple[dict | None, list[float]]:
    """The highest stable probe BELOW the lowest diverging one, and any stable probes above it.

    A capacity claims that every rate up to it held and the next one up did not. Taking
    the highest stable probe regardless would let a lucky verdict above a diverging probe
    become the capacity (e.g. 4.75 diverging, 4.875 stable, 4.9375 diverging). The stray
    stable probes are returned so the record says the bracket was not clean.
    """
    diverged = [p["rate_reqs_s"] for p in probes if not p["stable"]]
    first_bad = min(diverged) if diverged else float("inf")
    below = [p for p in probes if p["stable"] and p["rate_reqs_s"] < first_bad]
    stray = sorted(p["rate_reqs_s"] for p in probes
                   if p["stable"] and p["rate_reqs_s"] > first_bad)
    return (max(below, key=lambda p: p["rate_reqs_s"]) if below else None), stray


def find_bracket(probe, start: float, *, min_rate: float,
                 max_rate: float = 1e5) -> tuple[float, float]:
    """Find ``(lo, hi)`` with ``probe(lo)`` stable and ``probe(hi)`` diverging.

    ``probe(rate) -> dict`` runs one probe and returns its record; only ``["stable"]`` is
    read here, so this is a pure function of the stability oracle and can be tested
    without a GPU.

    Searches in BOTH directions from ``start``: up (doubling) while the probe is stable,
    and down (halving) when the very first probe already diverges, so a capacity below
    ``start`` is bracketed rather than reported as ``start``.

    Refuses rather than guesses: if nothing is stable down to ``min_rate`` it raises, since
    a reported floor is a number where there should be none.
    """
    r = float(start)
    p = probe(r)
    if p["stable"]:
        lo = r                                   # search UP for the divergence
        while True:
            r *= 2
            if r > max_rate:
                return lo, r
            if not probe(r)["stable"]:
                return lo, r
            lo = r
    hi = r                                       # search DOWN for a stable rate
    while True:
        r /= 2
        if r < min_rate:
            raise RuntimeError(
                f"no stable rate found down to {min_rate} req/s (started at {start}). "
                f"Refusing to report a capacity: {start} would be a floor, not a measurement.")
        if probe(r)["stable"]:
            return r, hi
        hi = r


def _selfcheck() -> None:
    """The bracket must contain the true capacity whichever side of it the search starts."""
    for cap, start in ((10.0, 1.0),     # capacity above start: search up
                       (0.14, 1.0),     # capacity BELOW start: search down
                       (0.9, 1.0),      # just below start
                       (1.0, 1.0)):     # exactly at start (a probe AT cap diverges)
        seen = []
        lo, hi = find_bracket(lambda r: (seen.append(r) or {"stable": r < cap}),
                              start, min_rate=1e-3)
        assert lo < cap <= hi, (cap, start, lo, hi)
        assert hi / lo == 2.0, (cap, start, lo, hi)     # a real, non-collapsed bracket
        assert len(set(seen)) == len(seen), f"re-probed a rate: {seen}"
    # nothing is stable at all -> refuse, never report the floor
    try:
        find_bracket(lambda r: {"stable": False}, 1.0, min_rate=0.01)
    except RuntimeError:
        pass
    else:
        raise AssertionError("an all-diverging engine must raise, not return a floor")
    # The stability rule, on RECORDED probes (rate, slope, se) -> verdict. The first four
    # grow well above their noise, each by less than a 0.5/s floor would tolerate.
    for rate, slope, se, want in (
            (3.375, 0.4009, 0.0087, False),     # RTX qwen3-8b@64
            (4.9375, 0.3546, 0.0163, False),    # RTX qwen3-4b@128
            (4.75, 0.1302, 0.0106, False),      # RTX qwen3-4b@128
            (22.0, 0.4184, 0.0285, False),      # H100 qwen3-8b@64 sharegpt
            (3.0, 0.0176, 0.0032, True),        # 0.6% of the rate: below the 1% line
            (4.5, 0.0, 0.0, True),              # waiting list never moved
            (5.0, 0.02, 0.03, True),            # synthetic: a wander inside its noise
            (0.25, 0.05, 0.005, False),         # synthetic RTX arxiv: a 0.5/s floor would pass it
            (224.0, 11.0, 0.5, False),          # 5% of the rate would pass it
            (1.0, float("nan"), float("nan"), False)):   # too few samples
        got = is_stable(slope, se, rate)
        assert got is want, (rate, slope, se, got)
    # The capacity is the highest stable rate BELOW the lowest diverging one.
    recs = [{"rate_reqs_s": r, "stable": s}
            for r, s in ((4.5, True), (4.75, False), (4.875, True), (4.9375, False))]
    best, stray = monotone_capacity(recs)
    assert best["rate_reqs_s"] == 4.5 and stray == [4.875], (best, stray)
    best, stray = monotone_capacity([{"rate_reqs_s": 1.0, "stable": False}])
    assert best is None and stray == [], (best, stray)
    print("capacity_bisection selfcheck OK: bracket contains capacity from above and below start, "
          "refuses rather than reporting a floor, judges 10 recorded/synthetic probes "
          "without an absolute floor, and takes the capacity below the lowest divergence")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", default="sharegpt")
    ap.add_argument("--arrival", choices=("poisson", "bursty", "gamma"), default="poisson")
    ap.add_argument("--burst-duty", type=float, default=0.3)
    ap.add_argument("--burst-s", type=float, default=5.0)
    ap.add_argument("--lo", type=float, default=1.0)
    ap.add_argument("--hi", type=float, default=None, help="default: from a probe at --lo")
    ap.add_argument("--min-rate", type=float, default=0.01,
                    help="lowest rate the downward search will try before refusing. The "
                         "search goes DOWN from --lo when the first probe already "
                         "diverges.")
    ap.add_argument("--selfcheck", action="store_true",
                    help="test the bracket search against a synthetic oracle and exit")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--probe-s", type=float, default=90.0)
    ap.add_argument("--warmup-s", type=float, default=25.0)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    # The chunk budget, explicitly. Omitted means "whatever this vLLM defaults to", and
    # that is not recoverable afterwards: 0.11.0 exposes
    # vllm.utils.DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048, while 0.29.0 does not export it
    # at all. A capacity measured on a long-prefill workload depends on it, because the
    # budget clamps how much prefill a step can carry. Harmless on sharegpt (mean prefill
    # 87 is one chunk under any budget) and load-bearing on arxiv.
    ap.add_argument("--max-num-batched-tokens", type=int, default=None)
    ap.add_argument("--pool-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
        return

    spec = models.get(args.model)
    # Before run_dir: a multi-GPU instance writes under its own namespace (h100-80gb-tp4).
    hardware.set_instance(spec.tensor_parallel_size)
    wl = workloads.get(args.workload)
    max_seqs = args.max_num_seqs or spec.concurrency_grid[-1]
    if args.smoke:
        args.probe_s, args.warmup_s, args.iters, args.pool_size = 20.0, 6.0, 2, 512

    out = run_dir("capacity_bisection", args.out_tag or f"{spec.slug}__{wl.name}__{args.arrival}")
    print(f"capacity_bisection: capacity of {spec.model_id} on {wl.label()} ({args.arrival} arrivals)")
    print(f"  out {out}")

    sampler = ps.for_model(spec)  # its own card, or every card of a multi-GPU instance
    sampler.start()
    dev = sampler.device_info()  # captured now; the sampler is shut down before the result is assembled
    print(f"  gpu {dev['name']} (nvml index {dev['device_index']}, {dev['pci_bus_id']})")
    if sampler.num_devices > 1:
        print(f"  instance: TP={spec.tensor_parallel_size} over NVML {dev['device_indices']} "
              f"({', '.join(dev['pci_bus_ids'])}); power is the sum", flush=True)
    engine = eng.build_engine(spec, max_num_seqs=max_seqs,
                              max_num_batched_tokens=args.max_num_batched_tokens)
    if spec.tensor_parallel_size > 1:
        dev["mem_used_mib_after_load"] = ps.assert_instance_resident(sampler, spec)
    vocab, special = eng.tokenizer_info(engine)
    rng = random.Random(args.seed)
    specs = reqlib.pregenerate(wl, args.pool_size, vocab, rng, special)
    kv = eng.kv_sizing(engine)
    print(f"  kv {kv['kv_pool_tokens']:,} tokens, max_num_seqs {max_seqs}")

    probes: list[dict] = []
    try:
        lo, hi = args.lo, args.hi

        def probe(r: float) -> dict:
            p = probe_rate(engine, specs, r, args.probe_s, args.warmup_s, sampler,
                           rng, args.arrival, args.burst_duty, args.burst_s)
            probes.append(p)
            print(f"  probe {r:8.3f} req/s -> {'stable ' if p['stable'] else 'DIVERGE'} "
                  f"slope {p['waiting_slope_per_s']:+8.2f}+/-{p['waiting_slope_se']:5.2f}/s "
                  f"(thr {p['stability_threshold']:5.2f})  wait~{p['waiting_mean']:7.1f}  "
                  f"P {p['avg_power_w'] or float('nan'):6.1f} W  "
                  f"tok/s {p['throughput_decode_tok_s']:8.1f}", flush=True)
            return p

        if hi is None:
            lo, hi = find_bracket(probe, args.lo, min_rate=args.min_rate)
        for _ in range(args.iters):
            mid = 0.5 * (lo + hi)
            p = probe(mid)
            if p["stable"]:
                lo = mid
            else:
                hi = mid
    finally:
        sampler.stop()
        sampler.shutdown()
        eng.teardown(engine)

    best, stray = monotone_capacity(probes)
    if stray:
        print(f"  WARNING: stable verdicts ABOVE the lowest diverging rate at {stray} -- "
              f"ignored; the capacity is the highest stable rate below it.", flush=True)
    result = {
        "model": spec.model_id, "model_slug": spec.slug,
        "workload": wl.descriptor(), "arrival": args.arrival,
        "max_num_seqs": max_seqs, "kv": kv,
        # Recorded so a capacity can be told apart from one measured under a different
        # budget. None means the engine's own default, which differs across vLLM versions.
        "max_num_batched_tokens": args.max_num_batched_tokens,
        # Provenance: capacity is a property of a model *on a card*, so the card
        # travels with the number. Never compare across different "hardware".
        "hardware": hardware.current(),
        "dtype": hardware.resolve_dtype(spec.dtype),
        "device": dev,
        "capacity_reqs_s": best["rate_reqs_s"] if best else None,
        "capacity_decode_tok_s": best["throughput_decode_tok_s"] if best else None,
        "power_at_capacity_w": best["avg_power_w"] if best else None,
        "bracket": {"lo": lo, "hi": hi},
        # Which rule judged these probes. Files without this key were judged WITH the
        # 0.5/s floor, and their capacity can be re-derived from the probes they record.
        "stability_rule": STABILITY_RULE,
        "stray_stable_probes": stray,
        "probe_s": args.probe_s, "warmup_s": args.warmup_s,
        "probes": probes,
        "caveat": ("in-process engine core (VLLM_ENABLE_V1_MULTIPROCESSING=0) costs "
                   "~1.24x capacity vs the out-of-process default; applied uniformly"),
    }
    if spec.tensor_parallel_size > 1:
        result["tensor_parallel_size"] = spec.tensor_parallel_size
        result["power_scope"] = f"sum over the instance's {sampler.num_devices} GPUs"
    with open(out / "capacity.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\ncapacity {result['capacity_reqs_s']} req/s "
          f"({result['capacity_decode_tok_s']} decode tok/s) -> {out}/capacity.json")


if __name__ == "__main__":
    main()
