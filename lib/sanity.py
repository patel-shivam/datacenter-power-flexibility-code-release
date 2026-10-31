"""Physical-plausibility checks that run at the END of every fleet experiment.

Why this file exists
--------------------
NVML indices ignore ``CUDA_VISIBLE_DEVICES``, so a replica that samples NVML index 0
reads physical GPU 0 whatever card it is pinned to, and a fleet of them reports pool
power as N copies of one card. Nothing downstream notices: the traces stay smooth, the
numbers plausible and the fits good, while conclusions come out backwards -- packing
appears to cost power it actually saves, and knob authority appears to collapse at high
load where it holds.

The fault shows up as an arithmetic impossibility -- 1992.7 W reported while exactly one
replica is lit (the case ``_selfcheck`` replays), and one H100 cannot draw 1993 W. That
check costs nothing, so it runs on every run, automatically, and the verdict is written
into the summary instead of being left for someone to notice later.

The rule these encode
---------------------
*Measure the same quantity a second, independent way, and treat disagreement as the
signal.* A hardware limit is the cheapest independent measurement available -- it needs
no second instrument, only arithmetic. Telemetry faults of this kind produce numbers
that are internally consistent and survive review; inspection does not catch them.

These are deliberately **necessary, not sufficient**. Passing does not mean the
telemetry is right; failing means it is definitely wrong.
"""

from __future__ import annotations

# NVIDIA H100-80 SXM. A card can transiently read a little over its limit, so the
# tolerance below is what separates "measurement noise" from "arithmetically impossible".
H100_TDP_W = 700.0
TOLERANCE = 1.05


def _lit_count(row: dict, idle_w: float | None = None) -> int | None:
    """Replicas drawing load, across BOTH pools.

    Counting only ``active_large`` charges the small replica's power to the lit large
    ones and cries wolf: on a 3-large + 1-small fleet whose samplers are provably
    distinct it can trip the per-replica test (748 W against a 735 W limit). A guard
    that false-positives is worse than none, because it teaches everyone to ignore it.

    ``active_*`` IS THE INSTANTANEOUS QUEUE, AND A FAST REPLICA EMPTIES IT BETWEEN TICKS.
    That is the same false positive arriving by a second route: on arxiv at f=0.10 the
    1B retires its handful of requests well inside the 1 s control tick, so
    ``active_small`` reads 0 on 55 of 1613 ticks while ``power_small_w`` averages 186 W
    against a 119 W idle -- it was busy during the NVML window and empty at the instant
    it was read. The count then under-reports, its watts are charged to the large pool,
    and a run whose telemetry is fine can be stamped IMPLAUSIBLE POWER -- DO NOT USE IT
    over a single tick.

    So when pool power is available, a pool drawing meaningfully above idle counts as lit
    whatever its queue says. Power is the thing being checked for plausibility; deciding
    the denominator from a *different* signal that can disagree with it produces false
    alarms.
    """
    if row.get("active_large") is None and row.get("active_small") is None:
        return None
    try:
        n_l = int(float(row.get("active_large") or 0))
        n_s = int(float(row.get("active_small") or 0))
    except (TypeError, ValueError):
        return None
    if idle_w:
        for key, n in (("power_large_w", n_l), ("power_small_w", n_s)):
            try:
                p = float(row.get(key) or 0)
            except (TypeError, ValueError):
                continue
            # 1.25x idle: comfortably above sampler noise, far below a busy replica.
            if n == 0 and p > idle_w * 1.25:
                if key == "power_large_w":
                    n_l = 1
                else:
                    n_s = 1
    return n_l + n_s


def check_power_plausibility(rows: list[dict], n_gpus: int, *,
                             card_tdp_w: float = H100_TDP_W,
                             idle_w: float = 110.0) -> dict:
    """Can the reported pool power have come from this many cards?

    .. warning::
       ``card_tdp_w`` and ``idle_w`` default to **H100-80** values (700 W / 110 W) and
       no caller overrides them. On a lower-TDP card this check goes INERT rather than
       wrong: a 260 W RTX 8000 fleet would be allowed n x 735 W instead of n x 273 W, so
       the N-copies-of-one-card signature it exists to catch could not trip until power
       was 2.7x impossible. That matters because a SLURM cluster has cgroup isolation
       and a shared node such as the RTX 8000 does not -- the one machine where the
       failure is still reachable is the one where the detector is asleep.

       Rather than silently scale a safety threshold from a value this code cannot
       verify without a GPU in hand, the mismatch is SHOUTED below and the caller is
       told to pass the real limit from ``PowerSampler.device_info()['power_limit_w']``,
       which already reads it from NVML.

    Two tests, both pure arithmetic on data the run already records:

    **Fleet ceiling.** Total power can never exceed ``n_gpus * TDP``. Exceeding it means
    the sampler is reading more cards than the fleet has -- the N-copies-of-one-card
    signature.

    **Per-lit-replica ceiling.** Charging the dark replicas at idle and splitting the
    rest over the lit ones cannot yield more than one card's TDP each. This is the
    sharper test: it fires on N-copies-of-one-card data while the fleet total still
    looks fine (see ``_selfcheck``), because a lit replica is a much smaller denominator.

    Ticks whose lit set is changing are excluded from the per-replica test only: power
    is averaged over the tick while occupancy is instantaneous, and a replica that just
    went empty takes ~1 s to fall, so a transition tick over-charges the lit ones for
    reasons that are not a defect.
    """
    # Refuse to be silently inert on hardware these defaults do not describe.
    if card_tdp_w == H100_TDP_W:
        from lib import hardware
        if not hardware.is_baseline():
            print("")
            print("  " + "!" * 76)
            print(f"  PLAUSIBILITY CHECK IS USING H100 LIMITS ON {hardware.slug()}")
            print(f"    card_tdp_w={card_tdp_w:.0f} W and idle_w={idle_w:.0f} W are "
                  f"H100-80 values. On this card the")
            print("    fleet ceiling is too generous, so the N-copies-of-one-card check "
                  "cannot fire.")
            print("    Pass card_tdp_w=PowerSampler.device_info()['power_limit_w'] and "
                  "the measured")
            print("    warm idle. Until then this verdict means nothing on this "
                  "hardware.")
            print("  " + "!" * 76)
            print("")

    fleet_cap = n_gpus * card_tdp_w * TOLERANCE
    worst_total = 0.0
    worst_per_lit = 0.0
    n_total_bad = 0
    n_per_lit_bad = 0
    checked = 0

    prev_lit = None
    for i, r in enumerate(rows):
        p = r.get("measured_w")
        if p is None:
            continue
        try:
            p = float(p)
        except (TypeError, ValueError):
            continue
        if p <= 0:
            continue
        checked += 1
        worst_total = max(worst_total, p)
        if p > fleet_cap:
            n_total_bad += 1

        lit = _lit_count(r, idle_w)
        if lit is None:
            continue
        nxt = _lit_count(rows[i + 1], idle_w) if i + 1 < len(rows) else lit
        stable = (prev_lit is None or prev_lit == lit) and (nxt is None or nxt == lit)
        prev_lit = lit
        if not stable or lit <= 0:
            continue
        per_lit = (p - idle_w * max(n_gpus - lit, 0)) / lit
        worst_per_lit = max(worst_per_lit, per_lit)
        if per_lit > card_tdp_w * TOLERANCE:
            n_per_lit_bad += 1

    ok = (n_total_bad == 0 and n_per_lit_bad == 0)
    return {
        "ok": ok, "n_ticks_checked": checked,
        "fleet_cap_w": fleet_cap, "worst_total_w": round(worst_total, 1),
        "n_ticks_over_fleet_cap": n_total_bad,
        "card_cap_w": card_tdp_w * TOLERANCE,
        "worst_per_lit_replica_w": round(worst_per_lit, 1),
        "n_ticks_over_card_cap": n_per_lit_bad,
    }


def check_queue_stable(rows: list[dict], *, growth_tol: int = 50,
                       deep_tol: int = 200) -> dict:
    """Is the run actually AT the utilization on its label, or over capacity?

    Why this exists
    ---------------
    ``waiting_total`` is written to every trace row, and ``steady()`` even reports its
    mean. That is not enough, and the reason is worth stating: **the mean of a growing
    series is a perfectly ordinary-looking number.** A run whose queue climbs 0 -> 40000
    reports a mean of 20000, which reads as "a big queue" rather than "this run
    diverged". Growth is the signal, so this computes it.

    The utilization on the label comes from an ANALYTIC capacity
    (``fleet_driver.fleet_capacity``): the step-time line gives tok/s at the concurrency
    cap, divided by mean decode length to get req/s, summed over both pools. That
    estimate has three optimistic assumptions -- it charges nothing for prefill work,
    evaluates each replica at its cap (the maximum-throughput point, not a sustainable
    one), and counts both pools as if traffic were split to match their capacities when
    the routing knob decides the split. Measured against the queue it runs 1.7-2x high.

    So the label cannot be trusted and the queue must be checked directly. This is the
    module's rule again -- measure the same quantity a second, independent way -- with
    offered load as the quantity and the scheduler's waiting list as the second
    measurement.

    Two failures, because they are different
    ----------------------------------------
    **Growing.** Last quarter above first quarter by more than ``growth_tol``. The fleet
    is not keeping up; the run is over capacity whatever its label says.

    **Deep but flat.** Stable, but parked above ``deep_tol``. Not diverging, yet not a
    healthy operating point either: the queue is a standing backlog, latency is set by it
    rather than by service time, and the fleet is at capacity rather than below it.
    """
    w = []
    for r in rows:
        if "calib" in (r.get("phase") or ""):
            continue                      # calibration drains on purpose
        v = r.get("waiting_total")
        if v is None:
            continue
        try:
            w.append(float(v))
        except (TypeError, ValueError):
            continue
    if len(w) < 20:
        return {"ok": True, "checked": False,
                "reason": "no usable waiting_total column"}
    n = len(w)
    first = sum(w[:n // 4]) / max(n // 4, 1)
    last = sum(w[-(n // 4):]) / max(n // 4, 1)
    growth = last - first
    peak = max(w)
    growing = growth > growth_tol
    deep = (not growing) and last > deep_tol
    return {
        "ok": not growing, "checked": True, "growing": growing, "deep_but_flat": deep,
        "q_first": round(first, 1), "q_last": round(last, 1),
        "q_growth": round(growth, 1), "q_peak": round(peak, 1),
        "growth_tol": growth_tol, "deep_tol": deep_tol,
    }


def report_queue(v: dict) -> None:
    """Print the queue verdict. Growth invalidates the run's utilization label."""
    if not v.get("checked"):
        print(f"  queue: not checked ({v.get('reason')})")
        return
    if v["ok"] and not v.get("deep_but_flat"):
        print(f"  queue: stable ({v['q_first']:.0f} -> {v['q_last']:.0f} waiting, "
              f"peak {v['q_peak']:.0f}) -- the fleet is keeping up")
        return
    if v.get("deep_but_flat"):
        print(f"  queue: STABLE BUT DEEP -- {v['q_last']:.0f} requests waiting "
              f"(peak {v['q_peak']:.0f}). Not diverging, but latency is set by this "
              f"backlog and the fleet is AT capacity, not below it.")
        return
    print("")
    print("  " + "!" * 76)
    print("  QUEUE IS GROWING -- THIS RUN IS OVER CAPACITY, ITS UTILIZATION LABEL IS WRONG")
    print(f"    waiting {v['q_first']:.0f} -> {v['q_last']:.0f} "
          f"(+{v['q_growth']:.0f}), peak {v['q_peak']:.0f}")
    print("    The --utils fraction comes from an analytic capacity that charges nothing")
    print("    for prefill, evaluates each replica at its concurrency cap, and assumes")
    print("    both pools are loaded in proportion to their capacity. Measured against")
    print("    the queue it runs 1.7-2x high. Quote the offered req/s, not the fraction,")
    print("    and do not compare this run against one that was not diverging.")
    print("  " + "!" * 76)
    print("")


def report(verdict: dict, n_gpus: int) -> None:
    """Print the verdict. A failure is shouted, because it invalidates the whole run."""
    if verdict.get("ok"):
        print(f"  sanity: power plausible over {verdict['n_ticks_checked']} ticks "
              f"(peak {verdict['worst_total_w']:.0f} W of {verdict['fleet_cap_w']:.0f} W "
              f"fleet cap, peak {verdict['worst_per_lit_replica_w']:.0f} W per lit "
              f"replica)")
        return
    print("")
    print("  " + "!" * 76)
    print("  IMPLAUSIBLE POWER -- THIS RUN'S TELEMETRY IS WRONG, DO NOT USE IT")
    if verdict["n_ticks_over_fleet_cap"]:
        print(f"    {verdict['n_ticks_over_fleet_cap']} ticks report more than "
              f"{n_gpus} cards can draw (peak {verdict['worst_total_w']:.0f} W > "
              f"{verdict['fleet_cap_w']:.0f} W)")
    if verdict["n_ticks_over_card_cap"]:
        print(f"    {verdict['n_ticks_over_card_cap']} ticks imply "
              f"{verdict['worst_per_lit_replica_w']:.0f} W from a single lit replica, "
              f"above the {verdict['card_cap_w']:.0f} W card limit")
    print("    Most likely cause: replicas sampling the same GPU. NVML indices ignore")
    print("    CUDA_VISIBLE_DEVICES -- check the 'power samplers:' line at startup, which")
    print("    shows the PCI bus id each replica sampled.")
    print("  " + "!" * 76)
    print("")


def _selfcheck() -> None:
    """Both tests must fire on the N-copies-of-one-card signature and stay quiet on
    correct telemetry."""
    # The N-copies signature: 4 x GPU 0, one replica lit, 1992.7 W.
    bad = [{"measured_w": 1992.7, "active_large": 1} for _ in range(5)]
    v = check_power_plausibility(bad, n_gpus=4)
    assert not v["ok"], v
    assert v["n_ticks_over_card_cap"] > 0, v
    assert v["n_ticks_over_fleet_cap"] == 0, "4x700 is 2800 W, so the fleet cap alone " \
                                             "does NOT catch this -- the per-replica " \
                                             "test is the one that matters"

    # Correct telemetry at the same operating point: 822 W, one card lit.
    good = [{"measured_w": 822.0, "active_large": 1} for _ in range(5)]
    assert check_power_plausibility(good, n_gpus=4)["ok"]

    # A fleet genuinely near its ceiling is fine.
    hot = [{"measured_w": 2100.0, "active_large": 3} for _ in range(5)]
    assert check_power_plausibility(hot, n_gpus=4)["ok"]

    # Absurd totals are caught even without occupancy.
    absurd = [{"measured_w": 5000.0} for _ in range(3)]
    assert not check_power_plausibility(absurd, n_gpus=4)["ok"]

    # A transition tick must not raise a false alarm.
    trans = [{"measured_w": 822.0, "active_large": 2},
             {"measured_w": 1900.0, "active_large": 1},
             {"measured_w": 822.0, "active_large": 2}]
    assert check_power_plausibility(trans, n_gpus=4)["ok"], \
        "lit-set transitions are excluded from the per-replica test"

    # REGRESSION: a mixed fleet with one large and the small replica both lit.
    # Counting only active_large would charge the small pool's power to that one large
    # replica and trip the limit on correctly-sampled data.
    mixed = [{"measured_w": 1080.0, "active_large": 1, "active_small": 1}
             for _ in range(5)]
    v = check_power_plausibility(mixed, n_gpus=4)
    assert v["ok"], f"3L+1S with 2 lit must pass: {v}"
    assert v["worst_per_lit_replica_w"] < 500, v
    # ...while the same total from a SINGLE lit replica is still impossible.
    alone = [{"measured_w": 1080.0, "active_large": 1, "active_small": 0}
             for _ in range(5)]
    assert not check_power_plausibility(alone, n_gpus=4)["ok"], \
        "one lit replica cannot draw 1080 W minus three idles"

    # --- queue stability -------------------------------------------------------
    # A queue that grows is over capacity whatever the label says.
    grow = [{"waiting_total": 100.0 * i} for i in range(40)]
    v = check_queue_stable(grow)
    assert v["checked"] and not v["ok"] and v["growing"], v

    # The mean ALONE cannot catch it -- this is why averaging waiting_total is not
    # enough. Same mean, one diverging and one flat.
    import statistics as _st
    flat_hi = [{"waiting_total": 1950.0} for _ in range(40)]
    assert abs(_st.mean(x["waiting_total"] for x in grow)
               - _st.mean(x["waiting_total"] for x in flat_hi)) < 1.0
    assert check_queue_stable(flat_hi)["ok"], "flat is not diverging"
    assert check_queue_stable(flat_hi)["deep_but_flat"], "...but 1950 deep is not healthy"

    # A healthy queue, as in a u=0.5 sharegpt run: 0-2 waiting.
    good_q = [{"waiting_total": float(x)} for x in ([0, 1, 2] * 14)]
    v = check_queue_stable(good_q)
    assert v["ok"] and not v["deep_but_flat"], v

    # Calibration drains on purpose and must not be counted.
    calib = ([{"waiting_total": 9e4, "phase": "calib_f0"} for _ in range(30)]
             + [{"waiting_total": 1.0, "phase": "ctrl"} for _ in range(30)])
    assert check_queue_stable(calib)["ok"], "calib rows must be excluded"

    print("sanity: selfcheck ok -- fires on the 4x-one-card signature, quiet on correct "
          "telemetry;")
    print("        queue check fires on growth that the MEAN alone cannot distinguish")


if __name__ == "__main__":
    _selfcheck()
