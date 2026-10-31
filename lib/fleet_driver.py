"""Shared driver for every fleet experiment: arrivals in, routing, telemetry out.

The fleet experiments are the same experiment with different target schedules and
different things swept, so they share one loop rather than a copy each. The loop is the
part where mistakes are expensive (a submitter that starves the engine it drives;
telemetry whose cost grows with elapsed time), so there should be exactly one of it.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import random
import time

from lib import arrivals
from lib import fleet as fleetlib
from lib.router import build_router

# Chunked-prefill token budget per scheduler step: prefill chunks and decode
# continuations share it.
#
# Left at None (the default), vLLM 0.11.0 resolves it to its own
# DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048 (config/scheduler.py __post_init__, then
# min(max_num_seqs*max_model_len, 2048)). It is settable and, more to the point, written
# into each run's summary: a scheduler budget that shapes every batch belongs in every
# result file.
#
# It is not a cosmetic knob. Measured at concurrency 64 on llama31-8b, the ratio of mean
# step time WITH a prefill chunk to WITHOUT it -- the "lump" the power sampler sees:
#
#            budget    512    1024    2048*    4096
#   arxiv             2.52x   4.04x   5.75x    8.44x     (* vLLM's default)
#   sharegpt          1.22x   1.26x   1.29x    1.27x
#
# At 2048 an arxiv prefill step runs 49.6 ms against 8.6 ms for a decode-only step. That
# 4.5x gap in lumpiness against sharegpt is the same size as the 4x gap in within-rung
# power excursion (162 W vs 41 W) -- the noise is below the controller, in the
# scheduler. Dropping to 512 cost <=3% decode throughput at fixed concurrency and was
# faster on two of six (model, workload) pairs; engine.py's warning about starving
# prefill admission is about the OPEN-LOOP high-arrival-rate regime and is untested here.
CHUNK_BUDGET = (int(os.environ["FLEET_MAX_NUM_BATCHED_TOKENS"])
                if os.environ.get("FLEET_MAX_NUM_BATCHED_TOKENS") else None)
VLLM_DEFAULT_CHUNK_BUDGET = 2048          # vllm.utils.DEFAULT_MAX_NUM_BATCHED_TOKENS

# The chunk size the residence model charges prefill at. It must be what the ENGINE
# actually uses, not a constant that can drift from it. The error from a mismatch is
# small -- A = E[P]*(t0/CHUNK + t1) + E[D]*t0 is dominated by the decode term, so 512 vs
# 2048 moves A by at most 1.0% (arxiv) and 0.0% on sharegpt -- but a constant that
# disagrees with the plant is a trap regardless of its current size.
CHUNK_SIZE = CHUNK_BUDGET or VLLM_DEFAULT_CHUNK_BUDGET

# A KNOWN APPROXIMATION, KEPT DELIBERATELY.
#
# `cap_for` below computes kappa = t0/CHUNK_SIZE + t1 and A = E[P]*kappa + E[D]*t0. That
# charges every prefill TOKEN a full t1, where t1 is the marginal step cost of one more
# concurrent SEQUENCE. The offline scorer (analyze/score_fleet_model.py) charges prefill
# per chunk instead: on arxiv the spurious E[P]*t1 term is 68.7% of A and drives
# predicted occupancy 2.2x high.
#
# It is not corrected here because A feeds `_update_active_targets`, which sizes
# `spread`'s active replica set every control tick: changing it would make runs on either
# side of the change route differently while being presented as one controlled
# comparison. The offline scorers change no hardware, so they carry the correction:
#     n_chunks = workload.mean_prefill / CHUNK_SIZE
#     A = n_chunks * t0 + workload.mean_decode * t0
#     B = (n_chunks + workload.mean_decode) * t1


def descending_branch(p_at_f: dict | None) -> list[tuple[float, float]]:
    """``(power, f)`` of a measured ``{f: power}`` map at and past its peak, sorted by power.

    The feedforward inverse (``FleetRunner.set_power_map``) searches only these: P(f) rises
    before it falls, and a target whose power matches a point on the rising side would
    otherwise send the knob the wrong way (see set_power_map)."""
    pts = [(float(v), float(k)) for k, v in (p_at_f or {}).items()
           if v is not None and v == v]
    if not pts:
        return []
    f_peak = max(pts)[1]                    # the f of the highest-power point
    return sorted(pv for pv in pts if pv[1] >= f_peak - 1e-12)


def _selfcheck() -> None:
    # A measured RTX 8000 map: P(0) = 1140 W on the rising branch, peak at 0.4
    m = {"0.0": 1140.0, "0.1": 1377.7, "0.2": 1363.8, "0.3": 1266.3, "0.4": 1384.5,
         "0.5": 1226.1, "0.6": 1195.0, "0.7": 1202.0, "0.8": 1036.0, "0.9": 1036.4,
         "0.9862": 1019.1}
    br = descending_branch(m)
    assert min(f for _, f in br) == 0.4 and len(br) == 7, br
    inv = lambda w: min(br, key=lambda pv: abs(pv[0] - w))[1]
    assert inv(1129.0) == 0.6, inv(1129.0)       # over the whole map it would be f = 0
    assert inv(1056.0) in (0.8, 0.9), inv(1056.0)  # the same rung (1 + 3 lit), 0.4 W apart
    assert descending_branch({}) == [] and descending_branch(None) == []
    print("fleet_driver selfcheck ok: the feedforward inverse stays on the descending branch")


class FleetRunner:
    """Owns the fleet, the router, and the arrival loop."""

    def __init__(self, cfgs, *, n_large: int, n_small: int, pack_cap_large: int,
                 pack_cap_small: int, policy: str, split, workload, offered_reqs_s: float,
                 arrival: str = "poisson", burst_duty: float = 0.3, burst_s: float = 5.0,
                 dt: float = 1.0, pool_size: int = 4096, seed: int = 0,
                 residence: dict | None = None):
        self.fleet = fleetlib.Fleet(cfgs)
        self.router = build_router(n_large, n_small, pack_cap_large, pack_cap_small,
                                   policy=policy)
        self.split = split
        self.workload = workload
        self.lam = offered_reqs_s
        self.arrival = arrival
        self.burst_duty, self.burst_s = burst_duty, burst_s
        self.dt = dt
        self.pool_size = pool_size
        self.rng = random.Random(seed)
        self.trace: list[dict] = []
        self.policy = policy
        # {role: {"A": s, "B": s}} from the measured step-time line. Used only by the
        # spread policy, to choose how many replicas to light from the OFFERED LOAD --
        # sizing that from instantaneous occupancy makes spreading collapse into packing
        # during fill-up (see Router.set_active_target).
        self.residence = residence or {}
        self.caps = {"large": pack_cap_large, "small": pack_cap_small}
        # Inverse of the measured P(f) map, set by set_power_map() from a calibration
        # sweep. None (the default) disables setpoint feedforward.
        self._f_of_power = None
        # The power the map measured at the point the inverse picked (same nearest point).
        self._level_of_power = None

    # --- lifecycle --------------------------------------------------------------
    def start(self):
        self.fleet.start()
        self.kv_ceilings = self._report_kv_ceiling()
        return self

    def _report_kv_ceiling(self) -> list[dict]:
        """Say up front whether each replica's concurrency cap is actually reachable.

        The cap is what we ask for; the KV pool decides what we get. At this workload's
        context lengths the pool can bind first, and when it does the consequences are
        silent and not local to the batch size: the router packs to ``pack_cap`` before
        spilling, so a cap that cannot be reached means first-fit never spills on time,
        and the power model's occupancy vectors describe a fill level that never occurs.
        """
        wl = self.workload
        try:
            ctx = float(wl.mean_prefill) + float(wl.mean_decode) / 2.0
        except Exception:
            return []
        rows = self.fleet.kv_concurrency_ceiling(ctx)
        if not rows:
            return []
        print(f"  KV headroom at mean context ~{ctx:.0f} tok "
              f"(prefill {wl.mean_prefill:.0f} + half of decode {wl.mean_decode:.0f}):")
        for r in rows:
            if r.get("kv_ceiling") is None:
                print(f"    replica {r['replica']} [{r['role']}] cap {r['cap']}: "
                      f"KV pool unknown -- cannot tell which ceiling binds")
                continue
            mark = "  <-- KV BINDS FIRST" if r["binds"] == "kv" else ""
            print(f"    replica {r['replica']} [{r['role']}] cap {r['cap']}, "
                  f"KV holds ~{r['kv_ceiling']:.0f} seqs of this size{mark}")
        if any(r.get("binds") == "kv" for r in rows):
            print("    NOTE: pack_cap targets a fill level KV cannot reach on the marked "
                  "replicas -- first-fit will not spill when the model assumes it does, "
                  "and predicted occupancy vectors will be wrong. Compare b_max in the "
                  "trace against the cap before reading any packing result.")
        return rows

    def stop(self):
        self.fleet.stop()

    def set_power_map(self, p_at_f: dict | None) -> None:
        """Enable setpoint feedforward from a measured ``{f: power}`` map.

        ``calibrate_band`` already produces exactly this, so the map costs nothing extra:
        it is the sweep the band came from. The inverse is by nearest measured point
        rather than by interpolation, because P(f) is NOT monotonic -- it rises from
        1636 to 1781 W as f goes 0 to 0.1, since the small replica lights before the large
        pool sheds anything -- and interpolating across that fold would invent an f that
        produces the wrong power. Nearest-point cannot.

        Only over the DESCENDING BRANCH, at and past the peak of P(f) (``descending_branch``).
        Over the whole map, nearest-point picks the rising branch whenever a target's power
        happens to match P(0) -- on the RTX 8000 map in ``_selfcheck``, a 1129 W drop target
        is nearest P(0) = 1140 W -- and sends the knob to f = 0, lighting every large
        replica: the opposite of what a drop asks for. A wrong pick is not one step of error
        the controller then removes; a slow integral (ki 0.003) never undoes the jump. A drop
        moves f up the descending branch, so that is the only branch its inverse may land on.
        """
        if not p_at_f:
            self._f_of_power = self._level_of_power = None
            return
        pts = descending_branch(p_at_f)
        if not pts:
            self._f_of_power = self._level_of_power = None
            return

        def _nearest(target_w: float):
            return min(pts, key=lambda pv: abs(pv[0] - target_w))

        self._f_of_power = lambda w: _nearest(w)[1]
        # Handed to the controller with the seed (``controller.seed_to`` in ``drive``), so
        # it can tell whether the target IS the level measured at that point.
        self._level_of_power = lambda w: _nearest(w)[0]

    def set_policy(self, policy: str) -> None:
        self.router.policy = policy
        self.policy = policy

    def _update_active_targets(self, lam: float, f: float) -> None:
        """Tell the spread policy how many replicas of each pool to use at this load.

        k = ceil(N/c) with N = lambda_pool * (A + B*c) -- the same count first-fit
        packing reaches, so the two policies differ only in arrangement.
        """
        if self.policy != "spread" or not self.residence:
            return
        flex = lam * self.split.flexible
        fixed_l = lam * self.split.large_only
        fixed_s = lam * self.split.small_only
        rates = {"large": fixed_l + flex * (1.0 - f), "small": fixed_s + flex * f}
        for role, lam_p in rates.items():
            r = self.residence.get(role)
            c = self.caps.get(role) or 1
            if not r or c <= 0:
                continue
            n_expected = lam_p * (r["A"] + r["B"] * c)
            self.router.set_active_target(role, max(1, math.ceil(n_expected / c)))

    # --- the loop ---------------------------------------------------------------
    def _build_scheduled_stream(self, schedule, seconds: float):
        """One arrival stream from piecewise-constant rates: ``[(t_start_s, lam), ...]``.

        Each segment is its own draw of the configured process over its own duration,
        shifted to its start, so a rate step lands at the tick it was asked for and the
        memoryless property holds within a segment. Returns the concatenated stream and
        the sorted segments (the current one is looked up by elapsed time in ``drive``)."""
        segs = sorted((float(t), float(l)) for t, l in schedule)
        if not segs or segs[0][0] > 0.0:
            segs.insert(0, (0.0, self.lam))
        offsets: list[float] = []
        for k, (t_k, lam_k) in enumerate(segs):
            t_end = segs[k + 1][0] if k + 1 < len(segs) else seconds
            dur = min(t_end, seconds) - t_k
            if dur <= 0 or lam_k <= 0:
                continue
            s = arrivals.build(self.arrival, lam_k, dur, self.rng,
                               duty=self.burst_duty, burst_s=self.burst_s)
            offsets.extend(t_k + o for o in s.offsets)
        stream = arrivals.Stream(offsets, self.arrival,
                                 {"rate": len(offsets) / seconds if seconds > 0 else 0.0,
                                  "horizon_s": seconds, "schedule": segs})
        return stream, segs

    def drive(self, seconds: float, *, f: float = 0.0, controller=None,
              target=None, phase: str = "", lam: float | None = None,
              lam_schedule=None, rate_hook=None) -> dict:
        """Run for ``seconds``. With ``controller`` and ``target``, ``f`` is recomputed
        each tick; otherwise it is pinned at ``f`` (the open-loop plant).

        ``lam_schedule`` (optional): ``[(t_start_s, lam), ...]`` piecewise-constant offered
        rates inside this phase -- a load step during a hold. Absent, the rate is ``lam``
        throughout. The trace's ``offered_reqs_s`` column carries the rate in force at
        each tick.

        ``rate_hook`` (optional): ``(period_s, window_s, fn)``. Every ``period_s`` the
        arrivals of the last ``window_s`` are counted into an OBSERVED rate and
        ``fn(lam_hat, measured_w, target_w, controller, elapsed_s)`` is called before the
        controller's update, so whatever it does (re-solve a model map, re-seed the knob)
        takes effect on that tick. What a load balancer measures, not the nominal rate.
        """
        lam = self.lam if lam is None else lam
        if lam_schedule:
            stream, segments = self._build_scheduled_stream(lam_schedule, seconds)
        else:
            stream = arrivals.build(self.arrival, lam, seconds, self.rng,
                                    duty=self.burst_duty, burst_s=self.burst_s)
            segments = [(0.0, lam)]
        hook_period = hook_window = hook_fn = None
        next_hook = None
        if rate_hook is not None:
            hook_period, hook_window, hook_fn = rate_hook
            next_hook = float(hook_window)      # the first estimate needs a full window
        fl, router = self.fleet, self.router
        t0 = time.monotonic()
        nxt = t0 + self.dt
        i = tick = 0
        seg_i = 0
        lam_now = segments[0][1]
        cur_f = f
        prev_tgt = None
        rows = []
        while True:
            now = time.monotonic()
            el = now - t0
            if el >= seconds:
                break
            while seg_i + 1 < len(segments) and el >= segments[seg_i + 1][0]:
                seg_i += 1
                lam_now = segments[seg_i][1]
            self._update_active_targets(lam_now, cur_f)
            while i < len(stream.offsets) and stream.offsets[i] <= el:
                cls = self.split.draw(self.rng)
                rep = router.route(cls, cur_f, self.rng)
                fl.submit(rep.idx, self.rng.randrange(self.pool_size), cls)
                i += 1
            fl.poll()
            # The router's occupancy view must come from the workers' scheduler truth,
            # not from its own submit/complete bookkeeping, which drifts.
            for rid, st in fl.states.items():
                router.replicas[rid].inflight = st.running + st.waiting
            if now >= nxt:
                nxt += self.dt
                p = fl.total_power_w()
                if controller is not None and target is not None:
                    tgt = float(target[min(tick, len(target) - 1)])
                    # SETPOINT FEEDFORWARD. On a target CHANGE, jump the knob straight to
                    # the f that the measured P(f) map says produces this power, and let
                    # the controller correct the residual. Without it the loop rediscovers
                    # that f by integration every time, at ki*e per second -- with a slow
                    # integral, a ramp that can span a whole run without settling.
                    #
                    # Only on a CHANGE: re-seeding every tick would overwrite the integral
                    # and destroy the steady-state correction the controller exists to
                    # provide.
                    if (self._f_of_power is not None and tgt == tgt
                            and (prev_tgt is None or abs(tgt - prev_tgt) > 1e-6)
                            and hasattr(controller, "seed_to")):
                        f_ff = self._f_of_power(tgt)
                        if f_ff is not None:
                            controller.seed_to(f_ff, p, tgt,
                                               self._level_of_power(tgt)
                                               if self._level_of_power else None)
                    prev_tgt = tgt
                    if hook_fn is not None and el >= next_hook:
                        j = bisect.bisect_right(stream.offsets, el - hook_window, 0, i)
                        lam_hat = (i - j) / float(hook_window)
                        hook_fn(lam_hat, p, tgt, controller, el)
                        next_hook += float(hook_period)
                    cur_f = controller.update(p, tgt, self.dt)
                else:
                    tgt = float("nan")
                row = {
                    "phase": phase, "t": round(el, 3), "f": round(cur_f, 4),
                    # Absolute CLOCK_MONOTONIC, the base requests.csv's t_done is on. "t"
                    # restarts at 0 every phase, so without this a request cannot be
                    # attributed to the part of a phase it completed in -- e.g. the hold
                    # of a power drop against the nominal level either side of it.
                    "t_mono": round(now, 3),
                    "target_w": tgt, "measured_w": round(p, 2),
                    "power_large_w": round(fl.pool_power_w("large"), 2),
                    "power_small_w": round(fl.pool_power_w("small"), 2),
                    "running_total": fl.total_running(),
                    "waiting_total": fl.total_waiting(),
                    # Whether a replica reached its concurrency cap or ran out of KV
                    # first is not recoverable after the fact without this column.
                    "kv_usage_max": round(fl.kv_usage_max(), 4),
                    "b_max": max(fl.occupancy().values(), default=0),
                    "active_large": fl.active_replicas("large"),
                    "active_small": fl.active_replicas("small"),
                    "occupancy": json.dumps(fl.occupancy()),
                    "n_completed": len(fl.completed),
                    "offered_reqs_s": round(lam_now, 3),
                    "policy": self.policy,
                }
                rows.append(row)
                self.trace.append(row)
                tick += 1
            time.sleep(0.002)
        fl.poll(block_s=0.2)
        out = {"rows": rows, "realized_rate": stream.mean_rate,
               "arrival_cv": stream.index_of_cv(),
               "peak_rate_1s": stream.peak_rate(1.0)}
        if lam_schedule:
            out["lam_schedule"] = segments
        return out

    def drain(self, timeout_s: float = 60.0, settle_s: float = 3.0,
              phase: str = "") -> dict:
        """Stop dispatching and wait for the fleet's backlog to reach zero.

        ``steady``'s tail-averaging is a settle hold, not a drain. Without a drain,
        backlog left by one phase carries into the next, inflating its true offered load
        above its nominal utilization and biasing its power reading with queueing rather
        than the thing under test. It bites hardest where the numbers matter most: fleet_routing_sweep
        sweeps utilization ascending with an inner f-grid, so each high-load point would
        hand its queue to the next, and interleaved pack/spread arms leave systematically
        different backlogs (pack queues at its fill boundary, spread does not), so the
        carryover would be confounded with the arm.

        No logical clock is needed: ``drive`` builds its own arrival stream per call with
        offsets relative to its own ``t0``, so pausing between calls cannot skip or burst
        arrivals.

        A timeout is **not** a failure to hide -- backlog that will not clear in
        ``timeout_s`` is a real capacity problem, and the caller is told so.
        """
        fl = self.fleet
        t0 = time.monotonic()
        reached = None
        while time.monotonic() - t0 < timeout_s:
            fl.poll(block_s=0.05)
            if not fl.states_readable():
                # A queue that cannot be read reports 0, which is exactly what a drained
                # fleet looks like. Refuse to call that success.
                print("  [WARN] scheduler state unreadable on at least one replica -- "
                      "cannot verify the drain; treating as NOT drained")
                return {"drained": False, "outstanding": None, "elapsed_s": None,
                        "reason": "scheduler_unreadable"}
            if fl.total_waiting() + fl.total_running() == 0:
                reached = time.monotonic() - t0
                break
            time.sleep(0.02)
        if reached is None:
            left = fl.total_waiting() + fl.total_running()
            print(f"  [WARN] drain timed out after {timeout_s:.0f}s with {left} still "
                  f"outstanding{' before ' + phase if phase else ''} -- the next phase's "
                  f"power reading inherits it, and that is a capacity result, not noise")
            return {"drained": False, "outstanding": left, "elapsed_s": timeout_s}
        # Let power fall back to the idle floor before the next phase starts measuring:
        # a replica that has just gone empty takes ~1 s to get there.
        end = time.monotonic() + settle_s
        while time.monotonic() < end:
            fl.poll(block_s=0.05)
            time.sleep(0.02)
        return {"drained": True, "outstanding": 0, "elapsed_s": reached}

    # --- helpers ----------------------------------------------------------------
    def steady(self, rows, tail_frac: float = 0.5) -> dict:
        """Average the last ``tail_frac`` of a phase -- the part after the transient."""
        if not rows:
            return {}
        k = max(1, int(len(rows) * (1 - tail_frac)))
        tail = rows[k:]
        def m(key):
            vals = [r[key] for r in tail if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else float("nan")
        return {
            "power_w": m("measured_w"), "power_large_w": m("power_large_w"),
            "power_small_w": m("power_small_w"), "running": m("running_total"),
            "waiting": m("waiting_total"), "active_large": m("active_large"),
            "active_small": m("active_small"), "n_ticks": len(tail),
            # Reported so a phase can be read as "reached its cap" or "ran out of KV
            # first" without going back to the raw trace.
            "kv_usage_max": m("kv_usage_max"), "b_max": m("b_max"),
        }

    def calibrate_band(self, seconds: float,
                       f_values=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
                       lam: float | None = None) -> dict:
        """Measure the reachable power band by sweeping the knob.

        Every target in every phase is placed inside this band. Choosing targets by eye
        instead produces phases whose setpoints are unreachable at that load, and a
        controller railed against an impossible target reads as a control failure when
        it is an experiment-design failure.

        **Sweeps rather than probing the two extremes**, because P(f) is not monotonic.
        At f=0 the small pool is idle and the large pool saturated; raising f slightly
        wakes the small replica and *adds* its whole jump before the large pool sheds
        anything, so power RISES first and peaks at intermediate f. Probing only f=0 and
        f=1 therefore reports a ceiling that is not the ceiling: at u=0.4 the endpoints
        give 1592 W while the model puts the true peak at ~1783 W near f=0.07, a band
        ~190 W too short at the top.
        """
        at = {}
        drained = {}
        for f in f_values:
            # The extremes run back to back and are the two ends of the band itself, so
            # a queue left by one lands squarely in the other's window and compresses
            # the measured band -- the single number every target below is placed inside.
            drained[f] = self.drain(phase=f"calib_f{f:g}")["drained"]
            res = self.drive(seconds, f=f, phase=f"calib_f{f:g}", lam=lam)
            at[f] = self.steady(res["rows"]).get("power_w", float("nan"))
        lo, hi = min(at.values()), max(at.values())
        return {"p_at_f": {str(k): v for k, v in at.items()},
                "p_lo_w": lo, "p_hi_w": hi, "band_w": hi - lo,
                "drained_before": {str(k): v for k, v in drained.items()},
                "f_at_lo": min(at, key=at.get), "f_at_hi": max(at, key=at.get)}


# How much higher the engine's admission ceiling sits than the router's packing cap.
#
# With no headroom the two are the same number: `--concurrency-large 64` sets BOTH
# `pack_cap` and `max_num_seqs` to 64, so the router hands over to the next replica at
# exactly the sequence count at which vLLM would itself start queueing. Two
# discontinuities on top of each other, and crossing the boundary changes the lit replica
# count AND the queueing regime in the same tick -- the noisiest possible place to put the
# transition, and the one place the plant is least predictable.
#
# With headroom the router still decides occupancy (it spills at `pack_cap`), but a burst
# that briefly overruns the cap is absorbed by the engine instead of queueing. 16 is small
# next to the caps in use (64 and 128) so it does not materially change the achievable
# concurrency or the power curve, and large enough to cover the arrival jitter between the
# router's decision and the scheduler's next step at these rates.
#
# NOTE for capacity lookups: `capacity_bisection` measures capacity at a given `max_num_seqs` with no
# router, so its number describes a replica held at that fill level. The router holds
# replicas at `pack_cap`, so the capacity that applies here is the one measured AT THE
# ROUTER CAP -- `fleet_routing_sweep` passes `args.concurrency_*` to `feasible.load_measured`, not the
# engine ceiling. Keep that the case if either value is changed.
ADMISSION_HEADROOM_SEQS = int(os.environ.get("FLEET_ADMISSION_HEADROOM_SEQS", 16))
#
# Settable so it can be a CONTROL: only an arm that varies the headroom alone can credit
# or dismiss its effect on power excursion.


def make_configs(n_large: int, n_small: int, large: str, small: str, workload: str,
                 concurrency_large: int, concurrency_small: int,
                 pack_cap_large: int | None = None, pack_cap_small: int | None = None,
                 pool_size: int = 4096, seed: int = 0, gpu_offset: int = 0,
                 concurrency_per_instance: list[int] | None = None,
                 gpu_memory_utilization: float | None = None,
                 require_gpus: bool = True):
    """One replica per GPU. Always.

    Concurrency cap
    ---------------
    The cap is the number of sequences an instance **actively serves at once**
    (vLLM's ``max_num_seqs``) -- queued requests do not count against it. It can be set

      * per pool, via ``concurrency_large`` / ``concurrency_small``; or
      * per instance, via ``concurrency_per_instance`` (large replicas first, then
        small), which overrides the pool values. Useful for a deliberately
        heterogeneous pool -- e.g. one deep replica to absorb bursts beside several
        shallow ones.

    ``pack_cap`` -- how full the router lets an instance get before lighting the next --
    **defaults to that instance's concurrency cap**, which is the setting that makes the
    router hand over exactly when the engine would otherwise start queueing. Override
    only to study packing below the engine's own ceiling.

    Why one replica per GPU, never co-located: the workers would all sample the same
    NVML device, so "fleet power" would be one card's draw counted N times; idling a
    replica would free no card, so the packing lever could not appear at all; and the
    replicas would contend for the same SMs, so latency would reflect that contention
    rather than the serving load.
    """
    n = n_large + n_small
    if concurrency_per_instance is not None:
        if len(concurrency_per_instance) != n:
            raise ValueError(
                f"--concurrency-per-instance needs {n} values "
                f"({n_large} large then {n_small} small), got "
                f"{len(concurrency_per_instance)}")
        caps = list(concurrency_per_instance)
    else:
        caps = [concurrency_large] * n_large + [concurrency_small] * n_small

    cfgs = []
    for i in range(n):
        role = "large" if i < n_large else "small"
        model = large if role == "large" else small
        pc = pack_cap_large if role == "large" else pack_cap_small
        pc = pc if pc is not None else caps[i]
        # The ENGINE gets headroom above the ROUTER's cap. See ADMISSION_HEADROOM_SEQS.
        cfgs.append(fleetlib.ReplicaConfig(i, i + gpu_offset, role, model,
                                           pc + ADMISSION_HEADROOM_SEQS, pc,
                                           workload, pool_size, seed,
                                           gpu_memory_utilization,
                                           max_num_batched_tokens=CHUNK_BUDGET))
    if require_gpus:
        _assert_enough_gpus(n)
    return cfgs


def _assert_enough_gpus(n_required: int) -> None:
    """Fail before loading any model if the allocation is too small.

    Discovering this after a 10-minute model load, as a CUDA error from replica 3, wastes
    the whole job and reads as a code fault rather than a resource one.
    """
    import os

    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        n_avail = len([x for x in vis.split(",") if x.strip() != ""])
    else:
        try:
            import pynvml
            pynvml.nvmlInit()
            n_avail = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
        except Exception:
            return
    if n_avail < n_required:
        raise SystemExit(
            f"this fleet needs {n_required} GPUs, one per replica, but only {n_avail} "
            f"are visible. Request more (e.g. --gres=gpu:h100:{n_required}); "
            f"replicas are never co-located on one card -- see make_configs.")


# ======================================================================================
# Shared CLI surface, so the fleet experiments agree on what a fleet is
# ======================================================================================

def add_fleet_args(ap) -> None:
    """Fleet shape, concurrency caps, and workload -- identical across fleet experiments."""
    ap.add_argument("--n-large", type=int, default=3)
    ap.add_argument("--n-small", type=int, default=1)
    ap.add_argument("--gpu-offset", type=int, default=0,
                    help="first PHYSICAL gpu index the fleet may use (default 0); "
                         "replicas land on offset..offset+n-1. 0 is fine under a "
                         "cgroup, which hands the job its own GPU 0, but not on a "
                         "shared node, and setting CUDA_VISIBLE_DEVICES instead "
                         "does NOT work, because fleet.py overwrites it per worker with "
                         "the absolute index. On the RTX 8000 node, --gpu-offset 1 is "
                         "what keeps the fleet off the display-attached card, which "
                         "idles at 24 W against 10-13 W and would otherwise read as a "
                         "permanently lit replica.")
    ap.add_argument("--large", default="llama31-8b")
    ap.add_argument("--small", default="llama32-1b")
    ap.add_argument("--workload", default="sharegpt")
    ap.add_argument("--concurrency-large", type=int, default=64,
                    help="sequences actively served at once per large instance "
                         "(vLLM max_num_seqs); queued requests do not count")
    ap.add_argument("--concurrency-small", type=int, default=128,
                    help="same, per small instance")
    ap.add_argument("--concurrency-per-instance", default=None,
                    help="comma-separated per-instance caps, large replicas first; "
                         "overrides the per-pool values")
    ap.add_argument("--pack-cap-large", type=int, default=None,
                    help="router fill level before lighting the next large replica "
                         "(default: the concurrency cap)")
    ap.add_argument("--pack-cap-small", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=None,
                    help="fraction of each card vLLM may use. Weights are fixed, so this "
                         "sets the KV POOL and therefore how many sequences a replica can "
                         "hold -- the only lever that makes memory the binding constraint "
                         "without changing the workload or the concurrency cap. At the "
                         "default sharegpt is request-bound by 30x, and even "
                         "arxiv_summarization leaves 2.6x headroom at cap 64. "
                         "Shrinking the pool is a ONE-VARIABLE way in; raising the "
                         "cap past the KV ceiling is the other, and confounds the batch "
                         "size with the constraint.")
    ap.add_argument("--policy", choices=("pack", "spread"), default="pack",
                    help="pack: fill an instance to its cap, then start the next -- the "
                         "power lever. spread: least-loaded, the control arm.")
    ap.add_argument("--arrival", choices=("poisson", "bursty", "gamma"), default="poisson")
    ap.add_argument("--burst-duty", type=float, default=0.3)
    ap.add_argument("--burst-s", type=float, default=5.0)
    ap.add_argument("--class-split", default="0.0,0.0,1.0",
                    help="large_only,small_only,flexible shares, summing to 1. ALL "
                         "flexible by default: only flexible traffic is routable, so a "
                         "pinned share is knob authority removed. A 0.2/0.0/0.8 split "
                         "keeps one large replica lit at all times, which raises the "
                         "floor ~400 W and roughly HALVES the band (617 W against "
                         "1185 W at 100%% flexible). Since the knob's authority in "
                         "watts per unit f IS the band, that doubles the knob travel "
                         "needed for any correction. Set non-zero shares "
                         "deliberately, per experiment, when studying pinned traffic.")
    ap.add_argument("--pool-size", type=int, default=4096)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-tag", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="run every CPU-only step -- argument parsing, replica configs, "
                         "analytic capacity, target construction -- then exit before "
                         "touching a GPU. `--help` does NOT cover these: it returns "
                         "during argparse, so a NameError further down would reach the "
                         "queue and fail the job hours later. Run this before submitting.")


def configs_from_args(args):
    per = ([int(x) for x in args.concurrency_per_instance.split(",")]
           if args.concurrency_per_instance else None)
    return make_configs(args.n_large, args.n_small, args.large, args.small,
                        args.workload, args.concurrency_large, args.concurrency_small,
                        args.pack_cap_large, args.pack_cap_small,
                        args.pool_size, args.seed,
                        gpu_offset=getattr(args, "gpu_offset", 0),
                        concurrency_per_instance=per,
                        gpu_memory_utilization=getattr(
                            args, "gpu_memory_utilization", None),
                        # a dry run validates the CPU path on a login node, where there
                        # is no GPU to count
                        require_gpus=not getattr(args, "dry_run", False))


def fleet_capacity(args, workload) -> dict:
    """Sustainable fleet capacity **at the chosen concurrency caps**, derived from the
    measured step-time line rather than from a separate calibration.

    A replica serving c sequences retires ``C_tok(c) = c/(t0 + t1*c)`` tok/s. That form
    matched an independent open-loop bisection to 0.3%, so offered load can be expressed
    as a fraction of capacity without spending GPU hours re-measuring it -- and, more
    importantly, the fraction then means the same thing at every concurrency cap, which
    a fixed req/s number does not.
    """
    import json
    from pathlib import Path

    rep_path = Path(getattr(args, "residence_report", None)
                    or "results/power_curves_report.json")
    if not rep_path.exists():
        return {"available": False, "reason": f"{rep_path} not found; run fit_power_model.py"}
    rep = json.loads(rep_path.read_text())

    def cap_for(model, c):
        # EXACT (model, workload) ONLY. Falling back to another workload's entry for the
        # same model would, on arxiv_summarization, divide llama31-8b's SHAREGPT
        # step-time line by arxiv's decode length and report a 244 req/s fleet where the
        # arxiv constants give 63.7 -- every load set as a fraction of it would be 3.8x
        # too high.
        #
        # t1 does not transfer across workloads (21-58% on 6 of 9 models), so borrowing
        # another workload's line is never sound. lib/feasible.load_measured already
        # refuses a near match for exactly this reason; two functions answering the same
        # question must not have opposite safety policies. Returning None makes --utils
        # unavailable, which is the correct outcome: measure the workload, or pass --loads.
        for key in (f"{model}__{workload.name}",):
            e = rep.get(key)
            if e and e.get("residence", {}).get("ok"):
                r = e["residence"]
                c_tok = c / (r["t0_s"] + r["t1_s"] * c)
                kappa = r["t0_s"] / CHUNK_SIZE + r["t1_s"]
                return {"key": key, "c_tok": c_tok,
                        "c_req": c_tok / workload.mean_decode,
                        "residence": {
                            "A": workload.mean_prefill * kappa
                                 + workload.mean_decode * r["t0_s"],
                            "B": workload.mean_decode * r["t1_s"]}}
        return None

    L = cap_for(args.large, args.concurrency_large)
    S = cap_for(args.small, args.concurrency_small) if args.n_small else None
    if L is None or (args.n_small and S is None):
        missing = [m for m, v in ((args.large, L), (args.small, S if args.n_small else L))
                   if v is None]
        return {"available": False,
                "reason": f"no residence fit for {', '.join(missing)} on "
                          f"{workload.name}; run concurrency_sweep on that workload (never borrow "
                          f"another workload's step-time line -- see fleet_driver.cap_for)"}
    cap_req = args.n_large * L["c_req"] + (args.n_small * S["c_req"] if S else 0.0)
    cap_tok = args.n_large * L["c_tok"] + (args.n_small * S["c_tok"] if S else 0.0)
    return {"available": True, "capacity_reqs_s": cap_req, "capacity_decode_tok_s": cap_tok,
            "large": L, "small": S, "residence": {
                k: v for k, v in (("large", L.get("residence")),
                                  ("small", S.get("residence") if S else None)) if v},
            "concurrency_large": args.concurrency_large,
            "concurrency_small": args.concurrency_small}


if __name__ == "__main__":
    _selfcheck()
