#!/usr/bin/env python
"""poisson_load -- ONE replica under Poisson arrivals: per-request residence, occupancy, power and
natural drains, for the single-GPU model figures.

What a phase measures
---------------------
At each offered rate the replica is warmed up, then observed for a measurement window
while every request's submit / first-token / completion times are logged (concurrency_sweep and capacity_bisection
keep only aggregates), the scheduler's running and waiting counts are sampled at 10 Hz and
GPU power is sampled by NVML. Then admissions STOP and the replica drains naturally: the
in-flight set is snapshotted at the stop instant (each request's prompt length P, output
length D and tokens generated so far n, so its residual q = D - n is known exactly, which
the suite's exact-length requests make possible), and the time until the engine is empty is
the measured drain. Admissions resume, the replica re-warms, and the stop-and-drain repeats
``--n-drains`` times per rate.

What it adds to concurrency_sweep and capacity_bisection
-------------------------------
* Residence per request needs timestamps per request, which the closed-loop sweeps do not
  record.
* Occupancy under Poisson arrivals is in capacity_bisection's probes, but those sit near capacity by
  construction (bisection). Here the rates span the operating range.
* A fleet drain can only be judged with a Monte-Carlo residual model; here the residuals
  are the ORACLE values, so the closed form t0*max(q) + t1*sum(q) is tested by itself, and
  the offline (length-biased) estimate can be scored against the same data.

Stability of every phase is judged exactly as capacity_bisection does (waiting-list slope, no floor), so
a phase above capacity is recorded but flagged, never silently used as a steady state.

Rates are offered as fractions of the MEASURED capacity of this (model, workload, cap)
cell when capacity_bisection has one on this hardware; otherwise the cell's own t0/t1 place the grid
(recorded as such), and the per-phase stability verdict is what decides validity.

    python poisson_load.py \\
        --model llama31-8b --workload sharegpt,humaneval --max-num-seqs 64
"""

from __future__ import annotations

import argparse
import csv
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
from lib.env import results_root, run_dir
from capacity_bisection import STABILITY_RULE, is_stable, reset_engine, stability_threshold

CHUNK = 2048  # the scheduler token budget S the suite runs at (recorded, not assumed)


# ------------------------------------------------------------------ rate reference -----
def reference_capacity(spec, wl, cap: int) -> tuple[float | None, str]:
    """(req/s, provenance) for placing the rate grid. Measured capacity_bisection first, else t0/t1."""
    try:
        from analyze.capacity_lookup import lookup
        r = lookup(results_root(), spec.slug, wl.name, cap)
    except Exception as e:  # noqa: BLE001
        r = None
        print(f"  capacity lookup failed: {e}")
    if r and r.get("reqs_s"):
        return float(r["reqs_s"]), f"measured ({r.get('source', 'capacity_bisection')})"
    rep = results_root() / "power_curves_report.json"
    if rep.exists():
        j = json.loads(rep.read_text())
        e = (j.get(f"{spec.slug}__{wl.name}") or {}).get("residence") or {}
        if e.get("ok"):
            s = float(wl.mean_decode) + float(wl.mean_prefill) / CHUNK
            c = cap / (s * (float(e["t0_s"]) + float(e["t1_s"]) * cap))
            return c, "analytic t0/t1 (grid placement only; phases judged by the queue)"
    return None, "none"


# ------------------------------------------------------------------ in-flight snapshot ---
def snapshot_inflight(engine, live: dict) -> dict:
    """Every request in the engine at the stop instant with its exact residual."""
    running_ids: set[str] = set()
    waiting_ids: set[str] = set()
    try:
        sched = engine.engine_core.engine_core.scheduler
        running_ids = {r.request_id for r in list(sched.running)}
        waiting_ids = {r.request_id for r in list(sched.waiting)}
    except Exception:  # noqa: BLE001  (older/newer engine layout: fall back to 'live')
        running_ids = set(live)
    run_rows, wait_rows = [], []
    for rid, rec in live.items():
        row = [int(rec["P"]), int(rec["D"]), int(rec["n"])]
        if rid in waiting_ids:
            wait_rows.append(row)
        else:
            run_rows.append(row)
    q = [d - n for _, d, n in run_rows]
    return {
        "n_running": len(run_rows), "n_waiting": len(wait_rows),
        "n_prefill_pending": sum(1 for _, _, n in run_rows if n == 0),
        "q_sum": int(sum(q)), "q_max": int(max(q)) if q else 0,
        "ctx_sum": int(sum(p + n for p, _, n in run_rows)),
        "running_PDn": run_rows, "waiting_PDn": wait_rows,
    }


# ------------------------------------------------------------------ one phase -----------
def run_phase(engine, specs, wl, rate: float, args, sampler, rng, phase_idx: int,
              req_writer, trace_writer) -> dict:
    reset_engine(engine)
    horizon = max(args.warmup_s, args.warmup_max_s) + args.max_measure_s + args.n_drains * (
        args.max_measure_s + args.drain_timeout_s) + 120.0
    stream = arrivals.build("poisson", rate, horizon, rng)
    t0 = time.monotonic()
    live: dict[str, dict] = {}
    completed: list[dict] = []
    samples: list[tuple] = []
    i = seq = 0
    admitting = True
    state, t_state = "warmup", 0.0
    measure_s = float(args.measure_s)
    rewarm_s = float(args.rewarm_min_s)
    T_warm = 0.0
    n_done_warm = 0
    window: list[float] = [0.0, 0.0]
    t_stop = None
    snapshot: dict | None = None
    drains: list[dict] = []
    last_sample = -1.0
    aborted = False

    def finish_request(o, now):
        rec = live.pop(o.request_id)
        row = {"phase": phase_idx, "workload": wl.name, "rate_reqs_s": rate,
               "rid": o.request_id, "t_submit": rec["t"] - t0,
               "t_first": (rec["first"] - t0) if rec["first"] else float("nan"),
               "t_done": now - t0, "P": rec["P"], "D": rec["D"], "n_out": rec["n"],
               "state_at_submit": rec["state"]}
        completed.append(row)
        req_writer.writerow(row)

    while True:
        now = time.monotonic()
        el = now - t0
        if admitting:
            while i < len(stream.offsets) and stream.offsets[i] <= el:
                spec = specs[seq % len(specs)]
                rid = f"p{phase_idx}r{seq}"
                seq += 1
                engine.add_request(rid, {"prompt_token_ids": spec.prompt_token_ids},
                                   reqlib.sampling_params(spec))
                live[rid] = {"t": now, "P": spec.num_prefill_tokens,
                             "D": spec.num_decode_tokens, "first": None, "n": 0,
                             "state": state}
                i += 1
        if engine.has_unfinished_requests():
            outs = engine.step()
        else:
            outs = []
            time.sleep(0.002)
        now = time.monotonic()
        el = now - t0
        for o in outs:
            rec = live.get(o.request_id)
            if rec is None:
                continue
            n = len(o.outputs[0].token_ids) if o.outputs else 0
            if n > 0 and rec["first"] is None:
                rec["first"] = now
            rec["n"] = max(rec["n"], n)
            if o.finished:
                finish_request(o, now)
        if now - last_sample >= 0.1:
            st = eng.read_state(engine)
            samples.append((el, st.num_running, st.num_waiting, int(admitting), state))
            trace_writer.writerow([phase_idx, wl.name, rate, round(now, 3), round(el, 3),
                                   st.num_running, st.num_waiting, int(admitting), state])
            last_sample = now

        # ---- state machine: warmup -> measure -> drain -> (rewarm -> drain)* ----
        if state == "warmup" and el - t_state >= args.warmup_s:
            e2e = [c["t_done"] - c["t_submit"] for c in completed]
            T_warm = float(np.mean(e2e)) if e2e else 0.0
            # The window is sized from residences SEEN in warm-up, and the in-flight count needs
            # ~2 residences to fill (1 - e^-2 = 86 %). A fixed warm-up shorter than one residence
            # would see none finish and measure a fallback window mid-ramp, so keep warming until
            # both hold.
            ready = len(e2e) >= args.warmup_min_done and el - t_state >= 2.0 * T_warm
            if ready or el - t_state >= args.warmup_max_s:
                measure_s = min(args.max_measure_s,
                                max(args.measure_s, args.window_residences * T_warm))
                rewarm_s = max(args.rewarm_min_s, args.rewarm_residences * T_warm)
                n_done_warm = len(e2e)
                state, t_state = "measure", el
                window = [now, 0.0]
                print(f"    warm-up done after {el:.0f} s ({n_done_warm} finished): mean residence "
                      f"{T_warm:6.2f} s -> window {measure_s:.0f} s, re-warm {rewarm_s:.0f} s"
                      f"{'' if ready else '  (warm-up cap reached)'}", flush=True)
        elif state == "measure" and el - t_state >= measure_s:
            window[1] = now
            admitting = False
            t_stop = now
            snapshot = snapshot_inflight(engine, live)
            state, t_state = "drain", el
        elif state == "drain":
            empty = not engine.has_unfinished_requests()
            timed_out = (now - t_stop) > args.drain_timeout_s
            if empty or timed_out:
                d = dict(snapshot or {})
                d.update({"t_stop": t_stop - t0, "drain_s": now - t_stop,
                          "timed_out": bool(timed_out and not empty)})
                pw = sampler.average_over(t_stop, now) or {}
                d["avg_power_w"] = pw.get("avg_power_w")
                drains.append(d)
                print(f"    drain {len(drains)}: {d['n_running']} running (+{d['n_waiting']} waiting, "
                      f"{d['n_prefill_pending']} in prefill), q_max {d['q_max']}, q_sum {d['q_sum']}"
                      f" -> {d['drain_s']:7.2f} s{'  TIMED OUT' if d['timed_out'] else ''}", flush=True)
                if timed_out and not empty:
                    reset_engine(engine)
                    aborted = True
                    break
                if len(drains) >= args.n_drains:
                    break
                # resume a fresh (memoryless) arrival stream: drop the offsets that fell
                # due during the drain instead of releasing them as a burst
                while i < len(stream.offsets) and stream.offsets[i] <= el:
                    i += 1
                admitting = True
                state, t_state = "rewarm", el
        elif state == "rewarm" and el - t_state >= rewarm_s:
            admitting = False
            t_stop = now
            snapshot = snapshot_inflight(engine, live)
            state, t_state = "drain", el
        if el > horizon:
            print("    phase horizon exceeded; aborting the phase", flush=True)
            reset_engine(engine)
            aborted = True
            break

    # ---- window statistics ----
    w0, w1 = window[0] - t0, (window[1] - t0) if window[1] else (t_stop - t0 if t_stop else el)
    win = [s for s in samples if w0 <= s[0] < w1]
    run_v = np.array([s[1] for s in win], float)
    wait_t = np.array([s[0] for s in win], float)
    wait_v = np.array([s[2] for s in win], float)
    slope = slope_se = float("nan")
    if len(wait_t) > 10:
        A = np.vstack([np.ones(len(wait_t)), wait_t]).T
        (intercept, slope), *_ = np.linalg.lstsq(A, wait_v, rcond=None)
        resid = wait_v - (intercept + slope * wait_t)
        sxx = float(np.sum((wait_t - wait_t.mean()) ** 2))
        slope_se = float(np.sqrt(np.sum(resid ** 2) / max(len(wait_t) - 2, 1) / sxx)) if sxx > 0 else float("nan")
    in_win = [c for c in completed if w0 <= c["t_submit"] < w1]
    e2e = np.array([c["t_done"] - c["t_submit"] for c in in_win], float)
    ttft = np.array([c["t_first"] - c["t_submit"] for c in in_win if np.isfinite(c["t_first"])], float)
    tbt = np.array([(c["t_done"] - c["t_first"]) / max(c["D"] - 1, 1) for c in in_win
                    if np.isfinite(c["t_first"]) and c["D"] > 1], float)
    done_in_win = sum(1 for c in completed if w0 <= c["t_done"] < w1)
    pw = sampler.average_over(window[0], window[1]) if window[1] else None
    pq = sampler.percentiles_over(window[0], window[1]) if window[1] else None
    dur = max(w1 - w0, 1e-9)

    def pct(a, q):
        return float(np.percentile(a, q)) if len(a) else float("nan")

    return {
        "phase": phase_idx, "workload": wl.name, "rate_reqs_s": rate,
        # the warm-up actually run (the window opens at w0), which readers take as the window start
        "warmup_s": float(w0) if window[0] else float(el), "warmup_min_s": args.warmup_s,
        "n_done_warmup": n_done_warm,
        "measure_s": float(w1 - w0), "residence_warmup_s": T_warm,
        "n_submitted_window": len(in_win), "realized_rate_reqs_s": len(in_win) / dur,
        "n_completed_window": done_in_win, "throughput_reqs_s": done_in_win / dur,
        "throughput_decode_tok_s": float(sum(c["n_out"] for c in completed
                                             if w0 <= c["t_done"] < w1)) / dur,
        "running_mean": float(run_v.mean()) if len(run_v) else float("nan"),
        "running_p5": pct(run_v, 5), "running_p95": pct(run_v, 95),
        "waiting_mean": float(wait_v.mean()) if len(wait_v) else float("nan"),
        "waiting_slope_per_s": slope, "waiting_slope_se": slope_se,
        "stability_threshold": stability_threshold(rate, slope_se),
        "stable": is_stable(slope, slope_se, rate),
        "e2e_s_mean": float(e2e.mean()) if len(e2e) else float("nan"),
        "e2e_s_p50": pct(e2e, 50), "e2e_s_p95": pct(e2e, 95),
        "ttft_s_mean": float(ttft.mean()) if len(ttft) else float("nan"),
        "tbt_ms_mean": float(1e3 * tbt.mean()) if len(tbt) else float("nan"),
        "avg_power_w": (pw or {}).get("avg_power_w"),
        "p5_power_w": (pq or {}).get("p5"), "p95_power_w": (pq or {}).get("p95"),
        "aborted": aborted, "drains": drains,
    }


# ------------------------------------------------------------------ main ----------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", default="sharegpt", help="comma-separated; one engine load")
    ap.add_argument("--max-num-seqs", type=int, required=True,
                    help="the replica's concurrency cap c (the fleet role cap: 64 / 128)")
    ap.add_argument("--max-num-batched-tokens", type=int, default=CHUNK)
    ap.add_argument("--fractions", default="0.15,0.3,0.45,0.6,0.75",
                    help="offered rates as fractions of the reference capacity")
    ap.add_argument("--rates", default=None, help="absolute req/s list; overrides --fractions")
    ap.add_argument("--warmup-s", type=float, default=30.0, help="minimum warm-up")
    ap.add_argument("--warmup-min-done", type=int, default=10,
                    help="warm up until this many requests finished and >= 2 mean residences passed")
    ap.add_argument("--warmup-max-s", type=float, default=900.0, help="warm-up cap")
    ap.add_argument("--measure-s", type=float, default=120.0, help="minimum window")
    ap.add_argument("--max-measure-s", type=float, default=480.0)
    ap.add_argument("--window-residences", type=float, default=8.0,
                    help="window >= this many mean residence times (measured in warm-up)")
    ap.add_argument("--n-drains", type=int, default=3)
    ap.add_argument("--rewarm-min-s", type=float, default=20.0)
    ap.add_argument("--rewarm-residences", type=float, default=4.0)
    ap.add_argument("--drain-timeout-s", type=float, default=900.0)
    ap.add_argument("--pool-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()

    spec = models.get(args.model)
    # Before run_dir: a multi-GPU instance writes under its own namespace (h100-80gb-tp4),
    # and reads its reference capacity from there too.
    hardware.set_instance(spec.tensor_parallel_size)
    wls = [workloads.get(w) for w in args.workload.split(",") if w]
    if args.smoke:
        args.warmup_s, args.measure_s, args.max_measure_s = 8.0, 15.0, 20.0
        args.warmup_min_done, args.warmup_max_s = 2, 30.0
        args.n_drains, args.rewarm_min_s, args.pool_size = 2, 5.0, 512
        args.fractions = "0.3,0.6"
    tag = args.out_tag or f"{spec.slug}__{'-'.join(w.name for w in wls)}__cap{args.max_num_seqs}"
    out = run_dir("poisson_load", tag)
    print(f"poisson_load: {spec.model_id} cap {args.max_num_seqs} on {[w.name for w in wls]}")
    print(f"  out {out}")

    sampler = ps.for_model(spec)  # its own card, or every card of a multi-GPU instance
    sampler.start()
    dev = sampler.device_info()
    print(f"  gpu {dev['name']} (nvml index {dev['device_index']}, {dev.get('pci_bus_id')})")
    if sampler.num_devices > 1:
        print(f"  instance: TP={spec.tensor_parallel_size} over NVML {dev['device_indices']} "
              f"({', '.join(dev['pci_bus_ids'])}); power is the sum", flush=True)
    engine = eng.build_engine(spec, max_num_seqs=args.max_num_seqs,
                              max_num_batched_tokens=args.max_num_batched_tokens)
    if spec.tensor_parallel_size > 1:
        dev["mem_used_mib_after_load"] = ps.assert_instance_resident(sampler, spec)
    vocab, special = eng.tokenizer_info(engine)
    kv = eng.kv_sizing(engine)
    print(f"  kv {kv['kv_pool_tokens']:,} tokens")
    rng = random.Random(args.seed)

    meta = {
        "experiment": "poisson_load",
        "model": spec.model_id, "model_slug": spec.slug,
        "workloads": {w.name: w.descriptor() for w in wls},
        "workload_stats": {w.name: {"mean_prefill": w.mean_prefill, "mean_decode": w.mean_decode,
                                    "decode_weighted_context": w.decode_weighted_context}
                           for w in wls},
        "max_num_seqs": args.max_num_seqs, "max_num_batched_tokens": args.max_num_batched_tokens,
        "kv": kv, "hardware": hardware.current(), "dtype": hardware.resolve_dtype(spec.dtype),
        "device": dev, "seed": args.seed, "stability_rule": STABILITY_RULE,
        "settings": {k: getattr(args, k) for k in ("warmup_s", "warmup_min_done", "warmup_max_s",
                                                    "measure_s", "max_measure_s",
                                                    "window_residences", "n_drains",
                                                    "rewarm_min_s", "rewarm_residences",
                                                    "drain_timeout_s", "pool_size")},
        "rate_reference": {},
    }
    if spec.tensor_parallel_size > 1:
        meta["tensor_parallel_size"] = spec.tensor_parallel_size
        meta["power_scope"] = f"sum over the instance's {sampler.num_devices} GPUs"
    phases: list[dict] = []
    req_f = open(out / "requests.csv", "w", newline="")
    req_writer = csv.DictWriter(req_f, fieldnames=["phase", "workload", "rate_reqs_s", "rid",
                                                   "t_submit", "t_first", "t_done", "P", "D",
                                                   "n_out", "state_at_submit"])
    req_writer.writeheader()
    tr_f = open(out / "trace.csv", "w", newline="")
    trace_writer = csv.writer(tr_f)
    trace_writer.writerow(["phase", "workload", "rate_reqs_s", "t_mono", "t_phase",
                           "running", "waiting", "admitting", "state"])

    def dump():
        with open(out / "summary.json", "w") as f:
            json.dump({"meta": meta, "phases": phases}, f, indent=1, default=str)

    phase_idx = 0
    try:
        for wl in wls:
            specs = reqlib.pregenerate(wl, args.pool_size, vocab, rng, special)
            if args.rates:
                rates = [float(x) for x in args.rates.split(",")]
                ref, prov = None, "absolute --rates"
            else:
                ref, prov = reference_capacity(spec, wl, args.max_num_seqs)
                if ref is None:
                    # Skip THIS workload, not the rest of the run: a workload without a
                    # measured capacity at this cap must not stop the others. The exit
                    # status still reports it.
                    why = (f"no reference capacity for {spec.slug}/{wl.name}@"
                           f"{args.max_num_seqs} on this hardware; pass --rates")
                    print(f"\n== {wl.name}: SKIPPED -- {why}", flush=True)
                    meta.setdefault("skipped_workloads", {})[wl.name] = why
                    dump()
                    continue
                rates = [round(float(f) * ref, 4) for f in args.fractions.split(",")]
            meta["rate_reference"][wl.name] = {"reqs_s": ref, "provenance": prov, "rates": rates}
            print(f"\n== {wl.name}: reference capacity {ref} req/s [{prov}] -> rates {rates}",
                  flush=True)
            for r in rates:
                print(f"  phase {phase_idx}: {wl.name} at {r} req/s", flush=True)
                ph = run_phase(engine, specs, wl, r, args, sampler, rng, phase_idx,
                               req_writer, trace_writer)
                phases.append(ph)
                phase_idx += 1
                req_f.flush()
                tr_f.flush()
                dump()
                print(f"  -> {'stable ' if ph['stable'] else 'DIVERGE'} running {ph['running_mean']:7.2f}"
                      f"  waiting {ph['waiting_mean']:6.2f} (slope {ph['waiting_slope_per_s']:+.3f}/s)"
                      f"  e2e {ph['e2e_s_mean']:6.2f} s  thr {ph['throughput_reqs_s']:6.2f}/s"
                      f"  P {ph['avg_power_w'] or float('nan'):6.1f} W  "
                      f"drains {[round(d['drain_s'], 2) for d in ph['drains']]}", flush=True)
    finally:
        req_f.close()
        tr_f.close()
        samples = sampler.stop()
        sampler.shutdown()
        try:
            ps.write_trace_csv(samples, out / "power_trace.csv")
        except Exception as e:  # noqa: BLE001
            print(f"  power trace not written: {e}")
        eng.teardown(engine)
        dump()
    print(f"\ndone: {len(phases)} phases -> {out}")
    if meta.get("skipped_workloads"):
        print(f"  {len(meta['skipped_workloads'])} workload(s) SKIPPED: "
              f"{', '.join(meta['skipped_workloads'])}", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
