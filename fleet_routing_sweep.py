#!/usr/bin/env python
"""fleet_routing_sweep -- knob authority vs offered token traffic (open loop, no controller).

At each offered load, pin the routing fraction ``f`` across a grid and measure the
steady-state fleet power. The span from the lowest to the highest reachable power is the
knob's **authority** at that load: the range of power a controller could ever command.

Open loop on purpose. A closed-loop trace confounds the plant with the controller, and
the question here -- how much power range does the fleet even have -- is a property of
the plant. It is also the data the analytical model is validated against, for the same
reason.

Authority is expected to collapse as load rises, and the mechanism is capacity, not
power: at high load the small pool cannot absorb much flexible traffic without its own
queue diverging, so the feasible range of ``f`` shrinks toward a point. Sweeping the
offered load is what makes that visible, and it is why the x-axis here is **token
traffic** rather than a utilization fraction -- utilization needs a capacity number to
normalize by, and it is exactly the capacity that is being probed.

    python fleet_routing_sweep.py \\
        --loads 10,20,40,60,80 --n-large 3 --n-small 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import feasible, metrics, sanity, workloads
from lib.env import results_root, run_dir
from lib.fleet_driver import (CHUNK_BUDGET, CHUNK_SIZE, FleetRunner,
                              add_fleet_args, configs_from_args,
                              fleet_capacity)
from lib.router import ClassSplit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_fleet_args(ap)
    ap.add_argument("--utils", default="0.1,0.2,0.35,0.5,0.65,0.8",
                    help="offered load as a FRACTION of the fleet's sustainable capacity "
                         "at the chosen concurrency caps. This is the x-axis the "
                         "authority plot wants: a fixed req/s number means a different "
                         "utilization at every cap, so bands measured at different caps "
                         "would not be comparable.")
    ap.add_argument("--loads", default=None,
                    help="absolute offered req/s; overrides --utils")
    ap.add_argument("--residence-report", default=None,
                    help="power_curves_report.json (for the analytic capacity)")
    ap.add_argument("--f-grid", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--point-s", type=float, default=60.0, help="seconds per (load, f)")
    args = ap.parse_args()

    fgrid = [float(x) for x in args.f_grid.split(",")]
    split = ClassSplit(*[float(x) for x in args.class_split.split(",")])
    wl = workloads.get(args.workload)

    cap = fleet_capacity(args, wl)
    if args.loads:
        loads = [float(x) for x in args.loads.split(",")]
        utils = [None] * len(loads)
    else:
        if not cap.get("available"):
            ap.error(f"--utils needs the analytic capacity: {cap.get('reason')}. "
                     f"Pass --loads to give absolute req/s instead.")
        utils = [float(x) for x in args.utils.split(",")]
        loads = [u * cap["capacity_reqs_s"] for u in utils]

    # WHICH f ARE EVEN FEASIBLE -- recorded, never enforced.
    #
    # fleet_routing_sweep pins f across the whole grid ON PURPOSE: it is an open-loop sweep of the
    # plant, and clamping it would delete the very points that show where authority runs
    # out; there is no controller here.
    #
    # Without a record, nothing downstream could tell which grid points are operating
    # points and which are a pool being offered more than it retires: arxiv f=0 offers
    # the large pool 30.94 req/s against 3 x 9.12 = 27.36 measured capacity -- 113% -- and
    # its waiting list climbs 0 -> 102 at +1.93 req/s, while `drained` (which records the
    # drain BEFORE the probe) says True.
    #
    # So every point carries `f_feasible`, and the run prints the band up front. This is
    # deliberately SOFT: a missing measured capacity records None and changes nothing,
    # so fleet_routing_sweep can run on workloads that have not been bisected.
    def _feasible_band(lam):
        try:
            # THIS accelerator's results (env.results_root); no-op on the H100 baseline.
            res_root = results_root()
            c_l = feasible.load_measured(res_root, args.large, args.workload,
                                         args.concurrency_large)
            c_s = (feasible.load_measured(res_root, args.small, args.workload,
                                          args.concurrency_small)
                   if args.n_small else 0.0)
            if c_l is None or (args.n_small and c_s is None):
                return None
            return feasible.band(lam, split, args.n_large * c_l,
                                 (args.n_small or 0) * (c_s or 0.0))
        except Exception:
            return None

    cfgs = configs_from_args(args)
    if args.dry_run:
        print(f"DRY RUN ok: {len(cfgs)} replicas, caps "
              f"{[c.max_num_seqs for c in cfgs]}, pack_caps {[c.pack_cap for c in cfgs]}, "
              f"capacity {cap.get('capacity_reqs_s', float('nan')):.1f} req/s, "
              f"loads {[round(l, 1) for l in loads]}, f grid {fgrid}")
        # Exercise the feasibility path HERE, not only inside the run: a --dry-run must
        # reach the code it is meant to preflight.
        for _l in loads:
            _fb = _feasible_band(_l)
            if _fb is None:
                print(f"  lam {_l:.1f}: feasible band UNKNOWN -- no measured capacity for "
                      f"{args.large}/{args.small} on {args.workload} at caps "
                      f"{args.concurrency_large}/{args.concurrency_small}. Points will "
                      f"record f_feasible=None.")
            else:
                inside = [x for x in fgrid
                          if _fb["f_lo"] - 1e-9 <= x <= _fb["f_hi"] + 1e-9]
                print(f"  lam {_l:.1f}: feasible f in "
                      f"[{_fb['f_lo']:.3f}, {_fb['f_hi']:.3f}] -- {len(inside)}/"
                      f"{len(fgrid)} grid points inside. The rest are swept on purpose "
                      f"(this is an open-loop plant sweep) and recorded f_feasible=False.")
        return
    tag = args.out_tag or (f"{args.n_large}L{args.n_small}S__{wl.name}__{args.arrival}"
                           f"__{args.policy}__c{args.concurrency_large}-{args.concurrency_small}")
    out = run_dir("fleet_routing_sweep", tag)
    print(f"fleet_routing_sweep knob authority: {args.n_large}x{args.large} + {args.n_small}x{args.small}")
    print(f"  {wl.label()}  policy {args.policy}  "
          f"concurrency caps L/S {args.concurrency_large}/{args.concurrency_small}")
    if cap.get("available"):
        print(f"  fleet capacity at those caps: {cap['capacity_reqs_s']:.1f} req/s "
              f"({cap['capacity_decode_tok_s']:.0f} decode tok/s)")
    print(f"  loads: " + ", ".join(
        f"{l:.1f}" + (f" (u={u:.2f})" if u is not None else "") for l, u in zip(loads, utils)))
    print(f"  f grid {fgrid}   out {out}")
    print(f"  estimated measurement time: "
          f"{len(loads) * len(fgrid) * args.point_s / 60:.0f} min plus model load")

    runner = FleetRunner(cfgs, n_large=args.n_large, n_small=args.n_small,
                         pack_cap_large=cfgs[0].pack_cap,
                         pack_cap_small=cfgs[-1].pack_cap, policy=args.policy,
                         split=split, workload=wl, offered_reqs_s=loads[0],
                         arrival=args.arrival, burst_duty=args.burst_duty,
                         burst_s=args.burst_s, dt=args.dt, pool_size=args.pool_size,
                         seed=args.seed,
                         residence=(cap.get("residence") if cap.get("available") else None))
    points: list[dict] = []
    bands: list[dict] = []
    drains: list[dict] = []
    summary = {"args": vars(args), "workload": wl.descriptor(), "fleet_capacity": cap,
               "concurrency_caps": [c.max_num_seqs for c in cfgs],
               "pack_caps": [c.pack_cap for c in cfgs],
               "max_num_batched_tokens": CHUNK_SIZE,
               "max_num_batched_tokens_explicit": CHUNK_BUDGET}
    try:
        print("  starting replicas...", flush=True)
        runner.start()
        summary["idle_power_w_total"] = runner.fleet.idle_power_total_w()
        # Stamp which physical card each replica sampled; analysis refuses a summary
        # without this key rather than trust its power.
        summary["power_sampling"] = runner.fleet.sampling_provenance()
        print(f"    fleet warm-idle total {summary['idle_power_w_total']:.1f} W")
        mean_decode = wl.mean_decode

        for lam, u in zip(loads, utils):
            n0 = len(runner.fleet.completed)
            row_for_load = []
            for f in fgrid:
                # Drain between points. Utilization is swept ascending with an inner
                # f-grid; without a drain each point would inherit the previous one's
                # queue (hundreds of requests at high load), and that backlog would land
                # in the next point's band reading.
                dr = runner.drain(phase=f"L{lam:g}_f{f:g}")
                # Carry the offered rate as well as the utilization: `util` is None
                # whenever the run was driven by --loads (absolute req/s), so it is not
                # a reliable coordinate for reporting which points failed to drain.
                drains.append({"util": u, "offered_reqs_s": lam, "f": f, **dr})
                res = runner.drive(args.point_s, f=f, phase=f"L{lam:g}_f{f:g}", lam=lam)
                st = runner.steady(res["rows"])
                fb = _feasible_band(lam)
                rec = {"offered_reqs_s": lam, "util": u, "drained": dr["drained"],
                       # drained is the PRE-probe drain, not this probe's stability.
                       "f_feasible": (None if fb is None
                                      else bool(fb["f_lo"] - 1e-9 <= f <= fb["f_hi"] + 1e-9)),
                       "offered_decode_tok_s": lam * mean_decode,
                       "realized_reqs_s": res["realized_rate"],
                       "arrival_cv": res["arrival_cv"], "f": f, **st}
                points.append(rec)
                row_for_load.append(rec)
                print(f"    load {lam:6.1f} req/s"
                      + (f" (u={u:.2f})" if u is not None else "")
                      + f"  f={f:4.2f}  P={st['power_w']:7.1f} W  "
                      f"N={st['running']:6.1f}  wait={st['waiting']:7.1f}  "
                      f"lit L/S={st['active_large']:.1f}/{st['active_small']:.1f}",
                      flush=True)
            ok = [r for r in row_for_load if r["power_w"] == r["power_w"]]
            lo = min(ok, key=lambda r: r["power_w"])
            hi = max(ok, key=lambda r: r["power_w"])
            band = {"offered_reqs_s": lam, "util": u,
                    "offered_decode_tok_s": lam * mean_decode,
                    "p_min_w": lo["power_w"], "f_at_min": lo["f"],
                    "p_max_w": hi["power_w"], "f_at_max": hi["f"],
                    "band_w": hi["power_w"] - lo["power_w"],
                    "max_waiting": max(r["waiting"] for r in ok),
                    "n_completed_this_load": len(runner.fleet.completed) - n0}
            bands.append(band)
            print(f"    -> band at {lam:.1f} req/s: "
                  f"{band['p_min_w']:.0f}..{band['p_max_w']:.0f} W "
                  f"= {band['band_w']:.0f} W "
                  f"(f {band['f_at_min']:.2f} -> {band['f_at_max']:.2f})", flush=True)
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        runner.stop()
        summary["worker_errors"] = runner.fleet.errors
        # Arithmetic independent of the instrument: no fleet can exceed n_gpus x TDP,
        # and no single lit replica can exceed one card.
        summary["sanity"] = sanity.check_power_plausibility(
            runner.trace, n_gpus=args.n_large + args.n_small)
        sanity.report(summary["sanity"], args.n_large + args.n_small)
        summary["points"] = points
        summary["bands"] = bands
        summary["drains"] = drains
        stuck = [d for d in drains if not d.get("drained")]
        # A machine-readable verdict, not just a line in the log: a point that began with
        # a backlog reads queueing, not an operating point (power can even appear to rise
        # with f while the backlog accumulates). Anything downstream should consult this.
        frac_ok = 1.0 - (len(stuck) / len(drains)) if drains else 1.0
        summary["queue"] = {
            "ok": not stuck,
            "drained_points": len(drains) - len(stuck),
            "total_points": len(drains),
            "fraction_drained": round(frac_ok, 3),
            "verdict": ("clean" if not stuck
                        else "CONTAMINATED" if frac_ok < 0.5 else "partial"),
        }
        if stuck:
            summary["drain_incomplete"] = stuck

        # WRITE THE MEASUREMENTS BEFORE PRINTING ANYTHING ABOUT THEM: a diagnostic must
        # never be able to destroy the data it is describing.
        for name, data in (("points.csv", points), ("bands.csv", bands)):
            if data:
                with open(out / name, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=sorted({k for r in data for k in r}))
                    w.writeheader()
                    w.writerows(data)

        if stuck:
            bar = "=" * 78
            # `util` is only populated on the --utils path; --loads gives absolute req/s
            # and leaves it None. Report whichever coordinate the run actually used.
            us = [d["util"] for d in stuck if d.get("util") is not None]
            lams = [d["offered_reqs_s"] for d in stuck
                    if d.get("offered_reqs_s") is not None]
            if us:
                worst = f"lowest u affected: {min(us):.2f}"
            elif lams:
                worst = f"lowest offered rate affected: {min(lams):.2f} req/s"
            else:
                worst = "affected points carry no load coordinate"
            print(f"\n{bar}\n  QUEUE {summary['queue']['verdict']}: "
                  f"{len(stuck)} of {len(drains)} points began with backlog that would not "
                  f"clear in 60 s\n  ({worst}). Their power readings carry "
                  f"queueing from the\n  previous point and are NOT operating-point "
                  f"measurements. If most points are\n  affected the offered rate is over "
                  f"capacity -- re-measure capacity, do not read this run.\n{bar}\n")
        if runner.trace:
            with open(out / "trace.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sorted({k for r in runner.trace for k in r}))
                w.writeheader()
                w.writerows(runner.trace)
        comp = runner.fleet.completed
        if comp:
            summary["latency"] = {
                "ttft_ms": metrics.percentiles([r["ttft_ms"] for r in comp]),
                "tbt_ms": metrics.percentiles([r["tbt_ms"] for r in comp]), "n": len(comp)}
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nwrote {out}")
        try:
            from analyze.plot_fleet import plot_run
            plot_run(out, "authority")
        except Exception as e:
            print(f"  (figure skipped: {type(e).__name__}: {e})")
        if bands:
            print(f"  {'u':>6} {'req/s':>9} {'tok/s':>9} {'band W':>8} {'Pmin':>7} {'Pmax':>7}")
            for b in bands:
                print(f"  {(b['util'] if b['util'] is not None else float('nan')):6.2f} "
                      f"{b['offered_reqs_s']:9.1f} {b['offered_decode_tok_s']:9.0f} "
                      f"{b['band_w']:8.0f} {b['p_min_w']:7.0f} {b['p_max_w']:7.0f}")


if __name__ == "__main__":
    main()
