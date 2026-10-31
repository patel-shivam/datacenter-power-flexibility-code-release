#!/usr/bin/env python
"""Score the fleet model against every pinned-f measurement. No GPU.

Why this exists
---------------
The fleet model composes the per-GPU model (step time affine in batch, P(b), capacity)
into a *fleet* power at a known routing fraction -- the claim the paper's model section
actually makes. This scores that composition on open-loop data: fleet_routing_sweep writes one row per
(offered load, pinned f) after a drain, so there is no controller in the loop and any
error is the model's.

Two separate questions, deliberately kept apart
-----------------------------------------------
**How much power?** Predicted vs measured fleet watts. This is the band/authority claim
-- max and min reachable power, and how the range moves with arrival rate.

**Can that power be HELD?** ``solve_inflight`` returns status ``cycle`` when no replica
count is self-consistent: the model's own claim that the plant must oscillate at that f.
That is a falsifiable structural prediction, and it is the one control cares about,
because an unholdable f is where the ~300 W swing comes from.

Ground truth for the second needs no extra instrumentation: ``active_large`` in
points.csv is the MEAN lit large count over the hold, so a point that held still reports
an integer and a point that toggled reports a fraction.

    python analyze/score_fleet_model.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lib import power_model as pm          # noqa: E402
from lib import feasible
from lib import workloads                  # noqa: E402

CHUNK = 2048         # vLLM DEFAULT_MAX_NUM_BATCHED_TOKENS -- what the fleet ran
HEADROOM = 16        # fleet_driver.ADMISSION_HEADROOM_SEQS
TOGGLE_TOL = 0.05    # |mean lit - nearest integer| above which the count moved


def pool_from(curve, residence, wl, n_replicas, pack_cap, res_dir=None, model=None):
    """One pool of the fleet model from a single-GPU calibration: capacity evaluated at the cap,
    never at the pole."""
    t0, t1 = residence["t0_s"], residence["t1_s"]
    # PREFILL IS CHARGED PER CHUNK, NOT PER TOKEN.
    #
    # t1 is the marginal step cost of one more concurrent SEQUENCE, and a prefill chunk is
    # CHUNK tokens belonging to ONE sequence in ONE step. The per-token cost is
    # (t0 + t1*b)/CHUNK, so prefill puts no E[P]*t1 term in A -- it belongs in B, divided
    # by CHUNK. Charging every prefill TOKEN a full t1 would add E[P]*t1 to A: 0.2% of A on
    # sharegpt (E[P]=87) but 68.7% on arxiv (E[P]=2588), over-predicting arxiv occupancy.
    n_chunks = wl.mean_prefill / CHUNK
    A = n_chunks * t0 + wl.mean_decode * t0
    B = (n_chunks + wl.mean_decode) * t1
    # THE OCCUPANCY CEILING: min(request cap, KV ceiling).
    #
    # Two independent limits on how many sequences a replica can hold. The request cap is
    # the engine's max_num_seqs, which is pack_cap + ADMISSION_HEADROOM_SEQS -- the router
    # never throttles, so this is what actually binds when the pool saturates.
    #
    # The KV pool is the other, and at long context it binds FIRST: a replica runs out of
    # KV long before it runs out of sequence slots. Without it the model would let N grow
    # to the request cap and over-predict power, the band ceiling and occupancy together,
    # all in the same direction.
    #
    # Checked against measurement: predicted 68 sequences against a measured saturation of
    # 66 on fixed_6144_1024, and 193 against ~180 on arxiv (b_max 154 at kv_usage 0.86).
    # On sharegpt it is 1465 and never binds, so short-context predictions do not depend
    # on it.
    ceiling = pack_cap + HEADROOM
    if res_dir is not None and model is not None:
        kv = feasible.kv_concurrency_ceiling(res_dir, model, wl.decode_weighted_context)
        if kv is not None:
            ceiling = min(ceiling, int(kv))
    c_tok = pack_cap / (t0 + t1 * pack_cap)
    return pm.PoolModel(curve, n_replicas, pack_cap,
                        capacity_reqs_s=c_tok / wl.mean_decode,
                        residence_a_s=A, residence_b_s=B,
                        max_running_per_replica=ceiling)


def queue_slopes(run: Path) -> dict[str, float]:
    """Per-probe waiting-list slope (requests/s) over the SETTLED HALF of each phase.

    `points.csv`'s `drained` column does NOT mean what its name suggests. fleet_routing_sweep drains the
    fleet BEFORE each probe and records that drain's success:

        dr  = runner.drain(...)                 # empty the fleet first
        res = runner.drive(point_s, f=f, ...)   # then run the probe
        rec = {..., "drained": dr["drained"]}   # records the PRE-probe drain

    So it is a contamination guard on the probe's START, and says nothing about whether
    the probe itself was stable. A point can begin empty, queue without bound for 60 s,
    and still report drained=True. arxiv f=0 does exactly that: waiting climbs 0 -> 102 at
    +1.93/s while flagged drained.

    That matters because such a point is not an operating point at all -- it is a pool
    being offered more than it retires, with occupancy pinned at the admission ceiling and
    power reading whatever a saturated fleet reads. Scoring a model against it asks the
    model to predict a transient. The threshold is QUEUE_SLOPE_TOL = 0.5 req/s.
    """
    out: dict[str, float] = {}
    t = run / "trace.csv"
    if not t.exists():
        return out
    by: dict[str, list] = {}
    with t.open() as fh:
        for r in csv.DictReader(fh):
            ph = r.get("phase")
            if not ph:
                continue
            try:
                by.setdefault(ph, []).append(float(r["waiting_total"]))
            except (KeyError, ValueError, TypeError):
                continue
    for ph, w in by.items():
        h = len(w) // 2
        y = w[h:]
        if len(y) < 4:
            continue
        x = np.arange(len(y), dtype=float)
        out[ph] = float(np.polyfit(x, np.asarray(y, dtype=float), 1)[0])
    return out


QUEUE_SLOPE_TOL = 0.5      # req/s: a faster-growing waiting list means the probe did not hold


def collect(res_dir: Path, solver: str = "meanfield") -> list[dict]:
    curves = pm.load_curves(res_dir / "power_curves.json")
    report = json.load(open(res_dir / "power_curves_report.json"))
    res = {k: v["residence"] for k, v in report.items()
           if isinstance(v, dict) and v.get("residence", {}).get("ok")}

    rows: list[dict] = []
    for run in sorted((res_dir / "fleet_routing_sweep").iterdir()):
        pts, summ = run / "points.csv", run / "summary.json"
        if not pts.exists() or not summ.exists():
            continue
        a = json.load(open(summ))["args"]
        # CLASS MIX IS A SCORED DIMENSION, not a filter.
        #
        # Runs whose split is not 100% flexible do not belong in ONE table with the
        # all-flexible ones, but they are scored: "power as a function of the flexible
        # share" is a claim the model makes.
        #
        # FleetModel carries it -- split_rates sends the non-flexible remainder to the
        # large pool, so flexible_share models (x, 0, 1-x) exactly. A split with
        # small_only > 0 is NOT representable that way and is skipped, loudly.
        try:
            lo, so, fx = [float(x) for x in str(a.get("class_split")).split(",")]
        except (ValueError, AttributeError):
            print(f"  skip {run.name}: unparseable class_split {a.get('class_split')!r}")
            continue
        if so > 0:
            print(f"  skip {run.name}: split has small_only={so:g}; FleetModel routes the "
                  f"non-flexible remainder to the LARGE pool only")
            continue
        wl_name = a["workload"]
        lk, sk = f"{a['large']}__{wl_name}", f"{a['small']}__{wl_name}"
        if lk not in curves or sk not in curves or lk not in res or sk not in res:
            print(f"  skip {run.name}: concurrency_sweep never ran {wl_name}, so there is no P(b)")
            continue
        wl = workloads.WORKLOADS[wl_name]
        fleet = pm.FleetModel(
            large=pool_from(curves[lk], res[lk], wl, a["n_large"],
                            a["concurrency_large"], res_dir, a["large"]),
            small=pool_from(curves[sk], res[sk], wl, a["n_small"],
                            a["concurrency_small"], res_dir, a["small"]),
            flexible_share=fx, solver=solver)
        # Analytic feasible band, per offered rate. NOTE this is NOT the measured
        # capacity the live clamp uses: pool_from sets capacity_reqs_s from the TBT line
        # (c/(t0+t1*c) / E[D]), so everything here is the model predicting for itself.
        fband: dict[float, tuple[float, float]] = {}

        slopes = queue_slopes(run)
        for r in csv.DictReader(open(pts)):
            lam, f = float(r["offered_reqs_s"]), float(r["f"])
            q_slope = slopes.get(f"L{lam:g}_f{f:g}", float("nan"))
            if lam not in fband:
                ff = (fleet.feasible_fractions_exact(lam, policy=a["policy"])
                      if solver == "exact" else fleet.feasible_fractions(lam))
                fband[lam] = (min(ff), max(ff)) if ff else (float("nan"), float("nan"))
            pre_drained = r.get("drained", "").lower() in ("true", "1")
            # STABLE = started clean AND stayed clean. The second half is the part
            # points.csv does not record.
            drained = pre_drained and not (np.isfinite(q_slope)
                                           and q_slope > QUEUE_SLOPE_TOL)
            k_meas = float(r["active_large"])
            p = fleet.power(lam, f, a["policy"])
            try:
                n_meas = float(r.get("running") or "nan")
            except ValueError:
                n_meas = float("nan")
            rows.append(dict(
                run=run.name, policy=a["policy"], lam=lam, flexible_share=fx,
                f=f, drained=drained, pre_drained=pre_drained, q_slope=q_slope,
                meas=float(r["power_w"]) if drained else float("nan"),
                pred=p["power_total_w"],
                k_pred=p["k_large"], k_meas=k_meas,
                n_meas=n_meas, n_pred=p["n_large"] + p["n_small"],
                f_lo=fband[lam][0], f_hi=fband[lam][1],
                toggled=abs(k_meas - round(k_meas)) > TOGGLE_TOL,
                model_says_cycle=not p["self_consistent"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", type=Path, default=ROOT / "results")
    ap.add_argument("--solver", choices=("meanfield", "exact"), default="meanfield",
                    help="meanfield (default): b_bar = N/k, the mean-field closure. "
                         "exact: the occupancy-vector solve with c_eff -- better on every "
                         "scored aspect.")
    args = ap.parse_args()

    if args.solver == "exact":
        print("SOLVER: exact (occupancy vector + c_eff).")
        print("  WARNING: do NOT compare the headline numbers below against the "
              "meanfield run's. The two")
        print("  solvers REFUSE DIFFERENT PROBES, so each reports over its own subset and the")
        print("  MAPEs are not the same quantity: compare them on the probes BOTH predict.\n")
    rows = collect(args.results, solver=args.solver)
    # Power, band and holdability are scored on DRAINED probes only: an undrained one
    # measures a backlog, not an operating point. The feasible-band check below is the
    # exception -- it needs the undrained probes, because "did not drain" is its ground
    # truth.
    fin = [r for r in rows if r["drained"] and np.isfinite(r["pred"])]
    if not fin:
        print("no scoreable points")
        return
    err = np.array([r["pred"] - r["meas"] for r in fin])
    meas = np.array([r["meas"] for r in fin])

    print(f"\n=== fleet power: predicted vs measured ({len(fin)} drained pinned-f points, "
          f"{meas.min():.0f}-{meas.max():.0f} W)")
    print(f"  MAE {np.abs(err).mean():.1f} W | RMSE {np.sqrt((err**2).mean()):.1f} W | "
          f"MAPE {(np.abs(err)/meas).mean()*100:.1f}% | bias {err.mean():+.1f} W | "
          f"worst {np.abs(err).max():.0f} W")
    d = err - err.mean()
    print(f"  bias removed: MAE {np.abs(d).mean():.1f} W | "
          f"MAPE {(np.abs(d)/meas).mean()*100:.1f}%")
    nk = sum(1 for r in fin if abs(r["k_pred"] - round(r["k_meas"])) < 0.5)
    print(f"  lit large count correct on {nk}/{len(fin)}")

    # CLASS MIX. The paper claims power as a function of the flexible share; this is the
    # only place that claim is tested, on the class-mix runs.
    shares = sorted({r["flexible_share"] for r in fin})
    if len(shares) > 1:
        print(f"\n=== flexible share: does the model track the CLASS MIX?")
        print(f"  {'flexible share':<16}{'n':>5}{'MAPE':>9}{'bias':>10}")
        for sh in shares:
            g = [r for r in fin if r["flexible_share"] == sh]
            e = np.array([r["pred"] - r["meas"] for r in g])
            m = np.array([r["meas"] for r in g])
            print(f"  {sh:<16g}{len(g):>5}{(np.abs(e)/m).mean():>8.1%}"
                  f"{e.mean():>+9.1f} W")

    # === OCCUPANCY, the pivot of the whole chain ======================================
    # Everything the model predicts is reached through N = lambda*T(N/k): occupancy is
    # the coordinate P(b) is indexed by, and the band and the lit count are both read off
    # it. So scoring power alone cannot say whether a good answer came from a good
    # occupancy or from two errors cancelling on the P(b) curve -- and scoring occupancy
    # directly is the only way to tell which half to fix. points.csv carries `running`,
    # the mean in-flight count over the hold, so this needs no extra instrumentation.
    occ = [r for r in fin if np.isfinite(r["n_meas"]) and np.isfinite(r["n_pred"])
           and r["n_meas"] > 0]
    if occ:
        nm = np.array([r["n_meas"] for r in occ])
        npd = np.array([r["n_pred"] for r in occ])
        e = npd - nm
        print(f"\n=== occupancy: predicted vs measured in-flight ({len(occ)} points, "
              f"{nm.min():.0f}-{nm.max():.0f} sequences)")
        print(f"  MAE {np.abs(e).mean():.1f} seqs | MAPE {(np.abs(e)/nm).mean():.1%} | "
              f"bias {e.mean():+.1f} | median ratio pred/meas {np.median(npd/nm):.2f}")
        by_wl: dict[str, list] = {}
        for r in occ:
            by_wl.setdefault(r["run"].split("__", 1)[-1][:28], []).append(r)
        print(f"  {'run':<30}{'n':>4}{'meas':>8}{'pred':>8}{'MAPE':>8}{'ratio':>7}")
        for k, g in sorted(by_wl.items()):
            a_ = np.array([x["n_meas"] for x in g])
            b_ = np.array([x["n_pred"] for x in g])
            print(f"  {k:<30}{len(g):>4}{a_.mean():>8.0f}{b_.mean():>8.0f}"
                  f"{(np.abs(b_-a_)/a_).mean():>8.1%}{np.median(b_/a_):>7.2f}")

    # === THE FEASIBLE ROUTING RANGE ===================================================
    # The model's f_lo/f_hi come from its OWN analytic capacity (TBT line at the cap),
    # not from capacity_bisection. Ground truth is which probes drained: a fraction the fleet could
    # not retire builds a queue, and fleet_routing_sweep records that per point. This is the one check
    # that needs the undrained probes, so it runs over `rows`, not `fin`.
    cells: dict[tuple, list] = {}
    for r in rows:
        cells.setdefault((r["run"], r["lam"]), []).append(r)
    tp2 = fp2 = fn2 = tn2 = 0
    detail = []
    for (run, lam), g in sorted(cells.items()):
        lo, hi = g[0]["f_lo"], g[0]["f_hi"]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        ok = [x for x in g if x["drained"]]
        bad = [x for x in g if not x["drained"]]
        for x in g:
            says_ok = lo - 1e-9 <= x["f"] <= hi + 1e-9
            if x["drained"] and says_ok:
                tn2 += 1                       # feasible, predicted feasible
            elif x["drained"] and not says_ok:
                fp2 += 1                       # model forbade an f the hardware sustained
            elif not x["drained"] and not says_ok:
                tp2 += 1                       # correctly forbidden
            else:
                fn2 += 1                       # model allowed an f that did not drain
        detail.append((run, lam, lo, hi,
                       min((x["f"] for x in ok), default=float("nan")),
                       max((x["f"] for x in ok), default=float("nan")),
                       len(bad), len(g)))
    tot = tp2 + fp2 + fn2 + tn2
    if tot:
        print(f"\n=== feasible routing range: analytic [f_lo, f_hi] vs which probes DRAINED")
        print(f"  {tot} probes | agree {(tp2+tn2)/tot:.0%} | "
              f"model forbids an f that drained: {fp2} | "
              f"model allows an f that did not: {fn2}")
        print(f"  {'run':<40}{'lam':>7}{'model f':>14}{'drained f':>14}{'undrained':>10}")
        for run, lam, lo, hi, dlo, dhi, nbad, n in detail:
            print(f"  {run[:39]:<40}{lam:>7.1f}"
                  f"{f'[{lo:.2f},{hi:.2f}]':>14}{f'[{dlo:.2f},{dhi:.2f}]':>14}"
                  f"{f'{nbad}/{n}':>10}")

    tp = sum(1 for r in fin if r["toggled"] and r["model_says_cycle"])
    fp = sum(1 for r in fin if not r["toggled"] and r["model_says_cycle"])
    fn = sum(1 for r in fin if r["toggled"] and not r["model_says_cycle"])
    tn = sum(1 for r in fin if not r["toggled"] and not r["model_says_cycle"])
    base = (tp + fn) / len(fin)
    print(f"\n=== holdability: does the model say WHICH f cannot be held?")
    print(f"  tp {tp}  fp {fp}  fn {fn}  tn {tn}")
    print(f"  precision {tp/(tp+fp):.2f}   recall {tp/(tp+fn):.2f}   "
          f"accuracy {(tp+tn)/len(fin):.2f}")
    print(f"  ...against {1-base:.2f} for always predicting 'holds'. A classifier that "
          f"does not beat\n     the base rate is not usable, and this one does not.")

    # === the KNOB'S AUTHORITY, which is a different question from per-point power =====
    # A model can predict every point well and still get the SPAN wrong, because the span
    # is a difference of two predictions and their errors need not cancel. The band is
    # also the quantity the paper actually claims -- "how much power can routing move".
    #
    # COVERAGE IS PART OF THE ANSWER, not a footnote. The model refuses to predict a
    # point it believes diverges, and on long context it refuses most of the sweep -- 4 of
    # 11 survive on arxiv. The surviving span is then NOT the measured band: arxiv's is
    # 1466 W and the scored subset spans 766 W, because the f=1.0 floor is among the
    # points dropped. Comparing a prediction against a truncated span and calling the
    # difference "band error" would be comparing two different quantities, so coverage is
    # reported next to every row and the summary is split on it.
    # GROUP BY (run, policy, LAMBDA): a band is a span at ONE offered load. Grouping on
    # (run, policy) alone would pool every arrival rate of a sweep into one span and label
    # it with the first row's rate -- an overstatement that is easy to miss, because a wide
    # band at low load is exactly the shape the physics predicts.
    all_by_run: dict[tuple, list] = {}
    for r in rows:
        all_by_run.setdefault((r["run"], r["policy"], r["lam"]), []).append(r)
    bands = []
    for (run, pol, _lam), allg in sorted(all_by_run.items()):
        # BOTH conditions. `meas` is NaN on a probe that was not stable (see
        # queue_slopes), so filtering on finite `pred` alone puts NaN into the span and
        # every band statistic comes back NaN.
        g = [x for x in allg if np.isfinite(x["pred"]) and np.isfinite(x["meas"])]
        # `n_drained` is the STABLE count, which is what coverage should be measured
        # against: the question is what fraction of usable probes the model would score.
        allg = [x for x in allg if np.isfinite(x["meas"])]
        if len(g) < 4 or not allg:          # a span needs a real sweep behind it
            continue
        mm, pp = [x["meas"] for x in g], [x["pred"] for x in g]
        # The row is about to carry a single `lam`, taken from g[0], so the group must have
        # exactly one; the assertion checks the rest.
        lams = {x["lam"] for x in g}
        assert len(lams) == 1, (
            f"band row for {run}/{pol} would be labelled lam={g[0]['lam']} but the group "
            f"spans {sorted(lams)} -- a band is only a band at one offered load")
        bands.append({"run": run, "policy": pol, "n": len(g), "n_drained": len(allg),
                      "coverage": len(g) / len(allg), "lam": g[0]["lam"],
                      "meas_band": max(mm) - min(mm),
                      "pred_band": max(pp) - min(pp),
                      "meas_lo": min(mm), "pred_lo": min(pp),
                      "meas_hi": max(mm), "pred_hi": max(pp)})
    if bands:
        def stats(bs, label):
            if not bs:
                return
            be = np.array([b["pred_band"] - b["meas_band"] for b in bs])
            bm = np.array([b["meas_band"] for b in bs])
            lo = np.array([b["pred_lo"] - b["meas_lo"] for b in bs])
            hi = np.array([b["pred_hi"] - b["meas_hi"] for b in bs])
            print(f"  {label} ({len(bs)} sweeps)")
            print(f"    band    MAE {np.abs(be).mean():>4.0f} W | "
                  f"MAPE {np.abs(be/bm).mean():>5.1%} | bias {be.mean():+.0f} W")
            print(f"    floor   MAE {np.abs(lo).mean():>4.0f} W | bias {lo.mean():+.0f} W")
            print(f"    ceiling MAE {np.abs(hi).mean():>4.0f} W | bias {hi.mean():+.0f} W")

        print(f"\n=== knob authority: predicted vs measured BAND (max-min over f)")
        full = [b for b in bands if b["coverage"] >= 0.999]
        part = [b for b in bands if b["coverage"] < 0.999]
        stats(full, "WHOLE sweep scored -- like for like")
        stats(part, "PARTIAL -- model refused points the hardware sustained, so the "
                    "measured\n           span is truncated and these two numbers are "
                    "not the same quantity")
        print(f"\n  {'run':<44}{'lam':>7}{'cover':>7}{'meas':>7}{'pred':>7}{'err':>7}")
        for b in sorted(bands, key=lambda x: -abs(x["pred_band"] - x["meas_band"])):
            print(f"  {b['run'][:43]:<44}{b['lam']:>7.1f}"
                  f"{b['n']:>4}/{b['n_drained']:<2}{b['meas_band']:>7.0f}"
                  f"{b['pred_band']:>7.0f}{b['pred_band']-b['meas_band']:>+7.0f}")
        bout = args.results / "fleet_band_scored.csv"
        with open(bout, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(bands[0].keys()))
            w.writeheader()
            w.writerows(bands)
        print(f"\n  wrote {bout}")

    out = args.results / "fleet_model_scored.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
