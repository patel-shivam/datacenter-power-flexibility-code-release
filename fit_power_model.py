#!/usr/bin/env python
"""Fit P(b) from concurrency_sweep output, compare candidate forms, and write power_curves.json.

Login-node safe: reads CSV, fits, writes JSON and plots. No GPU.

Ranking is by **held-out** RMSE, not in-sample. In-sample error always falls as
parameters are added, so it cannot choose between a one-parameter and a four-parameter
shape; withholding every third concurrency level asks the only question that matters
for a model that will later be evaluated at fleet states nobody measured -- does this
shape interpolate to concurrencies it never saw.

    python fit_power_model.py \
        results/concurrency_sweep/*/ --out results/power_curves.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import power_model as pm


# A level that RETIRED NOTHING is not a measurement of serving power. The engine sat at
# some occupancy for the whole window and no request finished, so the watts are the watts
# of a stuck engine, and every throughput-derived quantity (goodput, energy per decode
# token, the TBT used for the residence line) is either zero or undefined.
#
# WHY THIS TEST AND NOT A KV THRESHOLD. The first version of this filter dropped levels
# with kv_usage_max >= 0.99, on the reasoning that a full block allocator means vLLM is
# preempting. Checked against the data, that rule is both too aggressive and too lax:
#
#   dropped by KV, but FINE      qwen3-8b arxiv c=192   b=153.5   396 completions
#                                qwen3-8b arxiv c=256   b=158.0   346 completions
#                                qwen3-4b 1024 c=256    b=242.5   244 completions
#                                llama31-8b 1024 c=256  b=239.1   182 completions
#                                llama31-8b 6144 c=64   b= 63.7    47 completions
#   MISSED by KV, but broken     llama32-1b 6144 c=384  kv=98.4%    0 completions
#
# A full KV pool simply PINS the batch -- and the fit is a function of the ACHIEVED b, so
# a KV-limited b is a perfectly good operating point, just a lower one than requested.
# What is not a good point is one where nothing came out. Six levels fail that test, all
# of them fixed_6144_1024 at concurrencies whose batch had already pinned (llama31-8b
# pinned at b~66 for c=96..256 -- four redundant readings of one point, none of which
# retired a request in 80 s).
def _drop_stalled(rows: list[dict], run_dir: Path, workload: str) -> list[dict]:
    """Drop levels that completed no requests. Loud, never silent."""
    keep, dropped = [], []
    for r in rows:
        try:
            n_done = int(float(r.get("n_completed") or 0))
        except (TypeError, ValueError):
            n_done = 0
        (keep if n_done > 0 else dropped).append(r)
    for r in dropped:
        # A fit that silently discards its top points reports a curve over a narrower
        # range than the caller believes they asked for.
        print(f"    DROPPED {run_dir.name[:40]} {workload} "
              f"c={r.get('concurrency_requested')} (b={float(r.get('concurrency_achieved', 0)):.1f}, "
              f"kv={float(r.get('kv_usage_max') or 0)*100:.0f}%): 0 requests completed in "
              f"{float(r.get('wall_s') or 0):.0f} s -- not a measurement of serving power")
    return keep


def load_levels(run_dir: Path):
    """Yield ``(meta, workload_name, b, p, rows)`` once per workload in the run.

    concurrency_sweep sweeps several workloads on one engine load (a cold model read from Lustre
    costs minutes), so one run directory holds several curves and they must not be
    pooled -- P(b) is workload-dependent, which is the whole reason the shape is a
    first-class part of the experiment.
    """
    with open(run_dir / "summary.json") as f:
        meta = json.load(f)["meta"]
    all_rows = list(csv.DictReader(open(run_dir / "levels.csv")))
    idle = meta.get("idle", {}).get("idle_power_w")
    if idle is None or not np.isfinite(float(idle)):
        raise ValueError(f"{run_dir}: no measured idle power; P_idle must be measured")

    names = sorted({r.get("workload") or "" for r in all_rows})
    for name in names:
        rows = [r for r in all_rows if (r.get("workload") or "") == name]
        rows = _drop_stalled(rows, run_dir, name)
        b = np.array([float(r["concurrency_achieved"]) for r in rows])
        p = np.array([float(r["avg_power_w"]) for r in rows])
        # NORMALISE THE KEY. A levels.csv without a `workload` column leaves `name` empty,
        # and the label falls back to `workload_label` -- the DESCRIPTOR,
        # "fixed_128_128(P=128,D=128)", not the name -- which would produce a second,
        # malformed key for a curve that already exists.
        # Everything downstream looks curves up as "<model>__<workload name>", so the
        # descriptor form is permanently unreachable while still sitting in
        # power_curves.json looking like data. Strip at the paren so both forms collapse
        # onto the real name and the later, fuller run wins.
        label = (name or meta.get("workload_label", "unknown")).split("(", 1)[0].strip()
        # b=0 is a real measured point and anchors the whole fit.
        yield (meta, label,
               np.concatenate([[0.0], b]), np.concatenate([[float(idle)], p]), rows)


def residence_coefficients(rows: list[dict]) -> dict:
    """Fit the per-replica residence time T(b) = A + B*b from the TBT curve.

    TBT is the time between output tokens, so a request of D decode tokens resides for
    about D*TBT(b) plus its prefill. Regressing mean TBT on the achieved batch gives
    the step-time line t_step(b) = t0 + t1*b, and A/B follow from the workload's mean
    lengths. Reported with its R^2 -- if the step time is not linear in b, that is a
    finding about the plant, not a licence to use the slope anyway.
    """
    b, tbt = [], []
    for r in rows:
        try:
            bb, tt = float(r["concurrency_achieved"]), float(r["tbt_ms_mean"])
        except (KeyError, ValueError, TypeError):
            continue
        if np.isfinite(bb) and np.isfinite(tt) and bb > 0:
            b.append(bb)
            tbt.append(tt / 1e3)
    if len(b) < 3:
        return {"ok": False}
    b, tbt = np.array(b), np.array(tbt)
    A = np.vstack([np.ones_like(b), b]).T
    (t0, t1), *_ = np.linalg.lstsq(A, tbt, rcond=None)
    pred = t0 + t1 * b
    ss_res = float(np.sum((tbt - pred) ** 2))
    ss_tot = float(np.sum((tbt - tbt.mean()) ** 2))
    return {"ok": True, "t0_s": float(t0), "t1_s": float(t1),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "n": int(len(b))}


def pack_cap_from_slo(rows: list[dict], tbt_slo_ms: float) -> dict:
    """Largest achieved concurrency whose p95 TBT still meets the SLO.

    This is where ``pack_cap`` comes from. It is deliberately a *latency* question:
    power alone would always say "pack deeper", and on a curve that relaxes past its
    peak it would say so all the way to the engine's ceiling.
    """
    ok = []
    for r in rows:
        try:
            b, tbt = float(r["concurrency_achieved"]), float(r["tbt_ms_p95"])
        except (KeyError, ValueError, TypeError):
            continue
        if np.isfinite(b) and np.isfinite(tbt) and tbt <= tbt_slo_ms:
            ok.append((b, tbt, int(float(r["concurrency_requested"]))))
    if not ok:
        return {"pack_cap": None, "reason": f"no level met p95 TBT <= {tbt_slo_ms} ms"}
    b, tbt, c = max(ok, key=lambda x: x[0])
    return {"pack_cap": c, "achieved_b": b, "tbt_p95_ms": tbt, "slo_ms": tbt_slo_ms}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="concurrency_sweep run directories")
    ap.add_argument("--out", type=Path, default=Path("results/power_curves.json"))
    ap.add_argument("--tbt-slo-ms", type=float, default=50.0)
    ap.add_argument("--holdout-every", type=int, default=3)
    ap.add_argument("--min-levels", type=int, default=8,
                    help="skip runs with fewer levels. A --smoke run has 3, which is "
                         "too few to hold any out: the held-out RMSE comes back nan and "
                         "the ranking silently falls back to in-sample error, which "
                         "always prefers the most parameters.")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    curves, report = {}, {}
    for run in args.runs:
        run = Path(run)
        if not (run / "levels.csv").exists():
            print(f"skip {run} (no levels.csv)")
            continue
        for meta, wl_name, b, p, rows in load_levels(run):
            key = f"{meta['model_slug']}__{wl_name}"
            if len(rows) < args.min_levels:
                print(f"skip {key}: {len(rows)} levels (< --min-levels {args.min_levels})")
                continue
            cmp = pm.fit_all_forms(b, p, p_idle_w=p[0], holdout_every=args.holdout_every)
            best = cmp["best"]
            if best is None:
                print(f"{key}: no form converged")
                continue
            curve = pm.PowerCurve.from_dict(cmp["forms"][best]["curve"])
            curve.model_slug = meta["model_slug"]
            curve.workload = wl_name
            curves[key] = curve

            res = residence_coefficients(rows)
            cap = pack_cap_from_slo(rows, args.tbt_slo_ms)
            b_max = int(max(b))
            peak_b = int(np.argmax(curve.power(np.arange(1, b_max + 1)))) + 1
            # A form can BELONG to a non-monotone family and still be monotone over the
            # range that was actually measured -- jump_relax fits several models with its
            # peak pushed past the top of the grid. Report what the fitted curve does
            # inside the measured range, not just which family won.
            interior_peak = peak_b < 0.95 * b_max
            r = {
                "run": str(run), "model": meta["model"], "workload": wl_name,
                "p_idle_w": float(p[0]), "p_max_measured_w": float(np.max(p)),
                "b_at_measured_peak": float(b[int(np.argmax(p))]),
                "best_form": best, "ranking": cmp["ranking_by_heldout_rmse"],
                "in_sample": cmp["forms"][best]["in_sample"],
                "held_out": cmp["forms"][best]["held_out"],
                "family_monotone": pm.FORMS[best].monotone,
                "b_at_model_peak": peak_b, "b_max_measured": b_max,
                "turns_down_within_measured_range": bool(interior_peak),
                "residence": res, "pack_cap_from_slo": cap,
                "all_forms": {k: {"held_out_rmse_w": v.get("held_out", {}).get("rmse_w"),
                                  "in_sample_rmse_w": v.get("in_sample", {}).get("rmse_w")}
                              for k, v in cmp["forms"].items() if "error" not in v},
            }
            report[key] = r
            print(f"{key}")
            print(f"  idle {r['p_idle_w']:6.1f} W   peak {r['p_max_measured_w']:6.1f} W "
                  f"at b={r['b_at_measured_peak']:.0f}   dynamic range "
                  f"{r['p_max_measured_w'] - r['p_idle_w']:.0f} W")
            print(f"  best form: {best:11s} held-out RMSE "
                  f"{r['held_out'].get('rmse_w', float('nan')):5.2f} W  "
                  f"in-sample {r['in_sample']['rmse_w']:5.2f} W  R2 {r['in_sample']['r2']:.4f}"
                  f"{'   [turns down at b=%d]' % peak_b if interior_peak else ''}")
            print(f"  ranking: {', '.join(cmp['ranking_by_heldout_rmse'])}")
            if res.get("ok"):
                print(f"  step time t_step(b) = {res['t0_s']*1e3:.3f} + "
                      f"{res['t1_s']*1e3:.4f}*b ms (R2 {res['r2']:.4f})")
            print(f"  pack_cap from p95 TBT <= {args.tbt_slo_ms:.0f} ms: "
                  f"{cap.get('pack_cap')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pm.save_curves(curves, args.out)
    rep_path = args.out.with_name(args.out.stem + "_report.json")
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nwrote {args.out} ({len(curves)} curves) and {rep_path}")

    if args.plot and curves:
        _plot(args.runs, curves, args.out.with_suffix(".png"))


def _plot(runs, curves, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for run in runs:
        run = Path(run)
        if not (run / "levels.csv").exists():
            continue
        for meta, wl_name, b, p, rows in load_levels(run):
            key = f"{meta['model_slug']}__{wl_name}"
            if len(rows) < args.min_levels:
                print(f"skip {key}: {len(rows)} levels (< --min-levels {args.min_levels})")
                continue
            c = curves.get(key)
            axes[0].plot(b, p, "o", ms=4, label=key)
            if c is not None:
                g = np.linspace(0, max(b), 400)
                axes[0].plot(g, c.power(g), "-", lw=1.2, alpha=0.8)
            tp = [float(r["throughput_decode_tok_s"]) for r in rows]
            axes[1].plot([float(r["concurrency_achieved"]) for r in rows], tp, "o-", ms=4,
                         label=key)
    axes[0].set(xlabel="running sequences b", ylabel="power (W)",
                title="P(b): points measured, lines fitted")
    axes[1].set(xlabel="running sequences b", ylabel="decode tok/s",
                title="throughput vs concurrency")
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
