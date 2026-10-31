#!/usr/bin/env python
"""Figures for the fleet experiments: knob authority (fleet_routing_sweep), power tracking, packing A/B.

Each kind of run gets the plot its data actually calls for:

**knob authority** (`--authority`, an fleet_routing_sweep run) -- two views. Power against routing
fraction (one line per offered load, ordinal ramp since load is ordered), and the band
width against offered load as a fraction of MEASURED fleet capacity, which is the
headline: how much power the knob can move, and how that collapses as the fleet fills.
The capacities are in the title.

**power tracking / shedding** (`--tracking`, a run whose trace.csv carries a power
target) -- a time series. Target and measured power share one axis because they are the
*same measure*; the routing knob rides a twinned right axis over the same time, and the
per-pool power and the lit replica count get their own panels below.

**packing A/B** (`--packing`, a run with arms.csv) -- a paired comparison with the
model's prediction beside it. Arms are categorical (pack vs spread), so they take
categorical slots 1 and 2.

    python analyze/plot_fleet.py --authority results/fleet_routing_sweep/<run>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt

from analyze import capacity_lookup
from analyze import plotting as P
from lib import feasible
from lib.router import ClassSplit


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path)))


def fcol(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except (KeyError, ValueError, TypeError):
            out.append(np.nan)
    return np.array(out)


# ======================================================================================
def fig_tracking(run: Path, out: Path) -> None:
    rows = read_csv(run / "trace.csv")
    summary = json.loads((run / "summary.json").read_text())
    mode = summary.get("args", {}).get("mode", "?")
    phase = [r["phase"] for r in rows]
    keep = [i for i, p in enumerate(phase) if not p.startswith("calib")]
    if not keep:
        keep = list(range(len(rows)))
    rows = [rows[i] for i in keep]

    t = fcol(rows, "t")
    # THE POWER-TRACKING STYLE.
    #
    # The top panel: measured power in BLUE on the left axis, the routing knob in GREEN on
    # a twinned right axis over the SAME x, a dashed black target, and the measured band
    # shaded. The knob belongs on the power axis rather than below it because reading
    # power without it cannot distinguish a wobble in the plant from the controller
    # hunting, nor show the knob railed at 0 or 1.
    #
    # The lower two panels -- pool split and lit-replica count -- are per-run detail, and
    # the lit count is the actuator.
    MEAS_C, KNOB_C, BAND_C = "#4c78a8", "#54a24b", "#9ecae1"
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1.5]})

    ax = axes[0]
    band = summary.get("band", {})
    lo, hi = band.get("p_lo_w"), band.get("p_hi_w")
    if lo is not None and hi is not None:
        ax.axhspan(lo, hi, color=BAND_C, alpha=0.18, lw=0,
                   label="measured band (floor..ceiling)")
    ax.plot(t, fcol(rows, "measured_w"), "-", color=MEAS_C, lw=1.2,
            label="measured power")
    ax.plot(t, fcol(rows, "target_w"), "k--", lw=1.2, alpha=0.7, label="target")

    kax = ax.twinx()
    kax.plot(t, fcol(rows, "f"), "-", color=KNOB_C, lw=1.4, alpha=0.8,
             label="routing knob f (right axis, 0..1)")
    kax.set_ylim(-0.05, 1.05)
    kax.set_ylabel("flex->small knob f", color=KNOB_C, fontsize=9)
    kax.tick_params(axis="y", colors=KNOB_C, labelsize=8)
    kax.grid(False)

    tr = summary.get("tracking", {})
    sub = (f"RMSE {tr['rmse_w']:.0f} W · bias {tr['bias_w']:+.0f} W" if tr else "")
    P.finish(ax, ylabel="fleet power (W)")
    # Title above, legend in a row beneath it: inside the axes the legend would sit on top
    # of the knob trace, which starts pinned at f=1.0 in exactly that corner.
    fig.suptitle(f"Power tracking — {mode}   {sub}", y=0.995, fontsize=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = kax.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=4, frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(t, fcol(rows, "power_large_w"), "-", color=P.CAT[0], lw=1.4, label="large pool")
    ax.plot(t, fcol(rows, "power_small_w"), "-", color=P.CAT[1], lw=1.4, label="small pool")
    P.finish(ax, ylabel="pool power (W)", legend=True)
    ax.set_title("Where the power is drawn", fontsize=9.5)

    ax = axes[2]
    ax.plot(t, fcol(rows, "active_large"), "-", color=P.CAT[0], lw=1.4, label="large lit")
    ax.plot(t, fcol(rows, "active_small"), "-", color=P.CAT[1], lw=1.4, label="small lit")
    P.finish(ax, xlabel="time (s)", ylabel="replicas lit", legend=True)
    ax.set_title("Replicas drawing power (the actuator)", fontsize=9.5)

    P.save(fig, out)


# ======================================================================================
def _feasible(r: dict) -> bool | None:
    """The probe's own f_feasible flag. None when the run does not record the column."""
    v = r.get("f_feasible")
    if v is None or str(v).strip() == "" or str(v).strip().lower() == "none":
        return None
    return str(v).strip().lower() in ("true", "1")


def _capacity_feasible(cap: dict, lam: float, f: float) -> bool:
    """Is f servable at lam, on THIS figure's measured capacity (lib.feasible.band)?"""
    b = feasible.band(lam, ClassSplit(*cap["class_split"]),
                      cap["n_large"] * cap["c_large"], cap["n_small"] * cap["c_small"])
    return bool(b["feasible"] and b["f_lo"] - 1e-9 <= f <= b["f_hi"] + 1e-9)


def _key(r: dict) -> tuple[float, float]:
    return (round(float(r["offered_reqs_s"]), 4), round(float(r["f"]), 4))


def _probe_windows(run: Path):
    """(key, 1 Hz fleet power) for each probe's averaging window, keyed (lambda, f).

    The window is exactly the one the point's mean was taken over: `fleet_driver.steady`
    averages the last half of a probe, and an fleet_routing_sweep trace holds only drive rows -- one
    phase per probe, all the same length -- so the last half of a phase's rows is that
    window. Nothing when the run has no trace.
    """
    f = run / "trace.csv"
    if not f.exists():
        return
    by: dict[str, list[dict]] = {}
    for r in csv.DictReader(open(f)):
        by.setdefault(r.get("phase", ""), []).append(r)
    for rows in by.values():
        try:
            key = _key(rows[0])
        except (KeyError, ValueError, TypeError):
            continue
        w = fcol(rows[max(1, int(len(rows) * 0.5)):], "measured_w")
        w = w[np.isfinite(w)]
        if len(w) >= 3:
            yield key, w


def probe_spread(run: Path) -> dict[tuple[float, float], tuple[float, float]]:
    """(p5, p95) of 1 Hz fleet power in each probe's averaging window (`_probe_windows`), keyed
    (lambda, f). Empty when the run has no trace."""
    return {k: (float(np.percentile(w, 5)), float(np.percentile(w, 95))) for k, w in _probe_windows(run)}


def probe_sd(run: Path) -> dict[tuple[float, float], float]:
    """Standard deviation of 1 Hz fleet power in each probe's averaging window (`_probe_windows`),
    keyed (lambda, f). Empty when the run has no trace."""
    return {k: float(np.std(w, ddof=1)) for k, w in _probe_windows(run)}


def band_whiskers(rows: list[dict], spread: dict) -> tuple[float, float, float] | None:
    """(band, low, high) for max - min of mean power over `rows`.

    low / high are the same band with its two END probes read at their 5th / 95th
    percentiles: low = p5(top) - p95(bottom), high = p95(top) - p5(bottom). A band is a
    difference of two probes, so that is the spread that matters; whiskers are nan when
    the run has no trace. None with fewer than two probes -- a band needs two.
    """
    if len(rows) < 2:
        return None
    top = max(rows, key=lambda r: float(r["power_w"]))
    bot = min(rows, key=lambda r: float(r["power_w"]))
    band = float(top["power_w"]) - float(bot["power_w"])
    st, sb = spread.get(_key(top)), spread.get(_key(bot))
    if not (st and sb):
        return band, np.nan, np.nan
    return band, max(0.0, st[0] - sb[1]), st[1] - sb[0]


def band_sd(rows: list[dict], sd: dict) -> float:
    """Standard deviation of the band (max - min of mean power over `rows`) from the 1 Hz power of its
    two END probes, the same two `band_whiskers` reads: sqrt(sd_top^2 + sd_bottom^2), as for the
    difference of two independent readings. nan with fewer than two probes or without a trace."""
    if len(rows) < 2:
        return float("nan")
    top = max(rows, key=lambda r: float(r["power_w"]))
    bot = min(rows, key=lambda r: float(r["power_w"]))
    st, sb = sd.get(_key(top)), sd.get(_key(bot))
    if st is None or sb is None:
        return float("nan")
    return float(np.hypot(st, sb))


AUTHORITY_CAPTION = (
    "Whiskers — left: 5th–95th percentile of 1 Hz fleet power over the window each point "
    "averages (the last half of its probe). Right: the band recomputed with its two end "
    "probes read at their 5th / 95th percentiles.\nHoldable band = max − min of power over "
    "the routing fractions both pools can sustain at that load; the hollow markers are the "
    "fractions that overload a pool, and are left out of it.")


def fig_authority(run: Path, out: Path) -> None:
    """Power against f per load, and the holdable band against load.

    `bands.csv` records the `drained` definition: max - min over EVERY routing fraction,
    including fractions the fleet cannot serve. At those fractions a pool is overloaded:
    its waiting list grows and its power sits flat at the saturation level, because a
    saturated pool cannot draw more. Counting them, the band can rise with load while the
    band over feasible fractions collapses -- the opposite sign.

    So every probe is drawn -- nothing is hidden -- but infeasible ones are hollow and
    faded, and the right panel carries the band over feasible fractions only. The
    feasible one is the claim.

    LOAD IS A FRACTION OF MEASURED FLEET CAPACITY, u = lambda / (nL cL + nS cS), with the
    capacities in the title. Offered decode tok/s is not comparable across fleets or
    workloads, and u is what every claim here is stated in. The capacity is the MEASURED
    per-replica one (analyze/capacity_lookup: this run's hardware, latest bisection,
    floor-judged files re-judged), never the analytic one the run recorded. Feasibility
    is recomputed from the SAME capacity, so the hollow markers and the axis cannot
    disagree -- a run's own `f_feasible` was computed from the capacities the run was
    planned with, which can differ from the measured ones (the title counts the probes
    where they disagree). Where no measured capacity exists the axis falls back to req/s
    and the run's flags.

    WHISKERS, and the card. Each point carries the 5th-95th percentile of the 1 Hz power it
    was averaged from (`probe_spread`), and each band the spread of its two end probes
    (`band_whiskers`) -- a toggling replica or a probe still ramping shows up as a long
    whisker rather than hiding inside a mean. The title names the card, because what the
    knob moves differs in kind between them: an H100 replica's power rises with batch, an
    RTX 8000's is close to on/off.
    """
    points = read_csv(run / "points.csv")
    loads = sorted({float(r["offered_reqs_s"]) for r in points})
    cap = capacity_lookup.fleet_capacity(run)
    C = cap["fleet_reqs_s"] if cap and cap["fleet_reqs_s"] > 0 else None
    spread = probe_spread(run)
    summ = json.loads((run / "summary.json").read_text())
    a = summ.get("args") or {}
    hw_name, hw_note = P.hardware_label(capacity_lookup.hardware_of(run.resolve().parent.parent))

    def feas(r):
        if C is None:
            return _feasible(r)
        return _capacity_feasible(cap, float(r["offered_reqs_s"]), float(r["f"]))

    def x_of(lam: float) -> float:
        return lam / C if C else lam

    has_flag = any(feas(r) is not None for r in points)
    n_differ = sum(1 for r in points
                   if C and _feasible(r) is not None and _feasible(r) != feas(r))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

    ax = axes[0]
    if len(loads) > P.MAX_ORDINAL:
        idx = np.linspace(0, len(loads) - 1, P.MAX_ORDINAL).round().astype(int)
        loads_shown = [loads[i] for i in sorted(set(idx))]
    else:
        loads_shown = loads
    for lam, c in zip(loads_shown, P.ordinal_colors(len(loads_shown))):
        sub = [r for r in points if abs(float(r["offered_reqs_s"]) - lam) < 1e-9]
        sub.sort(key=lambda r: float(r["f"]))
        ax.plot(fcol(sub, "f"), fcol(sub, "power_w"), "-", color=c, lw=1.2, alpha=0.55,
                label=(f"u = {lam / C:.2f}  ({lam:.4g} req/s)" if C else f"{lam:.4g} req/s"))
        ok = [r for r in sub if feas(r) is not False]
        bad = [r for r in sub if feas(r) is False]
        es = [(float(r["f"]), float(r["power_w"]), spread[_key(r)])
              for r in sub if _key(r) in spread]
        if es:
            ax.errorbar([e[0] for e in es], [e[1] for e in es],
                        yerr=[[max(0.0, p - s[0]) for _, p, s in es],
                              [max(0.0, s[1] - p) for _, p, s in es]],
                        fmt="none", ecolor=c, elinewidth=0.9, capsize=2, alpha=0.55,
                        zorder=1)
        if ok:
            ax.plot(fcol(ok, "f"), fcol(ok, "power_w"), "o", color=c, ms=5)
        if bad:
            ax.plot(fcol(bad, "f"), fcol(bad, "power_w"), "o", mfc="none", mec=c,
                    ms=5, alpha=0.6)
    title = "What the knob reaches, per offered load"
    if has_flag:
        title += "  (hollow = infeasible f)"
    P.finish(ax, xlabel="routing fraction to small pool  f", ylabel="fleet power (W)",
             title=title)
    # Below the axes, not "best": the top load's curve crosses every corner of this
    # panel, and a five-entry legend inside it would sit on those points.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False,
              fontsize=8)

    ax = axes[1]
    # Both bands from the probes themselves (bands.csv's band_w is the same max - min over
    # every f), so each band's whisker comes from the two probes that set it.
    by_all: dict[float, list[dict]] = {}
    by_ok: dict[float, list[dict]] = {}
    for r in points:
        if float(r["power_w"]) == float(r["power_w"]):          # not NaN
            by_all.setdefault(float(r["offered_reqs_s"]), []).append(r)
            if feas(r):
                by_ok.setdefault(float(r["offered_reqs_s"]), []).append(r)

    def draw_band(groups, color, label, lw, ms):
        xs, ys, lo, hi = [], [], [], []
        for lam in sorted(groups):
            bw = band_whiskers(groups[lam], spread)
            if bw is not None:
                xs.append(x_of(lam))
                ys.append(bw[0])
                lo.append(bw[1])
                hi.append(bw[2])
        if not xs:
            return
        ax.plot(xs, ys, "-o", color=color, ms=ms, lw=lw, label=label)
        w = [i for i in range(len(xs)) if np.isfinite(lo[i])]
        if w:
            ax.errorbar([xs[i] for i in w], [ys[i] for i in w],
                        yerr=[[max(0.0, ys[i] - lo[i]) for i in w],
                              [max(0.0, hi[i] - ys[i]) for i in w]],
                        fmt="none", ecolor=color, elinewidth=1.0, capsize=3, alpha=0.7)

    # HOLDABLE ONLY. The max - min over every f ("published", bands.csv) is not drawn: it
    # counts fractions that overload a pool, where power sits at a saturation limit rather
    # than at an operating point, so it can move against load with the opposite sign.
    # bands.csv records it.
    if has_flag:
        draw_band(by_ok, P.CAT[0], "holdable: feasible f only", 2, 5)
    else:
        ax.text(0.5, 0.5, "no measured capacity for this cell and no feasibility flags\n"
                "in the run -- the holdable band cannot be computed",
                transform=ax.transAxes, ha="center", va="center", fontsize=9, color=P.MUTED)
    P.finish(ax, ylabel="holdable band (W)",
             title="Knob authority vs offered load  (feasible f only)",
             xlabel=("offered load  u = λ / fleet capacity" if C else
                     "offered load (req/s) -- no measured capacity for this cell"))
    ax.margins(x=0.10, y=0.18)
    if C:
        # The whole utilization range, always: every authority figure then shares one x
        # scale, a single-load run is not auto-zoomed into a sliver that reads as a sweep,
        # and how far toward capacity a sweep went is visible at a glance.
        top = max([x_of(lam) for lam in by_all] or [1.0])
        ax.set_xlim(0.0, max(1.0, 1.05 * top))

    # The card, the fleet, the capacities the u axis and the hollow markers rest on, and the
    # per-replica concurrency. Decode tok/s is req/s x this workload's mean decode length,
    # taken from the run's own points.
    nL, nS = a.get("n_large"), a.get("n_small")
    # The class split, whenever any traffic is pinned: a class-mix arm and its all-flexible
    # baseline would otherwise be indistinguishable by title, and f steers only the flexible share.
    try:
        L_, S_, F_ = (float(x) for x in str(a.get("class_split", "0,0,1")).split(","))
    except ValueError:
        L_, S_, F_ = 0.0, 0.0, 1.0
    split = ("" if F_ >= 0.9995 else
             f"  ·  traffic {L_:.1%} large-only / {S_:.1%} small-only / {F_:.1%} flexible "
             f"(f steers the flexible share only)")
    lines = [f"{hw_name}  ·  {nL}L+{nS}S  {a.get('large')} / {a.get('small')}  on "
             f"{a.get('workload')}" + split]
    if C:
        mean_dec = (float(points[0]["offered_decode_tok_s"])
                    / float(points[0]["offered_reqs_s"]))
        roles = f"{cap['n_large']} × {cap['c_large']:.2f} req/s ({cap['large']})"
        if cap["n_small"]:
            roles += f"  +  {cap['n_small']} × {cap['c_small']:.2f} req/s ({cap['small']})"
        lines.append(f"fleet capacity {C:.1f} req/s  ≈  {C * mean_dec:,.0f} decode tok/s  =  "
                     f"{roles}, measured ({cap['rule']})")
    engine = summ.get("concurrency_caps") or []
    caps = (f"concurrency per replica: large {a.get('concurrency_large')}, small "
            f"{a.get('concurrency_small')} (router pack cap)")
    if engine:
        caps += f", engine max_num_seqs {engine[0]} / {engine[-1]}"
    lines.append(caps + (f"  ·  {hw_note}" if hw_note else ""))
    if n_differ:
        lines.append(f"hollow markers use these capacities; {n_differ} of {len(points)} "
                     f"probes differ from the run's own f_feasible")
    fig.suptitle("\n".join(lines), fontsize=9.5, color=P.INK_2, y=1.0)
    fig.text(0.01, -0.005, AUTHORITY_CAPTION, ha="left", va="top", fontsize=8,
             color=P.MUTED)

    P.save(fig, out)


# ======================================================================================
def fig_packing(run: Path, out: Path) -> None:
    arms = read_csv(run / "arms.csv")
    summary = json.loads((run / "summary.json").read_text())
    res = summary.get("result", {})
    pred = summary.get("prediction", {})

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # per-repeat power, arms interleaved
    ax = axes[0]
    for policy, c in (("pack", P.CAT[0]), ("spread", P.CAT[1])):
        sub = [r for r in arms if r["policy"] == policy]
        x = [int(r["repeat"]) for r in sub]
        ax.plot(x, fcol(sub, "power_w"), "-o", color=c, ms=6, label=policy)
    ax.set_xticks(sorted({int(r["repeat"]) for r in arms}))
    P.finish(ax, xlabel="repeat (arms interleaved)", ylabel="fleet power (W)",
             title="Pack vs spread, matched load", legend=True)
    ax.margins(y=0.2)

    # replicas lit -- the mechanism
    ax = axes[1]
    for policy, c in (("pack", P.CAT[0]), ("spread", P.CAT[1])):
        sub = [r for r in arms if r["policy"] == policy]
        ax.plot([int(r["repeat"]) for r in sub], fcol(sub, "active_large"),
                "-o", color=c, ms=6, label=policy)
    ax.set_xticks(sorted({int(r["repeat"]) for r in arms}))
    P.finish(ax, xlabel="repeat", ylabel="large replicas lit",
             title="The mechanism: how many replicas draw power", legend=True)
    ax.margins(y=0.25)

    # measured vs predicted saving
    ax = axes[2]
    # Slot 1 for the measurement, muted grey for the model's prediction. NOT slot 2:
    # in the other panels slot 2 means "spread", and reusing it here for "predicted"
    # would make the same colour mean two unrelated things in one figure. A prediction
    # is a reference line against the data, not a peer category to it.
    labels, vals, colors = ["measured"], [res.get("measured_saving_w", np.nan)], [P.CAT[0]]
    if pred.get("available"):
        labels.append("model predicted")
        vals.append(pred.get("predicted_saving_w", np.nan))
        colors.append(P.AXIS)
    bars = ax.bar(labels, vals, color=colors, width=0.5)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.0f} W", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=8.5, color=P.INK_2)
    ax.grid(axis="x", visible=False)
    P.finish(ax, ylabel="power saved by packing (W)",
             title="Saving from packing: measured vs predicted")
    ax.margins(y=0.2)

    P.save(fig, out)


def plot_run(run: Path, kind: str) -> Path | None:
    """Render the figure for ONE run, into that run's own directory.

    Experiments call this as their last step, so every result directory is
    self-contained: the CSVs and the picture of them sit together, and a run is
    readable without first working out which joint figure it fed into.
    """
    P.apply_style()
    run = Path(run)
    try:
        if kind == "tracking" and (run / "trace.csv").exists():
            out = run / "figure.png"
            fig_tracking(run, out)
            return out
        if kind == "authority" and (run / "points.csv").exists():
            out = run / "figure.png"
            fig_authority(run, out)
            return out
        if kind == "packing" and (run / "arms.csv").exists():
            out = run / "figure.png"
            fig_packing(run, out)
            return out
    except Exception as e:
        # A plotting failure must never lose the measurement that produced it.
        print(f"  (figure for {run.name} failed: {type(e).__name__}: {e})")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracking", type=Path, default=None)
    ap.add_argument("--authority", type=Path, default=None)
    ap.add_argument("--packing", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = ap.parse_args()

    P.apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.tracking:
        fig_tracking(args.tracking, args.out_dir / f"tracking__{args.tracking.name}.png")
    if args.authority:
        fig_authority(args.authority, args.out_dir / f"routing_sweep__{args.authority.name}.png")
    if args.packing:
        fig_packing(args.packing, args.out_dir / f"packing__{args.packing.name}.png")


if __name__ == "__main__":
    main()
