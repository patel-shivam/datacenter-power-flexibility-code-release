#!/usr/bin/env python
"""Knob authority vs offered load for several fleet compositions -- published and holdable.

    python analyze/plot_authority_compare.py \\
        --out results/figures/authority_by_composition.png RUN [RUN ...]

Each RUN is one fleet_routing_sweep knob-authority sweep. Load is u = lambda / MEASURED fleet capacity and
feasibility is recomputed from that same capacity, exactly as `plot_fleet.fig_authority`
does for one run, so this figure and the per-run figures cannot disagree. The band a
controller can hold (feasible f only) against u; refuses to put two kinds of card on one
axis.

Colour is the SERIES -- the composition, and the workload when the runs differ in it -- and
style is the PROBE LENGTH: runs of one series at the same kind of probe are MERGED into one
line sorted by load (a wide-range sweep and a high-load one join up), drawn solid with filled
circles when the probes were too short for steady state on that card and dashed with open
squares when they were long enough. Steady state is 180 s on the RTX 8000, where a request
lives 34-65 s and 60 s probes are mid-ramp, and 30 s on the H100, whose requests live
seconds.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analyze import capacity_lookup
from analyze import plotting as P
from analyze.plot_fleet import _capacity_feasible, band_sd, band_whiskers, probe_sd, probe_spread

COMPARE_CAPTION = (
    "Holdable band = max − min of power over the routing fractions both pools can sustain\n"
    "at that load. Whiskers: the band recomputed with its two end probes read at their\n"
    "5th / 95th percentiles of 1 Hz power over the window their means were taken from.")
HELD_CAPTION = (
    "Holdable band = max − min of power over the routing fractions that are within both pools'\n"
    "measured capacity AND whose queue held on the GPU (the probe started drained and its waiting\n"
    "list grew < 0.5 req/s). Whiskers: the band recomputed with its two end probes read at their\n"
    "5th / 95th percentiles of 1 Hz power over the window their means were taken from.")


def bands(run: Path, cap: dict, require_held: bool = False, with_sd: bool = False) -> dict[str, list]:
    """Per load: u, then (band, low, high) published and holdable, as plot_fleet draws them; with
    ``with_sd`` each tuple also ends in the band's standard deviation (`plot_fleet.band_sd`).

    ``require_held`` also drops every probe whose queue did not hold on the GPU. Capacity
    arithmetic alone admits one that fails: ShareGPT u = 0.5 at f = 1.0 puts 67.1 req/s on a
    small replica rated 69.0, and its waiting list grows at 0.93 req/s -- worth 350 W of band."""
    by: dict[float, list[dict]] = {}
    for r in csv.DictReader(open(run / "points.csv")):
        if float(r["power_w"]) == float(r["power_w"]):          # not NaN
            by.setdefault(float(r["offered_reqs_s"]), []).append(r)
    spread = probe_spread(run)
    sds = probe_sd(run) if with_sd else None
    C = cap["fleet_reqs_s"]
    held = None
    if require_held:
        from analyze.plot_model_vs_load import held_probes
        held = {(lam, f) for lam, f, _ in held_probes(run)}
    out = {"pub": [], "hold": []}
    for lam in sorted(by):
        pb = band_whiskers(by[lam], spread)
        if pb:
            out["pub"].append((lam / C, *pb) + ((band_sd(by[lam], sds),) if with_sd else ()))
        rows = [r for r in by[lam]
                if _capacity_feasible(cap, lam, float(r["f"]))
                and (held is None or (lam, float(r["f"])) in held)]
        hb = band_whiskers(rows, spread)
        if hb:
            out["hold"].append((lam / C, *hb) + ((band_sd(rows, sds),) if with_sd else ()))
    return out


STEADY_S = 180.0   # probes at least this long are steady state on the slowest card run
# per card: the H100's in-flight count refills in 3-7 s, so any probe of 30 s or more is steady
STEADY_S_BY_HW = {"rtx-8000-48gb": STEADY_S}


def probe_s(run: Path) -> float | None:
    """The run's probe length (fleet_routing_sweep --point-s), None if the summary does not say."""
    try:
        a = json.loads((run / "summary.json").read_text()).get("args") or {}
        return float(a["point_s"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def composition(cap: dict) -> str:
    name = f"{cap['n_large']}L+{cap['n_small']}S"
    split = cap["class_split"]
    if split != [0.0, 0.0, 1.0]:
        name += f" ({split[0]:.0%} large-only)" if split[0] else f" (split {split})"
    return name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--note", default="", help="one caveat line for the title, e.g. probe length")
    ap.add_argument("--ustar", action="store_true",
                    help="mark each fleet's u* = n_S C_S / C_fleet, where the model puts the peak")
    ap.add_argument("--require-held", action="store_true",
                    help="a fraction counts only if it is within measured capacity AND its "
                         "queue held on the GPU (plot_model_vs_load.held_probes)")
    args = ap.parse_args()

    runs = [r.resolve() for r in args.runs]
    hw = {capacity_lookup.hardware_of(r.parent.parent) for r in runs}
    if len(hw) != 1:
        raise SystemExit(f"REFUSING: runs span {sorted(hw)} -- absolute watts across cards "
                         f"are not comparable on one axis.")
    caps = [capacity_lookup.fleet_capacity(r) for r in runs]
    if any(c is None for c in caps):
        missing = [r.name for r, c in zip(runs, caps) if c is None]
        raise SystemExit(f"REFUSING: no measured capacity for {missing}")

    P.apply_style()
    # Holdable only -- the band over every f, overloaded fractions included, is not drawn
    # (see plot_fleet.fig_authority).
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.6))
    top = 0.0
    # Whiskers are offset by a hair per series: at the lowest loads the curves sit within a
    # few watts of each other and their whiskers would otherwise coincide.
    dx = 0.006
    steady_s = STEADY_S_BY_HW.get(next(iter(hw)), 30.0)
    multi_wl = len({c["workload"] for c in caps}) > 1
    # Two model pairs on one composition and workload (e.g. the llama and qwen 3L+1S fleets on
    # 1024/1024) would otherwise share a label and be merged into one line.
    multi_pair = len({(c["large"], c["small"]) for c in caps}) > 1

    def label(cap):
        return ((f"{cap['large']} + {cap['small']} · " if multi_pair else "") + composition(cap)
                + (f" · {cap['workload']}" if multi_wl else ""))

    names = []
    for cap in caps:
        if label(cap) not in names:
            names.append(label(cap))
    series: dict[tuple, list] = {}
    # loads a run measured but that have fewer than two holdable fractions -- no band to draw,
    # and a silently shorter line would read as "not measured"
    # -- per SERIES: a load another run of the same series holds (an in-window re-probe) is
    # drawn, so it is not listed
    omitted: dict[str, list[float]] = {}
    held_by: dict[str, set] = {}
    for run, cap in zip(runs, caps):
        b = bands(run, cap, require_held=args.require_held)
        held_u = {round(p[0], 2) for p in b["hold"]}
        held_by.setdefault(label(cap), set()).update(held_u)
        omitted.setdefault(label(cap), []).extend(
            round(p[0], 2) for p in b["pub"] if round(p[0], 2) not in held_u)
        top = max([top] + [p[0] for p in b["pub"]])
        ps = probe_s(run)
        steady = ps is not None and ps >= steady_s
        series.setdefault((label(cap), steady), []).extend(b["hold"])
    kinds = {k[1] for k in series}
    for (name, steady), pts in series.items():
        if not pts:
            continue
        i = names.index(name)
        c, off = P.CAT[i], (i - (len(names) - 1) / 2) * dx
        u, y, lo, hi = zip(*sorted(pts))
        if steady:
            ax.plot(u, y, "--s", color=c, ms=6, lw=1.6, mfc=P.SURFACE, mec=c, mew=1.6,
                    zorder=3)
        else:
            ax.plot(u, y, "-o", color=c, ms=5, lw=2, zorder=3)
        w = [k for k in range(len(u)) if lo[k] == lo[k]]
        if w:
            ax.errorbar([u[k] + off for k in w], [y[k] for k in w],
                        yerr=[[max(0.0, y[k] - lo[k]) for k in w],
                              [max(0.0, hi[k] - y[k]) for k in w]],
                        fmt="none", ecolor=c, elinewidth=1.0, capsize=3, alpha=0.6)
    if args.ustar:
        marked = set()
        for cap in caps:
            n = label(cap)
            if n in marked or not cap.get("n_small") or not cap.get("c_small"):
                continue
            marked.add(n)
            us_ = cap["n_small"] * cap["c_small"] / cap["fleet_reqs_s"]
            ax.plot([us_], [0.0], marker="|", color=P.CAT[names.index(n)], ms=16, mew=2.4,
                    ls="none", zorder=4, clip_on=False)
    P.finish(ax, xlabel="offered load  u = λ / fleet capacity", ylabel="holdable band (W)",
             title="Holdable knob authority: feasible f only"
                   + (" (capacity and queue both hold)" if args.require_held else ""))
    handles = [Line2D([], [], color=P.CAT[i], lw=2, label=n) for i, n in enumerate(names)]
    if args.ustar:
        handles.append(Line2D([], [], marker="|", ls="none", color=P.INK_2, ms=12, mew=2.2,
                              label="tick on the axis: u* = n_S·C_S / C_fleet, the predicted peak"))
    if len(kinds) > 1:
        handles += [Line2D([], [], color=P.INK_2, ls="-", marker="o", ms=5, lw=1.6,
                           label=f"probes under {steady_s:.0f} s (mid-ramp on this card)"),
                    Line2D([], [], color=P.INK_2, ls="--", marker="s", ms=6, lw=1.4,
                           mfc=P.SURFACE, mec=P.INK_2,
                           label=f"probes of {steady_s:.0f} s or more (steady state)")]
    # up to four series fit in the upper-right corner (bands fall with load, so no line reaches
    # it); more than that cross the curves, so the legend goes under the axes and the footnote
    # under the legend
    if len(handles) > 4:
        leg = ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper center",
                        bbox_to_anchor=(0.5, -0.24), ncol=2)
    else:
        leg = ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper right")
    legend_below = len(handles) > 4
    ax.set_xlim(0.0, max(1.0, 1.05 * top))
    ax.margins(y=0.15)

    c0 = caps[0]
    hw_name, hw_note = P.hardware_label(hw.pop())
    pair = "model pairs as labelled" if multi_pair else f"{c0['large']} / {c0['small']}"
    if multi_wl or multi_pair:
        lines = [f"{hw_name}  ·  {pair}  ·  u against each fleet's measured capacity"]
    else:
        lines = [f"{hw_name}  ·  {c0['large']} / {c0['small']} on {c0['workload']}  ·  per "
                 f"replica {c0['c_large']:.2f} / {c0['c_small']:.2f} req/s, measured "
                 f"({c0['rule']})"]
    if hw_note:
        lines.append(hw_note)
    if args.note:
        lines.append(args.note)
    fig.suptitle("\n".join(lines), fontsize=9.5, color=P.INK_2, y=1.0)
    caption_y = -0.02
    if legend_below:
        # lay the axes out now, with the title in place (tight_layout ignores a legend outside
        # the axes), measure where the legend ends and hang the footnote under it; the save
        # must then not lay out again
        fig.tight_layout()
        fig.canvas.draw()
        bb = leg.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        caption_y = bb.y0 - 0.03
    caption = HELD_CAPTION if args.require_held else COMPARE_CAPTION
    omitted = {n: sorted(set(us) - held_by.get(n, set())) for n, us in omitted.items()}
    gone = [f"{n} at u = {', '.join(f'{u:.2g}' for u in us)}"
            for n, us in omitted.items() if us]
    if gone:
        caption += "\nNot drawn, fewer than two holdable fractions: " + "; ".join(gone) + "."
    fig.text(0.01, caption_y, caption, ha="left", va="top", fontsize=8, color=P.MUTED)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    P.save(fig, args.out, tight=not legend_below)


if __name__ == "__main__":
    main()
