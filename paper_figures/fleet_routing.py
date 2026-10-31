"""Fleet routing figures: the analytical fleet model (built from single-GPU constants only)
against the open-loop routing sweeps on 4- and 8-GPU fleets.

Main text:
  feasible_window_and_power_band    the sustainable window of the routing fraction f and the
                                    fleet-power band over it against load, model vs measured
                                    (the two 4-GPU model pairs on ShareGPT)
  power_vs_routing_fraction         fleet power against f at every load, measured vs model, and
                                    the holdable power range against load (Llama pair, ShareGPT)
Appendix:
  window_band_grid                  the first view for the five other fleets with a full load sweep
  plant_grid                        the second view for the same five fleets

Data: results/tables/fleet_routing/*.csv (analyze/plot_fleet_routing_figures.py).
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .common import P, SCI6, STYLE, black, savefig

TASK_SHORT = {"sharegpt": "ShareGPT", "humaneval": "HumanEval", "arxiv_summarization": "arXiv"}
FAMILY_COLOR = {"llama": SCI6[0], "qwen": SCI6[1]}
# The five fleets with a full load sweep that the main text does not show. The Llama 1B/8B
# (4S/4L) fleet is left out of the figures (measured at u <= 0.5 only: its u = 0.6-0.9 run is
# queue-contaminated) and kept in the appendix tables.
FIVE = ["qwen_sharegpt", "qwen_1024", "llama_arxiv", "llama_1024", "llama_sharegpt_5L3S"]


def fleet_marker(F, key):
    m = F.meta.loc[key]
    return STYLE["eight_mark"].get((int(m.n_large), int(m.n_small)), "s") if m.n_large + m.n_small > 4 \
        else STYLE["pair_mark"].get(m.pair, "o")


def pair_label(pair, wl, n_small, n_large):
    """The uniform fleet name: family small/large sizes (small/large replicas) · task."""
    fam, sizes = ("Llama", "1B/8B") if pair == "llama31-8b" else ("Qwen", "4B/8B")
    task = TASK_SHORT.get(wl, wl.replace("fixed_", "").replace("_", "/"))
    return f"{fam} {sizes} ({int(n_small)}S/{int(n_large)}L) · {task}"


def fleet_label(F, key):
    m = F.meta.loc[key]
    return pair_label(m.pair, m.workload, m.n_small, m.n_large)


def feasible_window_and_power_band(F):
    """Feasible routing window (left) and total-power band (right) against load for the two
    4-GPU model pairs on ShareGPT: the model's prediction (shaded) and the measured range at
    every load swept (bars, over the probes whose queue held)."""
    meta, grid, winb = F.meta, F.grid, F.winb
    MODELS_PLOT = ["llama", "qwen"]
    TASKS_PLOT = ["sharegpt"]  # sharegpt, 128, 256, 512, 1024, arxiv
    MIN_GAP = 0.035
    POWER_LABEL_GAP = 0.095

    LABELS = {"llama": "Llama 3.2-1B / 3.1-8B", "qwen": "Qwen3-4B / 8B"}
    PAIRS = [f"{m}_{t}" for m in MODELS_PLOT for t in TASKS_PLOT
             if f"{m}_{t}" in meta.index]
    assert PAIRS, "No matching fleets."

    fig, axes = plt.subplots(len(PAIRS), 2, figsize=(4, 3.7),
                             sharex=True, sharey="col", squeeze=False)

    for j, k in enumerate(PAIRS):
        m = k.split("_")[0]
        c = SCI6[MODELS_PLOT.index(m) % len(SCI6)]
        g = grid[grid.fleet == k].sort_values("u")
        w = winb[winb.fleet == k].sort_values("u")

        if MIN_GAP > 0 and len(w) > 1:
            selected = []
            for idx in range(len(w)):
                if not selected or w.iloc[idx].u - w.iloc[selected[-1]].u >= MIN_GAP:
                    selected.append(idx)
            w = w.iloc[selected]

        for i, (lo, hi, lm, hm) in enumerate([
            ("f_lo", "f_hi", "f_lo_meas", "f_hi_meas"),
            ("p_lo", "p_hi", "p_lo_meas", "p_hi_meas"),
        ]):
            ax = axes[j, i]
            d = g.dropna(subset=[lo, hi])
            scale = 1000 if i == 1 else 1

            ax.fill_between(d.u, d[lo] / scale, d[hi] / scale,
                            color=c, alpha=0.25, lw=0)
            ax.plot(d.u, d[lo] / scale, color=c, lw=1.2)
            ax.plot(d.u, d[hi] / scale, color=c, lw=1.2)

            for _, r in w.dropna(subset=[lm, hm]).iterrows():
                ax.plot([r.u, r.u], [r[lm] / scale, r[hm] / scale],
                        color=P.INK, lw=1, zorder=3)
                ax.scatter([r.u, r.u], [r[lm] / scale, r[hm] / scale],
                           color=P.INK, marker="_", s=16, zorder=4)

            ax.set_xlim(0, 1.05)
            if i == 0:
                ax.set_ylim(-0.02, 1.02)
                ax.set_yticks(np.arange(0, 1.01, 0.25))

            P.finish(ax, xlabel=None, ylabel=None)
            if i == 0:
                ax.set_title(LABELS.get(m, m), fontsize=9,
                             fontweight="normal", pad=8, color="black")

            ax.tick_params(axis="both", which="both", labelsize=8, colors="black")
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
            for axis in (ax.xaxis, ax.yaxis):
                axis.label.set_color("black")
                axis.get_offset_text().set_color("black")
            ax.title.set_color("black")

    fig.subplots_adjust(left=0.13, right=0.98, top=0.93, bottom=0.19,
                        hspace=0.42, wspace=0.42)

    y_center = (axes[0, 1].get_position().y1
                + axes[-1, 1].get_position().y0) / 2

    fig.supxlabel(r"Normalized arrival rate", y=0.09, fontsize=10, color="black")
    fig.text(0.015, y_center, r"Fraction of queries to small pool",
             rotation=90, va="center", ha="center", fontsize=10, color="black")
    fig.text(axes[0, 1].get_position().x0 - POWER_LABEL_GAP,
             y_center, "Total power (kW)",
             rotation=90, va="center", ha="center", fontsize=10, color="black")

    fig.legend(handles=[
        Patch(facecolor="gray", alpha=0.25, edgecolor="none",
              label="Predicted feasible range"),
        Line2D([], [], color="black", lw=1, marker="|",
               markersize=7, markeredgewidth=1,
               label="Measured feasible range"),
    ], handler_map={Line2D: HandlerLine2D(numpoints=2)},
       loc="lower center", bbox_to_anchor=(0.5, 0.015),
       ncol=2, frameon=False, fontsize=8, labelcolor="black")

    savefig(fig, "fleet_feasible_window_and_power_band_4gpu_pairs", "main")


def _loads_of(pr_all):
    """One sweep per rounded load, preferring the wide (u = 0.1-0.9) protocol over an older
    single-load sweep."""
    us = []
    for _, g in pr_all.groupby(pr_all.u.round(1)):
        best = g.groupby("u").agg(wide=("wide", "max"), n=("f", "size")).reset_index()
        us.append(float(best.sort_values(["wide", "n"], ascending=False).iloc[0].u))
    return sorted(us)


def power_vs_routing_fraction(F, key: str = "llama_sharegpt"):
    """Left: fleet power against the fraction of queries sent to the small pool at every load
    (colour), measured (solid, the feasible probes whose queue held) and model (dashed).
    Right: the holdable power range against load, measured (whiskers: +-1 standard deviation,
    from the 1 Hz power of its two end probes) and model (dashed)."""
    meta, probes, curves, grid, bandm = F.meta, F.probes, F.curves, F.grid, F.bandm
    X_LABEL_SIZE = 12
    Y_LABEL_SIZE = 12
    TICK_SIZE = 10
    LEGEND_SIZE = 12

    assert key in meta.index, f"No fleet: {key}"

    pr_all = probes[probes.fleet == key]
    us = _loads_of(pr_all)
    assert us, f"No probes: {key}"

    norm = Normalize(vmin=min(us) - 0.05, vmax=max(us) + 0.15)
    cmap = mpl.colormaps[STYLE["load_cmap"]]
    marker = fleet_marker(F, key)

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(14, 4.4),
        gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    for u in us:
        col = cmap(norm(u))
        pr = pr_all[(pr_all.u == u) & pr_all.feasible].sort_values("f")
        if len(pr):
            ax.plot(pr.f, pr.meas_w, color=col, lw=2,
                    marker=marker, markersize=6, zorder=3)

        cv = curves[
            (curves.fleet == key) & (curves.u == u)
            & curves.in_window & curves.pred_w.notna()
        ].sort_values("f")
        if len(cv):
            ax.plot(cv.f, cv.pred_w, color=col, lw=2,
                    linestyle="--", zorder=2)

    ax.set_xlim(-0.02, 1.02)
    P.finish(ax, xlabel=r"Fraction of queries to small pool", ylabel="Total power (W)")

    g = grid[grid.fleet == key].sort_values("u")
    ok = g.p_lo.notna() & g.p_hi.notna()
    axb.plot(g.u[ok], (g.p_hi - g.p_lo)[ok], color="black",
             lw=1.3, linestyle="--", zorder=2)

    b = bandm[bandm.fleet == key].sort_values("u")
    for _, r in b.iterrows():
        # whiskers: +-1 standard deviation of the band, from its two end probes' 1 Hz power
        sd = r.sd if np.isfinite(r.sd) else 0
        axb.errorbar(
            [r.u], [r.band], yerr=[[min(sd, r.band)], [sd]],
            fmt=marker, color=cmap(norm(r.u)),
            markersize=6, capsize=2, lw=2, zorder=3
        )
    if len(b):
        axb.plot(b.u, b.band, color="black", lw=2, zorder=2)

    axb.set_xlim(0, 1.05)
    axb.set_ylim(bottom=0)

    P.finish(axb, xlabel=r"Normalized arrival rate", ylabel="Power range (W)")

    # Label sizes and black axes/text
    for a in (ax, axb):
        a.set_title("")
        a.title.set_color("black")
        a.xaxis.label.set(fontsize=X_LABEL_SIZE, color="black")
        a.yaxis.label.set(fontsize=Y_LABEL_SIZE, color="black")
        a.tick_params(axis="both", which="both", colors="black", labelsize=TICK_SIZE)
        for spine in a.spines.values():
            spine.set_edgecolor("black")
        for axis in (a.xaxis, a.yaxis):
            axis.get_offset_text().set(fontsize=TICK_SIZE, color="black")

    handles = [
        Line2D([], [], color="black", lw=2, marker=marker,
               markersize=6, label="Measured"),
        Line2D([], [], color="black", lw=2, linestyle="--", label="Estimated"),
    ]

    # on the right: the load as a vertical colour bar (0.1-0.9, the curves' colours) with a two-line label beside its
    # centre, and the legend below it
    fig.tight_layout(rect=(0, 0.02, 0.8716, 1))
    lo_u, hi_u = round(min(us), 1), round(max(us), 1)
    ticks = np.round(np.arange(lo_u, hi_u + 1e-9, 0.2), 1)
    cax = fig.add_axes([0.8756, 0.36, 0.012, 0.44])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical",
                      boundaries=np.linspace(lo_u, hi_u, 257), ticks=ticks)
    cb.set_ticklabels([f"{t:.1f}" for t in ticks])
    cb.ax.tick_params(labelsize=TICK_SIZE, colors="black", length=3)
    cb.outline.set_visible(False)
    fig.text(0.9176, 0.36 + 0.44 / 2, "Normalized\narrival rate", ha="left", va="center", fontsize=LEGEND_SIZE,
             color="black", linespacing=1.3)
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=(0.8706, 0.30), ncol=1, frameon=False,
               fontsize=LEGEND_SIZE, labelcolor="black")

    savefig(fig, f"fleet_power_vs_routing_fraction_{key}", "main")


# ---- appendix ---------------------------------------------------------------------------------
def window_band_grid(F, keys=FIVE, name="fleet_feasible_window_and_power_band_grid", width=3.3, row_h=1.37, MIN_GAP=0.035):
    """The feasible window and power band against load (the main text's first fleet view) for any
    list of fleets, one row per fleet. Rows are titled with the uniform fleet name, so the figure
    is wider than the main figure's 4 in; each row keeps its own power axis (the 8-GPU fleet draws
    ~2x the power)."""
    grid, winb = F.grid, F.winb
    n = len(keys)
    H = 3.7 - 2 * row_h + n * row_h + 0.25             # the main figure is 3.7 in tall for two rows; +0.25 in for the two-row legend
    fig, axes = plt.subplots(n, 2, figsize=(width, H), sharex=True, squeeze=False)
    for j, k in enumerate(keys):
        c = FAMILY_COLOR[k.split("_")[0]]
        g = grid[grid.fleet == k].sort_values("u")
        w = winb[winb.fleet == k].sort_values("u")
        if MIN_GAP > 0 and len(w) > 1:
            selected = []
            for idx in range(len(w)):
                if not selected or w.iloc[idx].u - w.iloc[selected[-1]].u >= MIN_GAP:
                    selected.append(idx)
            w = w.iloc[selected]
        for i, (lo, hi, lm, hm) in enumerate([("f_lo", "f_hi", "f_lo_meas", "f_hi_meas"),
                                              ("p_lo", "p_hi", "p_lo_meas", "p_hi_meas")]):
            ax = axes[j, i]
            d = g.dropna(subset=[lo, hi])
            scale = 1000 if i == 1 else 1
            ax.fill_between(d.u, d[lo] / scale, d[hi] / scale, color=c, alpha=0.25, lw=0)
            ax.plot(d.u, d[lo] / scale, color=c, lw=1.2)
            ax.plot(d.u, d[hi] / scale, color=c, lw=1.2)
            for _, r in w.dropna(subset=[lm, hm]).iterrows():
                ax.plot([r.u, r.u], [r[lm] / scale, r[hm] / scale], color=P.INK, lw=1, zorder=3)
                ax.scatter([r.u, r.u], [r[lm] / scale, r[hm] / scale], color=P.INK, marker="_", s=16, zorder=4)
            ax.set_xlim(0, 1.05)
            if i == 0:
                ax.set_ylim(-0.02, 1.02)
                ax.set_yticks(np.arange(0, 1.01, 0.25))
            P.finish(ax, xlabel=None, ylabel=None)
            ax.tick_params(axis="both", which="both", labelsize=8, colors="black")
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
            for axis in (ax.xaxis, ax.yaxis):
                axis.label.set_color("black")
                axis.get_offset_text().set_color("black")
        axes[j, 0].set_title(fleet_label(F, k), fontsize=8.5, fontweight="normal", pad=6, color="black", loc="left")
    # margins in inches (0.52 in left, 0.36 in top, 0.95 in bottom: the x label sits above the two-row legend)
    fig.subplots_adjust(left=0.52 / width, right=0.98, top=1 - 0.36 / H, bottom=0.95 / H, hspace=0.42, wspace=0.42)
    y_center = (axes[0, 1].get_position().y1 + axes[-1, 1].get_position().y0) / 2
    fig.supxlabel(r"Normalized arrival rate", y=0.47 / H, fontsize=10, color="black")
    fig.text(0.06 / width, y_center, r"Fraction of queries to small pool", rotation=90, va="center", ha="center",
             fontsize=10, color="black")
    fig.text(axes[0, 1].get_position().x0 - 0.33 / width, y_center, "Total power (kW)", rotation=90,
             va="center", ha="center", fontsize=9, color="black")
    fig.legend(handles=[
        Patch(facecolor="gray", alpha=0.25, edgecolor="none", label="Predicted feasible range"),
        Line2D([], [], color="black", lw=1, marker="|", markersize=7, markeredgewidth=1, label="Measured feasible range"),
    ], handler_map={Line2D: HandlerLine2D(numpoints=2)}, loc="lower center", bbox_to_anchor=(0.5, 0.02 / H),
        ncol=1, frameon=False, fontsize=8, labelcolor="black")
    savefig(fig, name, "appendix")


def plant_grid(F, keys=FIVE, name="fleet_power_vs_routing_fraction_grid"):
    """The main text's second fleet view for the five fleets on one tall figure: one row per
    fleet, total power against f at every load (left) and the power range against load
    (right); colour = load, one scale for every row."""
    probes, curves, grid, bandm = F.probes, F.curves, F.grid, F.bandm
    cmap = mpl.colormaps[STYLE["load_cmap"]]
    norm = Normalize(vmin=0.05, vmax=1.05)
    n = len(keys)
    H = 1.75 * n + 1.15
    fig, axes = plt.subplots(n, 2, figsize=(5.0, H), gridspec_kw={"width_ratios": [1.35, 1.0]}, squeeze=False)
    for j, key in enumerate(keys):
        ax, axb = axes[j]
        marker = fleet_marker(F, key)
        pr_all = probes[probes.fleet == key]
        us = _loads_of(pr_all)
        for u in sorted(us):
            col = cmap(norm(u))
            pr = pr_all[(pr_all.u == u) & pr_all.feasible].sort_values("f")
            if len(pr):
                ax.plot(pr.f, pr.meas_w / 1e3, color=col, lw=1.3, marker=marker, markersize=3.2, zorder=3)
            cv = curves[(curves.fleet == key) & (curves.u == u) & curves.in_window & curves.pred_w.notna()].sort_values("f")
            if len(cv):
                ax.plot(cv.f, cv.pred_w / 1e3, color=col, lw=1.3, linestyle="--", zorder=2)
        ax.set_xlim(-0.02, 1.02)
        # right: the model's span over its window (dashed) and the measured holdable band, whiskers +-1 standard deviation
        # (from its two end probes' 1 Hz power)
        g = grid[grid.fleet == key].sort_values("u")
        ok = g.p_lo.notna() & g.p_hi.notna()
        axb.plot(g.u[ok], (g.p_hi - g.p_lo)[ok] / 1e3, color="black", lw=1.1, linestyle="--", zorder=2)
        b = bandm[bandm.fleet == key].sort_values("u")
        for _, r in b.iterrows():
            sd = r.sd if np.isfinite(r.sd) else 0
            axb.errorbar([r.u], [r.band / 1e3], yerr=[[min(sd, r.band) / 1e3], [sd / 1e3]], fmt=marker, color=cmap(norm(r.u)),
                         markersize=4, capsize=2, lw=1.4, zorder=3)
        if len(b):
            axb.plot(b.u, b.band / 1e3, color="black", lw=1.3, zorder=2)
        axb.set_xlim(0, 1.05)
        axb.set_ylim(bottom=0)
        last = j == n - 1
        P.finish(ax, xlabel="Fraction of queries to small pool" if last else None)
        P.finish(axb, xlabel="Normalized arrival rate" if last else None)
        ax.set_title(fleet_label(F, key), loc="left", fontsize=9, color="black")
        for a in (ax, axb):
            black(a, label=9, tick=7.5)
    fig.subplots_adjust(left=0.09, right=0.86, top=1 - 0.3 / H, bottom=0.95 / H, hspace=0.45, wspace=0.36)
    # one label per column (every row plots the same two quantities)
    yc = (axes[0, 0].get_position().y1 + axes[-1, 0].get_position().y0) / 2
    fig.text(0.1 / fig.get_figwidth(), yc, "Total power (kW)", rotation=90, va="center", ha="center", fontsize=10, color="black")
    fig.text(axes[0, 1].get_position().x0 - 0.42 / fig.get_figwidth(), yc, "Power range (kW)", rotation=90, va="center", ha="center",
             fontsize=10, color="black")
    cax = fig.add_axes([0.885, 0.3, 0.018, 0.4])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[0.1, 0.3, 0.5, 0.7, 0.9])
    cb.set_label("Normalized arrival rate", fontsize=9, color="black")
    cb.ax.tick_params(labelsize=7.5, colors="black")
    cb.outline.set_visible(False)
    fig.legend(handles=[Line2D([], [], color="black", lw=1.3, marker="o", markersize=3.2,
                               label="Measured Values"),
                        Line2D([], [], color="black", lw=1.3, linestyle="--", label="Predicted Values")],
               loc="lower center", bbox_to_anchor=(0.48, 0.0), ncol=1, frameon=False, fontsize=8, labelcolor="black")
    savefig(fig, name, "appendix")


MAIN = [feasible_window_and_power_band, power_vs_routing_fraction]
APPENDIX = [window_band_grid, plant_grid]
