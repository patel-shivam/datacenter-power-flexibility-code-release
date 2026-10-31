"""Single-GPU model figures: the model's prediction against measurement on one replica.

Main text (H100, the four core models, six tasks; one 2 x 2 panel per figure):
  power_vs_resident_requests          P(b): mean power against the resident requests b
  step_time_vs_resident_requests      tau(b) = t0 + t1 b, and t1 by task
  generation_time_vs_resident_requests  T(b) = s tau(b) against the measured mean residence
  resident_requests_vs_arrival_rate   b(lambda) = lambda s t0 / (1 - lambda s t1) against Poisson phases
  completion_rate_vs_resident_requests  mu(b) = b / T(b) against the measured request throughput
Appendix:
  parity_all_gpus                     five predicted quantities, measured against predicted, on four GPUs

Data: results/tables/single_gpu/*.csv (analyze/plot_single_gpu_model.py).
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .common import (HWL, ML, MM, MODELS, P, SCI6, STYLE, TASKS, WLC, WLL, black, savefig, stats)

HW_MAIN = "h100-80gb"


def _black_panel(ax):
    """The main figures' per-panel finish (title 9 pt, labels 9 pt, ticks 8 pt, all black)."""
    ax.title.set(fontsize=9, color="black")
    ax.xaxis.label.set(fontsize=9, color="black")
    ax.yaxis.label.set_color("black")
    ax.tick_params(axis="both", which="both", labelsize=8, colors="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
    for axis in (ax.xaxis, ax.yaxis):
        axis.get_offset_text().set_color("black")


def power_vs_resident_requests(T, hw: str = HW_MAIN):
    """P(b): the measured mean power of every closed-loop level against its achieved resident
    count, with the fitted power profile through it (one panel per core model)."""
    lv, pbc = T.lv, T.pbc
    models = [m for m in MODELS if ((lv.hw == hw) & (lv.model == m)).any()]

    fig, axes = plt.subplots(2, 2, figsize=(5, 5), sharey=True)
    axes = axes.ravel()

    for i, (ax, m) in enumerate(zip(axes, models)):
        for wl in TASKS:
            d = lv[(lv.hw == hw) & (lv.model == m) & (lv.workload == wl)].sort_values("b")
            c = pbc[(pbc.hw == hw) & (pbc.model == m) & (pbc.workload == wl)].sort_values("b")
            if d.empty:
                continue
            c = c[c.b >= 1]

            if not c.empty:
                ax.plot(c.b, c.P_fit_w, color=WLC[wl], lw=1.2)
            ax.scatter(d.b, d.P_w, s=24, marker=MM[m], color=WLC[wl], lw=0.8)

        ax.set_ylim(250, 720)
        ax.set_yticks(np.arange(300, 701, 100))
        P.finish(ax, xlabel=r"Request Concurrency $b$" if i >= 2 else None,
                 ylabel=None, title=ML[m])
        _black_panel(ax)

    for ax in axes[len(models):]:
        ax.set_visible(False)

    wls = [w for w in TASKS if ((lv.hw == hw) & lv.model.isin(models) & (lv.workload == w)).any()]
    handles = [Patch(facecolor=WLC[w], edgecolor="none", label=WLL[w]) for w in wls]
    handles += [
        Line2D([], [], color="black", lw=1.2, label="Predicted Power"),
        Line2D([], [], marker="o", color="black", ls="none", ms=5, label="Measured Power"),
    ]

    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.13),
               ncol=3, frameon=False, fontsize=8, labelcolor="black")
    fig.subplots_adjust(left=0.12, bottom=0.23, hspace=0.48, wspace=0.18)
    fig.supylabel("Mean power (W)", x=0.02, fontsize=10, color="black")

    savefig(fig, f"single_gpu_power_vs_resident_requests_{hw}", "main")


def step_time_vs_resident_requests(T, hw: str = HW_MAIN):
    """tau(b) = t0 + t1 b per task against the measured mean time between tokens of the
    steady-state levels (one panel per core model), and t1 by task (right)."""
    lv, cal = T.lv, T.cal
    cells_hw = cal[cal.hw == hw]
    models = [m for m in MODELS if (cells_hw.model == m).any()]
    wls = [w for w in TASKS if ((cells_hw.workload == w) & cells_hw.model.isin(models)).any()]

    n = len(models)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.0 * n + 5.0, 3), squeeze=False,
                             gridspec_kw={"width_ratios": [1] * n + [1.5]})
    axes = axes[0]
    axb = axes[-1]

    for ax, m in zip(axes[:-1], models):
        for wl in wls:
            d = lv[(lv.hw == hw) & (lv.model == m) & (lv.workload == wl)
                   & (lv.b > 0) & lv.valid & lv.tbt_ms.notna()]
            c = cells_hw[(cells_hw.model == m) & (cells_hw.workload == wl)]
            if d.empty or c.empty:
                continue
            t0, t1 = float(c.t0_ms.iloc[0]), float(c.t1_ms.iloc[0])
            bb = np.linspace(0, d.b.max() * 1.04, 60)
            ax.plot(bb, t0 + t1 * bb, color=WLC[wl], linewidth=1.2, zorder=2)
            ax.scatter(d.b, d.tbt_ms, s=24, marker=MM[m], facecolor=WLC[wl],
                       edgecolor=WLC[wl], linewidth=0.9, zorder=3)

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        P.finish(ax, xlabel=r"Request Concurrency $b$",
                 ylabel="mean TBT (ms)" if ax is axes[0] else None, title=ML[m])

    width = 0.8 / len(models)
    for i, wl in enumerate(wls):
        for j, m in enumerate(models):
            c = cells_hw[(cells_hw.model == m) & (cells_hw.workload == wl)]
            if c.empty:
                continue
            x = i + (j - (len(models) - 1) / 2) * width
            y = float(c.t1_ms.iloc[0])
            axb.bar(x, y, width=width * 0.92, color=WLC[wl], linewidth=0, zorder=2)
            axb.scatter(x, y, s=26, marker=MM[m], facecolor=P.SURFACE,
                        edgecolor=P.INK, linewidth=0.8, zorder=4)

    axb.set_xticks(range(len(wls)))
    axb.set_xticklabels([WLL[w] for w in wls], rotation=30, ha="right", fontsize=8)
    axb.set_ylim(bottom=0)
    P.finish(axb, ylabel=r"$t_1$ (ms per request)", title=r"$t_1$ by task")

    for ax in axes:
        ax.title.set_color("black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.tick_params(axis="both", which="both", colors="black")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
        for axis in (ax.xaxis, ax.yaxis):
            axis.get_offset_text().set_color("black")

    handles = [Patch(facecolor=WLC[w], edgecolor="none", label=WLL[w]) for w in wls]
    handles += [
        Line2D([], [], marker=MM[m], color="black", linestyle="none",
               markersize=7, label=ML[m])
        for m in models
    ]

    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.89, 0.5),
               frameon=False, fontsize=9, labelcolor="black")
    fig.subplots_adjust(right=0.89, wspace=0.28)

    print(f"  step time across tasks on the {HWL[hw]}: measured TBT points + fitted tau(b) = t0 + t1 b; "
          f"right: fitted t1 by task / model.  models: {', '.join(ML[m] for m in models)}; "
          f"tasks: {', '.join(WLL[w] for w in wls)}")
    savefig(fig, f"single_gpu_step_time_vs_resident_requests_{hw}", "main")


def generation_time_vs_resident_requests(T, hw: str = HW_MAIN):
    """T(b) = s tau(b) per task (line) against the measured mean residence of every
    steady-state closed-loop level (one panel per core model, log axes)."""
    pts, cal = T.pts, T.cal
    cells_hw = cal[cal.hw == hw]
    models = [m for m in MODELS if (cells_hw.model == m).any()]
    lvl = pts[(pts.figure == "residence_level") & (pts.hw == hw)]
    lvl = lvl[np.isfinite(lvl.b) & np.isfinite(lvl.y) & (lvl.b > 0) & (lvl.y > 0)]

    fig, axes = plt.subplots(2, 2, figsize=(5, 5), sharey=True)
    axes, shown = axes.ravel(), set()

    for i, (ax, m) in enumerate(zip(axes, models)):
        for wl in TASKS:
            c = cells_hw[(cells_hw.model == m) & (cells_hw.workload == wl)]
            d = lvl[(lvl.model == m) & (lvl.wl == wl)]
            if c.empty or d.empty:
                continue

            s = float(c.s.iloc[0])
            t0, t1 = float(c.t0_ms.iloc[0]) / 1e3, float(c.t1_ms.iloc[0]) / 1e3
            bb = np.geomspace(d.b.min() / 1.05, d.b.max() * 1.05, 100)
            ax.plot(bb, s * (t0 + t1 * bb), color=WLC[wl], lw=1.2)
            ax.scatter(d.b, d.y, s=24, marker=MM[m], color=WLC[wl], lw=0.8)
            shown.add(wl)

        ax.set_xscale("log")
        ax.set_yscale("log")
        P.finish(ax, xlabel=r"Request Concurrency $b$" if i >= 2 else None,
                 ylabel=None, title=ML[m])
        _black_panel(ax)

    for ax in axes[len(models):]:
        ax.set_visible(False)

    wls = [w for w in TASKS if w in shown]
    handles = [Patch(facecolor=WLC[w], edgecolor="none", label=WLL[w]) for w in wls]
    handles += [
        Line2D([], [], color="black", lw=1.2, label=r"Prediction: $T(b)=s\,\tau(b)$"),
        Line2D([], [], marker="o", color="black", ls="none", ms=5, label="Measured mean"),
    ]

    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, 0.13),
               ncol=3, frameon=False, fontsize=8, labelcolor="black")
    fig.subplots_adjust(left=0.12, bottom=0.23, hspace=0.48, wspace=0.18)
    fig.supylabel("Mean Request Residence Time (s)", x=0.02, fontsize=10, color="black")

    savefig(fig, f"single_gpu_generation_time_vs_resident_requests_{hw}", "main")


def resident_requests_vs_arrival_rate(T, hw: str = HW_MAIN):
    """The occupancy fixed point b(lambda) = lambda s t0 / (1 - lambda s t1) per task against the
    arrival rate, with every stable Poisson probe (capacity bisection) and phase (Poisson
    sweep) at its realised rate; dotted: the measured capacity. Log axes."""
    pts, cal = T.pts, T.cal
    cells_hw = cal[cal.hw == hw]
    models = [m for m in MODELS if (cells_hw.model == m).any()]
    occ = pts[(pts.figure == "occupancy") & (pts.hw == hw)].copy()
    occ = occ[np.isfinite(occ.lam) & np.isfinite(occ.y) & (occ.lam > 0) & (occ.y > 0)]
    for flag in ("hollow", "kv_bound", "clamped"):
        if flag in occ:
            occ = occ[~occ[flag].astype(bool)]
    capd = pts[(pts.figure == "capacity") & (pts.hw == hw)]

    fig, axes = plt.subplots(2, 2, figsize=(5, 5), sharey=True)
    axes = axes.ravel()
    shown = set()

    for i, (ax, m) in enumerate(zip(axes, models)):
        for wl in TASKS:
            c = cells_hw[(cells_hw.model == m) & (cells_hw.workload == wl)]
            d = occ[(occ.model == m) & (occ.wl == wl)]
            if c.empty or d.empty:
                continue

            a = float(c.s.iloc[0]) * float(c.t0_ms.iloc[0]) / 1e3
            beta = float(c.s.iloc[0]) * float(c.t1_ms.iloc[0]) / 1e3
            pole = 1 / beta
            cap = float(d.cap.max()) if d.cap.notna().any() else np.nan
            hi = min(pole * 0.995, cap / (a + cap * beta)) if np.isfinite(cap) else pole * 0.995
            lo = min(max(d.lam.min() * 0.5, pole * 1e-3), hi * 0.5)
            lam = np.geomspace(lo, hi, 200)

            ax.plot(lam, lam * a / (1 - lam * beta), color=WLC[wl],
                    linewidth=1.2, zorder=2)
            ax.scatter(d.lam, d.y, s=24, marker=MM[m], facecolor=WLC[wl],
                       edgecolor=WLC[wl], linewidth=0.8, zorder=3)

            k = capd[(capd.model == m) & (capd.wl == wl)]
            if len(k) and np.isfinite(k.y.iloc[0]) and k.y.iloc[0] > 0:
                ax.axvline(float(k.y.iloc[0]), color=WLC[wl],
                           linewidth=1.1, linestyle=":", zorder=1)
            shown.add(wl)

        ax.set_xscale("log")
        ax.set_yscale("log")
        P.finish(ax, xlabel=r"Arrival Rate $\lambda$ (reqs/s)" if i >= 2 else None,
                 ylabel=None, title=ML[m])

    for ax in axes:
        _black_panel(ax)

    for ax in axes[len(models):]:
        ax.set_visible(False)

    wls = [w for w in TASKS if w in shown]
    handles = [Patch(facecolor=WLC[w], edgecolor="none", label=WLL[w]) for w in wls]
    handles += [
        Line2D([], [], color="black", linewidth=1.2, label=r"Prediction: $b(\lambda)$"),
        Line2D([], [], marker="o", color="black", linestyle="none",
               markersize=5, label="Measured mean"),
        Line2D([], [], color="black", linestyle=":", label="Max throughput"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.13),
               ncol=3, frameon=False, fontsize=8, labelcolor="black")
    fig.subplots_adjust(left=0.12, bottom=0.23, hspace=0.48, wspace=0.18)
    fig.supylabel("Mean Request Concurrency", x=0.02, fontsize=10, color="black")

    print(f"  resident requests against the arrival rate on the {HWL[hw]}: solid = predicted occupancy up to the "
          f"concurrency cap, markers = measured Poisson means, dotted = measured capacity; above-cap / KV-pinned "
          f"points excluded.  models: {', '.join(ML[m] for m in models)}; workloads: {', '.join(WLL[w] for w in wls)}")
    savefig(fig, f"single_gpu_resident_requests_vs_arrival_rate_{hw}", "main")


def completion_rate_vs_resident_requests(T, hw: str = HW_MAIN):
    """mu(b) = b / T(b) per task against the resident count, with the measured closed-loop
    levels (window-edge corrected) on the curves; log axes."""
    pts, cal = T.pts, T.cal
    cells_hw = cal[cal.hw == hw]
    models = [m for m in MODELS if (cells_hw.model == m).any()]
    thr = pts[(pts.figure == "throughput_level") & (pts.hw == hw)]
    thr = thr[np.isfinite(thr.b) & np.isfinite(thr.y) & (thr.b > 0) & (thr.y > 0)]

    fig, axes = plt.subplots(2, 2, figsize=(5, 5), sharey=True)
    axes, shown = axes.ravel(), set()

    for i, (ax, m) in enumerate(zip(axes, models)):
        for wl in TASKS:
            c = cells_hw[(cells_hw.model == m) & (cells_hw.workload == wl)]
            d = thr[(thr.model == m) & (thr.wl == wl)]
            if c.empty or d.empty:
                continue

            s = float(c.s.iloc[0])
            t0, t1 = float(c.t0_ms.iloc[0]) / 1e3, float(c.t1_ms.iloc[0]) / 1e3
            bb = np.geomspace(d.b.min() / 1.05, d.b.max() * 1.05, 100)

            ax.plot(bb, bb / (s * (t0 + t1 * bb)), color=WLC[wl], lw=1.2)
            ax.scatter(d.b, d.y, s=15, marker=MM[m], color=WLC[wl], lw=0.8)
            shown.add(wl)

        ax.set_xscale("log")
        ax.set_yscale("log")
        P.finish(ax, xlabel=r"Request Concurrency $b$" if i >= 2 else None,
                 ylabel=None, title=ML[m])
        _black_panel(ax)

    for ax in axes[len(models):]:
        ax.set_visible(False)

    wls = [w for w in TASKS if w in shown]
    handles = [Patch(facecolor=WLC[w], edgecolor="none", label=WLL[w]) for w in wls]
    handles += [
        Line2D([], [], color="black", lw=1.2, label=r"Prediction: $\mu(b)=b/T(b)$"),
        Line2D([], [], marker="o", color="black", ls="none", ms=5, label="Measured mean"),
    ]

    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.13),
                     ncol=3, frameon=False, fontsize=8, labelcolor="black")

    for t in leg.get_texts():
        if "Prediction:" in t.get_text():
            t.set_fontsize(9)

    fig.subplots_adjust(left=0.12, bottom=0.23, hspace=0.48, wspace=0.18)
    fig.supylabel("Request Throughput (reqs/s)", x=0.02, fontsize=10, color="black")

    savefig(fig, f"single_gpu_completion_rate_vs_resident_requests_{hw}", "main")


# ---- appendix ---------------------------------------------------------------------------------
# One colour and marker per GPU (newest first) for the figure that pools the four cards.
HW_ORDER_APPENDIX = ["h100-80gb", "rtx-5090-32gb", "rtx-8000-48gb", "v100-32gb"]
GPU_SHORT = {"h100-80gb": "H100", "rtx-5090-32gb": "RTX 5090", "rtx-8000-48gb": "RTX 8000", "v100-32gb": "V100"}
GPU_COLOR = dict(zip(HW_ORDER_APPENDIX, SCI6[:4]))
GPU_MARK = dict(zip(HW_ORDER_APPENDIX, ["o", "s", "^", "D"]))

# the points a quantity is scored on: the exclusion the single-GPU data layer applies to it
EXCL = {"power_level": None, "power_rate": "kv_bound", "tbt": None, "residence_level": None,
        "throughput_level": None, "occupancy": "kv_bound", "capacity": None, "drain": "offline", "kv_ceiling": "peak"}


def scored(T, fig: str, hw: str | None = None):
    d = T.pts[T.pts.figure == fig]
    if hw is not None:
        d = d[d.hw == hw]
    return d[~d[EXCL[fig]]] if EXCL[fig] else d


def parity_all_gpus(T):
    """Five predicted quantities, measured against predicted, on all four GPUs (one panel per
    quantity; the MAPE per GPU in each panel). Capacity is scored in the appendix table only."""
    PAR = [("power_level", "Power Consumption (W)"), ("tbt", "TBT (ms)"), ("residence_level", "Mean Request\nResidence Time (s)"),
           ("throughput_level", "Request Throughput (reqs/s)"), ("occupancy", "Request Concurrency")]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.0))
    axes = axes.ravel()
    for i, (ax, (key, label)) in enumerate(zip(axes, PAR)):
        d = scored(T, key)
        d = d[(d.x > 0) & (d.y > 0)]
        lo, hi = min(d.x.min(), d.y.min()) * 0.8, max(d.x.max(), d.y.max()) * 1.25
        ax.plot([lo, hi], [lo, hi], color=P.MUTED, lw=0.9, zorder=1)
        lines = []
        for hw in HW_ORDER_APPENDIX:
            g = d[d.hw == hw]
            ax.scatter(g.x, g.y, s=9, marker=GPU_MARK[hw], facecolor="none", edgecolor=GPU_COLOR[hw], lw=0.6, alpha=0.8, zorder=3)
            lines.append(f"{GPU_SHORT[hw]} {stats(g.x, g.y)['mape_pct']:.1f} %")
        ax.text(0.04, 0.96, "MAPE\n" + "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=6.5, color="black")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
        if key == "power_level":                      # a narrow range (factor 5): name the ticks, no minor labels
            ticks = [200, 300, 500, 700]
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(mpl.ticker.FixedLocator(ticks))
                axis.set_major_formatter(mpl.ticker.FixedFormatter([str(t) for t in ticks]))
                axis.set_minor_formatter(mpl.ticker.NullFormatter())
        # x label on every panel with nothing below it (the top-right one sits over the legend)
        P.finish(ax, xlabel="Predicted" if i >= 2 else None, ylabel="Measured" if i % 3 == 0 else None)
        ax.set_title(label, loc="left", fontsize=9, color="black")
        black(ax, label=9, tick=7.5)
    # the sixth slot carries the legend
    axes[5].axis("off")
    handles = [Line2D([], [], marker=GPU_MARK[h], color=GPU_COLOR[h], markerfacecolor="none", ls="none", ms=6, label=HWL[h])
               for h in HW_ORDER_APPENDIX]
    handles += [Line2D([], [], color=P.MUTED, lw=0.9, label="measured = predicted")]
    axes[5].legend(handles=handles, loc="center", frameon=False, fontsize=8, labelcolor="black")
    # room above the top row for the two-line title (top 0.95 -> 0.92; hspace keeps the panel size and the saved size)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.09, hspace=0.297, wspace=0.32)
    savefig(fig, "single_gpu_parity_all_gpus", "appendix")


MAIN = [power_vs_resident_requests, step_time_vs_resident_requests, generation_time_vs_resident_requests,
        resident_requests_vs_arrival_rate, completion_rate_vs_resident_requests]
APPENDIX = [parity_all_gpus]
