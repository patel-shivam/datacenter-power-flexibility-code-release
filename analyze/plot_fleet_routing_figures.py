#!/usr/bin/env python
"""Fleet routing figures: the analytical model (exact occupancy solve, built from single-GPU
calibrations only) against the open-loop routing sweeps (fleet_routing_sweep) on every fleet.

  FR1  feasible routing-fraction window vs load (model vs the fractions whose queue held) and
       the power band vs load (model span vs measured held span), two rows, one column per fleet
  FR2  fleet power against the routing fraction f at several loads: measured held probes and
       the model's curve, one colour per load
  FR3  holdable band vs offered load on every fleet, measured (require-held) vs model, with u*
  FR4  where the band peaks: measured peak load against the predicted u* = n_S C_S / C_fleet
  FR5  fleet size: the 4-GPU and 8-GPU llama ShareGPT fleets, measured vs model band
  plus parity statistics over every held probe (power, lit large replicas, in-flight count).

Conventions as in plot_single_gpu_model.py: colour = workload, marker = model pair (▲ llama,
◆ qwen, ■ the 8-GPU fleet); within a panel, loads use the ordinal ramp (dark = heavier).

    python analyze/plot_fleet_routing_figures.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(E))
sys.path.insert(0, str(E / "analyze"))
import plotting as P  # noqa: E402
from analyze import capacity_lookup  # noqa: E402
from analyze.plot_authority_compare import bands as held_bands  # noqa: E402
from analyze.plot_model_vs_load import (CONFIGS, band_at, held_probes, measured, model_for,  # noqa: E402
                                        runs_of, U_GRID)
from plot_single_gpu_model import WL_COLOR as _WL_COLOR, WL_LABEL  # noqa: E402

# the 1:1 ladder rungs that the single-GPU figures only tabulate get the two spare slots here
WL_COLOR = dict(_WL_COLOR)
WL_COLOR.setdefault("fixed_256_256", P.CAT[7])
WL_COLOR.setdefault("fixed_512_512", "#6f6e68")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

H = E / "results"
RX = E / "results/rtx-8000-48gb"

# (key, short label, root, globs) -- the fleet_routing_sweep sweeps of one configuration (plot_model_vs_load's
# CONFIGS, plus the 8-GPU llama fleet of fleet_sweep_8gpu); runs_of keeps one composition per entry
FLEETS = [
    ("llama_sharegpt", "Llama pair · ShareGPT", H, CONFIGS[0][2]),
    ("llama_arxiv", "Llama pair · arXiv", H, CONFIGS[2][2]),
    ("llama_1024", "Llama pair · 1024/1024", H, CONFIGS[4][2]),
    ("qwen_sharegpt", "Qwen pair · ShareGPT", H, CONFIGS[3][2]),
    ("llama_128", "Llama pair · 128/128", H, CONFIGS[5][2]),
    ("llama_256", "Llama pair · 256/256", H, CONFIGS[6][2]),
    ("llama_512", "Llama pair · 512/512", H, CONFIGS[7][2]),
    ("qwen_1024", "Qwen pair · 1024/1024", H, CONFIGS[8][2]),
    ("llama_sharegpt_5L3S", "Llama 5L+3S (8 GPU) · ShareGPT", H,
     ["*llama31-8b-llama32-1b__sharegpt__5L3S__*wide_u*"]),
    # the other two eight-GPU compositions (fleet_sweep_8gpu): skipped when their runs are absent
    ("llama_sharegpt_4L4S", "Llama 4L+4S (8 GPU) · ShareGPT", H,
     ["*llama31-8b-llama32-1b__sharegpt__4L4S__*wide_u*"]),
    ("llama_sharegpt_6L2S", "Llama 6L+2S (8 GPU) · ShareGPT", H,
     ["*llama31-8b-llama32-1b__sharegpt__6L2S__*wide_u*"]),
]
CORE = ("llama_sharegpt", "llama_arxiv", "llama_1024", "qwen_sharegpt")
LADDER = ("llama_128", "llama_256", "llama_512", "qwen_1024")
EIGHT = ("llama_sharegpt_5L3S", "llama_sharegpt_4L4S", "llama_sharegpt_6L2S")
PAIR_MARK = {"llama31-8b": "^", "qwen3-8b": "D"}
PAIR_LABEL = {"llama31-8b": "Llama-3.1-8B ×3 + Llama-3.2-1B", "qwen3-8b": "Qwen3-8B ×3 + Qwen3-4B"}
# the eight-GPU fleets by composition (n_large, n_small)
EIGHT_MARK = {(5, 3): "s", (4, 4): "p", (6, 2): "h"}
EIGHT_LABEL = {(5, 3): "8-GPU fleet, 5 L + 3 S", (4, 4): "8-GPU fleet, 4 L + 4 S", (6, 2): "8-GPU fleet, 6 L + 2 S"}
F_GRID = np.linspace(0.0, 1.0, 51)


def _f(x, default=float("nan")):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ====================================================================== data ===========
class Fleet:
    def __init__(self, key, label, root, globs):
        self.key, self.label, self.root = key, label, root
        self.runs = runs_of(root, globs)
        self.ok = bool(self.runs)
        if not self.ok:
            return
        self.args = json.loads((self.runs[0] / "summary.json").read_text())["args"]
        self.cap = capacity_lookup.fleet_capacity(self.runs[0])
        self.model = model_for(root, self.args)
        self.ok = bool(self.cap and self.model)
        if not self.ok:
            return
        self.model.solver = "exact"
        self.policy = self.args.get("policy", "pack")
        self.C = float(self.cap["fleet_reqs_s"])
        self.n_large, self.n_small = int(self.cap["n_large"]), int(self.cap.get("n_small") or 0)
        self.u_star = self.n_small * float(self.cap["c_small"]) / self.C if self.n_small else float("nan")
        self.wl = self.args["workload"]
        self.pair = self.args["large"]
        self.marker = (EIGHT_MARK.get((self.n_large, self.n_small), "s") if self.n_large + self.n_small > 4
                       else PAIR_MARK.get(self.pair, "o"))
        # measured: held probes (lam, f, P) and per-load ranges, merged over the runs
        self.probes = [p for r in self.runs for p in held_probes(r)]
        by: dict[float, list] = {}
        lam_runs: dict[float, set] = {}
        for r in self.runs:
            for lam, f, p in held_probes(r):
                by.setdefault(round(lam, 4), []).append((f, p))
                lam_runs.setdefault(round(lam, 4), set()).add(r.name)
        self.meas = [{"lam": lam, "u": lam / self.C, "f_lo": min(f for f, _ in g), "f_hi": max(f for f, _ in g),
                      "p_lo": min(p for _, p in g), "p_hi": max(p for _, p in g), "n": len(g),
                      "probes": sorted(g),
                      "wide": any(("wide_u" in n) or ("__u10-80" in n) or ("__u90" in n) for n in lam_runs[lam])}
                     for lam, g in sorted(by.items()) if len(g) >= 2]
        # per-load in-flight / lit counts for the parity statistics (and, for the exported tables,
        # the per-pool power, the realised rate and whether the fraction is within capacity)
        from analyze.plot_fleet import _capacity_feasible
        self.points = []
        for r in self.runs:
            held = {(lam, f) for lam, f, _ in held_probes(r)}
            wide = ("wide_u" in r.name) or ("__u10-80" in r.name) or ("__u90" in r.name)
            for row in csv.DictReader((r / "points.csv").open()):
                lam, f = _f(row["offered_reqs_s"]), _f(row["f"])
                if (lam, f) not in held:
                    continue
                self.points.append({"lam": lam, "f": f, "P": _f(row["power_w"]), "n": _f(row.get("running")),
                                    "k_large": _f(row.get("active_large")), "k_small": _f(row.get("active_small")),
                                    "run": r.name, "wide": wide,
                                    "P_large": _f(row.get("power_large_w")), "P_small": _f(row.get("power_small_w")),
                                    "rate": _f(row.get("realized_reqs_s")),
                                    "feasible": _capacity_feasible(self.cap, lam, f)})
        # holdable band per load (paper's definition: capacity-feasible AND queue held)
        self.band = []
        for r in self.runs:
            for u, b, lo, hi, sd in held_bands(r, self.cap, require_held=True, with_sd=True)["hold"]:
                self.band.append({"u": u, "band": b, "lo": lo, "hi": hi, "sd": sd})
        self.band.sort(key=lambda d: d["u"])
        # model over the load grid
        nan4 = (float("nan"),) * 4
        self.pred = [(u, *(band_at(self.model, u * self.C, self.policy, "exact") or nan4)) for u in U_GRID]

    def model_curve(self, lam: float):
        """(f, P, k_large) over F_GRID at one load; NaN where the model has no steady state."""
        out = []
        for f in F_GRID:
            q = self.model.power(lam, f, self.policy)
            p = q.get("power_total_w", float("nan"))
            out.append((f, p if (p is not None and math.isfinite(p)) else float("nan"), q.get("k_large", float("nan"))))
        return out

    def model_band_at(self, u: float):
        b = band_at(self.model, u * self.C, self.policy, "exact")
        return (b[3] - b[2]) if b else float("nan")


# ====================================================================== drawing ========
def legend(fig, handles, title, rows_hint=None):
    ncol = min(6, max(1, math.ceil(len(handles) / 2)))
    rows = rows_hint or math.ceil(len(handles) / ncol)
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.06 + 0.05 * rows, 1, 0.95))


def fig_window_band(fleets: list[Fleet], path: Path, title: str):
    fleets = [f for f in fleets if f.ok]
    if not fleets:
        return
    n = len(fleets)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n + 0.6, 6.4), squeeze=False)
    for j, fl in enumerate(fleets):
        c = WL_COLOR.get(fl.wl, P.MUTED)
        u, flo, fhi, plo, phi = (np.array(x) for x in zip(*fl.pred))
        ok = np.isfinite(flo)
        ax = axes[0][j]
        ax.fill_between(u[ok], flo[ok], fhi[ok], color=c, alpha=0.25, linewidth=0, label="model window")
        ax.plot(u[ok], flo[ok], color=c, linewidth=1.0)
        ax.plot(u[ok], fhi[ok], color=c, linewidth=1.0)
        for m in fl.meas:
            ax.plot([m["u"], m["u"]], [m["f_lo"], m["f_hi"]], color=P.INK, linewidth=1.6, solid_capstyle="butt", zorder=3)
            ax.scatter([m["u"], m["u"]], [m["f_lo"], m["f_hi"]], s=14, marker="_", color=P.INK, zorder=4)
        if math.isfinite(fl.u_star):
            ax.axvline(fl.u_star, color=P.MUTED, linestyle=":", linewidth=1.0)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(-0.02, 1.02)
        P.finish(ax, ylabel="fraction to the small pool, f" if j == 0 else None, title=fl.label)
        ax = axes[1][j]
        ax.fill_between(u[ok], plo[ok], phi[ok], color=c, alpha=0.25, linewidth=0)
        ax.plot(u[ok], plo[ok], color=c, linewidth=1.0)
        ax.plot(u[ok], phi[ok], color=c, linewidth=1.0)
        for m in fl.meas:
            ax.plot([m["u"], m["u"]], [m["p_lo"], m["p_hi"]], color=P.INK, linewidth=1.6, solid_capstyle="butt", zorder=3)
            ax.scatter([m["u"], m["u"]], [m["p_lo"], m["p_hi"]], s=14, marker="_", color=P.INK, zorder=4)
        if math.isfinite(fl.u_star):
            ax.axvline(fl.u_star, color=P.MUTED, linestyle=":", linewidth=1.0)
        ax.set_xlim(0, 1.05)
        P.finish(ax, xlabel="offered load / measured fleet capacity", ylabel="fleet power (W)" if j == 0 else None)
    handles = [Patch(facecolor=P.INK_2, alpha=0.3, edgecolor="none", label="model: feasible window / power span (exact solve)"),
               Line2D([], [], color=P.INK, linewidth=1.6, label="measured: fractions / power of the probes whose queue held"),
               Line2D([], [], color=P.MUTED, linestyle=":", label="u* = n_S C_S / C_fleet")]
    legend(fig, handles, title, rows_hint=1)
    P.save(fig, path, tight=False)


def fig_power_vs_f(fleets: list[Fleet], path: Path, title: str, target_u=(0.1, 0.3, 0.5, 0.7, 0.9)):
    fleets = [f for f in fleets if f.ok]
    if not fleets:
        return
    n = len(fleets)
    fig, axes = plt.subplots(1, n, figsize=(3.7 * n + 0.6, 4.0), squeeze=False)
    stats = []
    for ax, fl in zip(axes[0], fleets):
        loads = sorted({m["u"]: m for m in fl.meas}.items())
        chosen = []
        for tu in target_u:
            best = min(loads, key=lambda kv: abs(kv[0] - tu), default=None)
            if best and abs(best[0] - tu) <= 0.07 and best not in chosen:
                chosen.append(best)
        chosen.sort()
        cols = P.ordinal_colors(max(len(chosen), 1))
        for (u, m), col in zip(chosen, cols):
            fs = [f for f, _ in m["probes"]]
            ps = [p for _, p in m["probes"]]
            ax.plot(fs, ps, color=col, linewidth=1.3, marker=fl.marker, markersize=4.5, zorder=3,
                    label=f"u = {u:.2f} measured")
            curve = fl.model_curve(m["lam"])
            cf = [f for f, p, _ in curve if math.isfinite(p)]
            cp = [p for f, p, _ in curve if math.isfinite(p)]
            ax.plot(cf, cp, color=col, linewidth=1.2, linestyle="--", zorder=2)
            # per-probe parity for the stats
            for f, p in m["probes"]:
                q = fl.model.power(m["lam"], f, fl.policy)["power_total_w"]
                if q is not None and math.isfinite(q):
                    stats.append({"fleet": fl.key, "u": u, "f": f, "meas": p, "pred": q})
        ax.set_xlim(-0.02, 1.02)
        P.finish(ax, xlabel="fraction to the small pool, f", ylabel="fleet power (W)" if ax is axes[0][0] else None,
                 title=fl.label)
        ax.text(0.03, 0.03, "solid: measured (queue held)\ndashed: model", transform=ax.transAxes, fontsize=7.5,
                color=P.INK_2, va="bottom")
        hs = [Line2D([], [], color=col, linewidth=1.5, label=f"u = {u:.2f}") for (u, _), col in zip(chosen, cols)]
        ax.legend(handles=hs, fontsize=7, loc="upper right", frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    P.save(fig, path, tight=False)
    return stats


def fig_band_vs_load(fleets: list[Fleet], path: Path, title: str, ncols: int = 3):
    fleets = [f for f in fleets if f.ok]
    if not fleets:
        return
    nrows = math.ceil(len(fleets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols + 0.6, 3.1 * nrows + 1.0), squeeze=False)
    for ax, fl in zip(axes.flat, fleets):
        c = WL_COLOR.get(fl.wl, P.MUTED)
        u, flo, fhi, plo, phi = (np.array(x) for x in zip(*fl.pred))
        ok = np.isfinite(plo)
        ax.plot(u[ok], (phi - plo)[ok], color=c, linewidth=1.3, linestyle="--", zorder=2)
        if fl.band:
            bu = [b["u"] for b in fl.band]
            bb = [b["band"] for b in fl.band]
            lo = [b["band"] - b["lo"] if math.isfinite(b["lo"]) else 0 for b in fl.band]
            hi = [b["hi"] - b["band"] if math.isfinite(b["hi"]) else 0 for b in fl.band]
            ax.errorbar(bu, bb, yerr=[lo, hi], fmt=fl.marker, color=c, markersize=5, capsize=2, linewidth=1.0, zorder=3)
            ax.plot(bu, bb, color=c, linewidth=1.0, zorder=3)
        if math.isfinite(fl.u_star):
            ax.axvline(fl.u_star, color=P.MUTED, linestyle=":", linewidth=1.0)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(bottom=0)
        P.finish(ax, xlabel="offered load / measured fleet capacity", ylabel="holdable band (W)", title=fl.label)
    for ax in list(axes.flat)[len(fleets):]:
        ax.axis("off")
    handles = [Line2D([], [], color=P.INK_2, marker="o", linewidth=1.0, label="measured holdable band (probes within capacity whose queue held; whiskers p5–p95)"),
               Line2D([], [], color=P.INK_2, linestyle="--", label="model band (exact solve)"),
               Line2D([], [], color=P.MUTED, linestyle=":", label="u* = n_S C_S / C_fleet")]
    legend(fig, handles, title, rows_hint=1)
    P.save(fig, path, tight=False)


def fig_ustar(fleets: list[Fleet], path: Path):
    fleets = [f for f in fleets if f.ok and f.band and math.isfinite(f.u_star)]
    if not fleets:
        return []
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    rows = []
    for fl in fleets:
        # the peak as a PLATEAU: every load whose band is within 10 % of the largest (the
        # 8-GPU fleet is flat over u = 0.3-0.6); the marker is its midpoint, the bar its extent
        # widened by half the load-grid spacing at each end
        bmax = max(b["band"] for b in fl.band)
        plateau = sorted(b["u"] for b in fl.band if b["band"] >= 0.9 * bmax)
        us = sorted({b["u"] for b in fl.band})
        step = min(np.diff(us)) if len(us) > 1 else 0.1
        lo, hi = plateau[0] - step / 2, plateau[-1] + step / 2
        mid = (lo + hi) / 2
        c = WL_COLOR.get(fl.wl, P.MUTED)
        ax.errorbar([fl.u_star], [mid], yerr=[[mid - lo], [hi - mid]], fmt=fl.marker, color=c, markersize=7,
                    capsize=2, linewidth=1.0, zorder=3)
        peak = max(fl.band, key=lambda b: b["band"])
        # model peak over its own grid
        pu = max(((u, phi - plo) for u, _, _, plo, phi in fl.pred if math.isfinite(plo)), key=lambda t: t[1], default=(float("nan"), 0))
        rows.append({"fleet": fl.key, "u_star": fl.u_star, "measured_peak_u": peak["u"], "measured_peak_w": peak["band"],
                     "plateau_lo": plateau[0], "plateau_hi": plateau[-1],
                     "model_peak_u": pu[0], "model_peak_w": pu[1]})
    ax.plot([0, 1], [0, 1], color=P.MUTED, linewidth=1.0, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    P.finish(ax, xlabel="predicted u* = n_S C_S / C_fleet", ylabel="load of the largest measured band",
             title="The band peaks where the small pool saturates")
    hw = [Patch(facecolor=WL_COLOR[w], edgecolor="none", label=WL_LABEL[w]) for w in WL_LABEL if any(f.wl == w for f in fleets)]
    hm = [Line2D([], [], marker=PAIR_MARK[p], color=P.INK_2, linestyle="none", markersize=6, label=PAIR_LABEL[p])
          for p in PAIR_MARK if any(f.pair == p and f.n_large + f.n_small <= 4 for f in fleets)]
    for comp in sorted({(f.n_large, f.n_small) for f in fleets if f.n_large + f.n_small > 4}):
        hm.append(Line2D([], [], marker=EIGHT_MARK.get(comp, "s"), color=P.INK_2, linestyle="none", markersize=6,
                         label=EIGHT_LABEL.get(comp, f"8-GPU fleet, {comp[0]} L + {comp[1]} S")))
    ax.legend(handles=hw + hm, fontsize=7.5, loc="lower right", frameon=False)
    P.save(fig, path)
    return rows


def fig_scaling(fleets: list[Fleet], path: Path):
    sel = [f for f in fleets if f.ok and f.key in ("llama_sharegpt",) + EIGHT]
    if len(sel) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    cols = P.ordinal_colors(len(sel))
    for fl, col in zip(sel, cols):
        u, flo, fhi, plo, phi = (np.array(x) for x in zip(*fl.pred))
        ok = np.isfinite(plo)
        axes[0].plot(u[ok], (phi - plo)[ok], color=col, linestyle="--", linewidth=1.3)
        if fl.band:
            axes[0].plot([b["u"] for b in fl.band], [b["band"] for b in fl.band], color=col, marker=fl.marker,
                         linewidth=1.0, markersize=5, label=f"{fl.n_large} L + {fl.n_small} S ({fl.n_large + fl.n_small} GPUs)")
        axes[1].fill_between(u[ok], plo[ok], phi[ok], color=col, alpha=0.2, linewidth=0)
        for m in fl.meas:
            axes[1].plot([m["u"], m["u"]], [m["p_lo"], m["p_hi"]], color=col, linewidth=1.8)
        if math.isfinite(fl.u_star):
            axes[0].axvline(fl.u_star, color=col, linestyle=":", linewidth=1.0)
    axes[0].set_xlim(0, 1.05); axes[0].set_ylim(bottom=0)
    axes[1].set_xlim(0, 1.05)
    P.finish(axes[0], xlabel="offered load / measured fleet capacity", ylabel="holdable band (W)",
             title="band: measured (markers) and model (dashed), u* dotted", legend=True)
    P.finish(axes[1], xlabel="offered load / measured fleet capacity", ylabel="fleet power (W)",
             title="power span: measured (bars) and model (shaded)")
    fig.suptitle("Fleet size: the Llama pair on ShareGPT at 4 and 8 GPUs, same single-GPU calibration", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    P.save(fig, path, tight=False)


def fig_plant_vs_load(fl: Fleet, path: Path):
    """The paper's 'plant across load' figure for one fleet, feasible probes only.

    Left: fleet power against f with one colour per load (light to dark with load); the
    probes within both pools' measured capacity whose queue held, on a solid line with filled
    markers; the model dashed in the same colour over the window it calls sustainable.
    Right: the holdable band against load, measured (points, whiskers from the end probes'
    5th-95th power percentiles) and predicted (dashed), with u* marked."""
    if not fl.ok or not fl.meas:
        return
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from analyze.plot_fleet import _capacity_feasible
    # one series per load on the 0.1 grid: where a single-load sweep duplicates a load of the
    # u = 0.1-0.9 protocol, the protocol's sweep (150 s probes) is the one drawn
    groups: dict[float, list] = {}
    for m in fl.meas:
        groups.setdefault(round(m["u"], 1), []).append(m)
    loads = sorted((max(g, key=lambda m: (m["wide"], m["n"])) for g in groups.values()), key=lambda m: m["u"])
    norm = Normalize(vmin=min(m["u"] for m in loads) - 0.05, vmax=max(m["u"] for m in loads) + 0.15)
    cmap = cm.get_cmap("viridis_r")
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.35, 1.0]})
    for m in loads:
        col = cmap(norm(m["u"]))
        pts = [(f, p) for f, p in m["probes"] if _capacity_feasible(fl.cap, m["lam"], f)]
        if pts:
            ax.plot([f for f, _ in pts], [p for _, p in pts], color=col, linewidth=1.3, marker=fl.marker,
                    markersize=4.5, zorder=3, label=f"u = {m['u']:.2f}")
        b = band_at(fl.model, m["lam"], fl.policy, "exact")
        if b:
            curve = [(f, p) for f, p, _ in fl.model_curve(m["lam"]) if math.isfinite(p) and b[0] - 1e-9 <= f <= b[1] + 1e-9]
            ax.plot([f for f, _ in curve], [p for _, p in curve], color=col, linewidth=1.2, linestyle="--", zorder=2)
    ax.set_xlim(-0.02, 1.02)
    P.finish(ax, xlabel="fraction of flexible traffic to the small pool, f", ylabel="fleet power (W)",
             title="fleet power against f: measured (solid, feasible probes) and model (dashed)")
    ax.legend(fontsize=7, loc="upper right", frameon=False, ncol=2, title="offered load / capacity", title_fontsize=7)
    # right: band vs load
    u, flo, fhi, plo, phi = (np.array(x) for x in zip(*fl.pred))
    ok = np.isfinite(plo)
    axb.plot(u[ok], (phi - plo)[ok], color=P.INK_2, linewidth=1.3, linestyle="--", zorder=2, label="model")
    if fl.band:
        bu = [b["u"] for b in fl.band]
        bb = [b["band"] for b in fl.band]
        lo = [b["band"] - b["lo"] if math.isfinite(b["lo"]) else 0 for b in fl.band]
        hi = [b["hi"] - b["band"] if math.isfinite(b["hi"]) else 0 for b in fl.band]
        for bu_i, bb_i, lo_i, hi_i in zip(bu, bb, lo, hi):
            axb.errorbar([bu_i], [bb_i], yerr=[[lo_i], [hi_i]], fmt=fl.marker, color=cmap(norm(bu_i)), markersize=6,
                         capsize=2, linewidth=1.0, zorder=3)
        axb.plot(bu, bb, color=P.INK, linewidth=1.0, zorder=2, label="measured (holdable)")
    if math.isfinite(fl.u_star):
        axb.axvline(fl.u_star, color=P.MUTED, linestyle=":", linewidth=1.0)
        axb.text(fl.u_star + 0.01, axb.get_ylim()[1] * 0.97, f"u* = {fl.u_star:.2f}", fontsize=8, color=P.INK_2, va="top")
    axb.set_xlim(0, 1.05)
    axb.set_ylim(bottom=0)
    P.finish(axb, xlabel="offered load / measured fleet capacity", ylabel="holdable band (W)",
             title="holdable band against load", legend=True)
    fig.suptitle(f"The plant across load: {fl.label}, all traffic flexible, u = {loads[0]['u']:.1f}–{loads[-1]['u']:.1f}",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    P.save(fig, path, tight=False)


# ====================================================================== stats ==========
def parity_stats(fleets: list[Fleet]) -> tuple[list[dict], list[dict]]:
    per_probe, per_fleet = [], []
    for fl in fleets:
        if not fl.ok:
            continue
        rows = []
        for pt in fl.points:
            q = fl.model.power(pt["lam"], pt["f"], fl.policy)
            pred = q.get("power_total_w", float("nan"))
            if pred is None or not math.isfinite(pred) or not math.isfinite(pt["P"]):
                continue
            n_pred = (q.get("n_large") or 0) + (q.get("n_small") or 0)
            rows.append({"fleet": fl.key, "lam": pt["lam"], "u": pt["lam"] / fl.C, "f": pt["f"], "meas_w": pt["P"],
                         "pred_w": pred, "k_large_meas": pt["k_large"], "k_large_pred": q.get("k_large"),
                         "n_meas": pt["n"], "n_pred": n_pred})
        per_probe += rows
        if rows:
            ape = np.array([abs(r["pred_w"] - r["meas_w"]) / r["meas_w"] for r in rows])
            kr = [r for r in rows if math.isfinite(r["k_large_meas"]) and r["k_large_pred"] is not None]
            k_ok = sum(1 for r in kr if round(r["k_large_meas"]) == int(r["k_large_pred"]))
            nr = [r["n_meas"] / r["n_pred"] for r in rows if r["n_pred"] and math.isfinite(r["n_meas"]) and r["n_meas"] > 0]
            per_fleet.append({"fleet": fl.key, "label": fl.label, "n_probes": len(rows), "power_mape_pct": 100 * ape.mean(),
                              "power_median_ratio": float(np.median([r["meas_w"] / r["pred_w"] for r in rows])),
                              "power_within_5pct": float(np.mean(ape <= 0.05)), "power_within_10pct": float(np.mean(ape <= 0.10)),
                              "lit_large_agree": k_ok / len(kr) if kr else float("nan"), "lit_n": len(kr),
                              "inflight_median_ratio": float(np.median(nr)) if nr else float("nan"),
                              "C_fleet": fl.C, "u_star": fl.u_star})
    return per_probe, per_fleet


def export_tables(fleets: list[Fleet], out: Path) -> None:
    """Everything the paper figures (paper_figures/) draw from, as flat tables (no model solve needed later):
    fleet_meta.csv, fleet_probes.csv (every held probe with its per-pool power, realised rate,
    capacity feasibility and the model's prediction), fleet_model_curves.csv (the model's P(f)
    at every measured load), fleet_model_band_grid.csv (window and span over the load grid)
    and fleet_band_measured.csv (the holdable band per load with its p5 / p95 whiskers, lo / hi, and its
    standard deviation, sd)."""
    meta, probes, curves, grid, bands = [], [], [], [], []
    for fl in fleets:
        if not fl.ok:
            continue
        meta.append({"fleet": fl.key, "label": fl.label, "workload": fl.wl, "pair": fl.pair,
                     "n_large": fl.n_large, "n_small": fl.n_small, "C_fleet": fl.C, "u_star": fl.u_star,
                     "c_large": fl.cap["c_large"], "c_small": fl.cap.get("c_small"), "marker": fl.marker,
                     "policy": fl.policy, "n_runs": len(fl.runs), "n_held_probes": len(fl.probes)})
        for pt in fl.points:
            q = fl.model.power(pt["lam"], pt["f"], fl.policy)
            pred = q.get("power_total_w", float("nan"))
            probes.append({"fleet": fl.key, "run": pt["run"], "lam": pt["lam"], "u": pt["lam"] / fl.C, "f": pt["f"],
                           "meas_w": pt["P"], "meas_large_w": pt["P_large"], "meas_small_w": pt["P_small"],
                           "realized_reqs_s": pt["rate"], "n_meas": pt["n"], "k_large_meas": pt["k_large"],
                           "k_small_meas": pt["k_small"], "feasible": pt["feasible"], "wide": pt["wide"],
                           "pred_w": pred if (pred is not None and math.isfinite(pred)) else float("nan"),
                           "pred_large_w": q.get("power_large_w"), "pred_small_w": q.get("power_small_w"),
                           "k_large_pred": q.get("k_large"), "k_small_pred": q.get("k_small"),
                           "n_pred": (q.get("n_large") or 0) + (q.get("n_small") or 0)})
        for m in fl.meas:
            b = band_at(fl.model, m["lam"], fl.policy, "exact")
            for f, p, k in fl.model_curve(m["lam"]):
                curves.append({"fleet": fl.key, "u": m["u"], "lam": m["lam"], "f": f, "pred_w": p, "k_large": k,
                               "in_window": bool(b and b[0] - 1e-9 <= f <= b[1] + 1e-9), "wide": m["wide"],
                               "n_held": m["n"]})
        for u, flo, fhi, plo, phi in fl.pred:
            grid.append({"fleet": fl.key, "u": u, "f_lo": flo, "f_hi": fhi, "p_lo": plo, "p_hi": phi})
        for bd in fl.band:
            bands.append({"fleet": fl.key, **bd})
    for name, rows in (("fleet_meta.csv", meta), ("fleet_probes.csv", probes), ("fleet_model_curves.csv", curves),
                       ("fleet_model_band_grid.csv", grid), ("fleet_band_measured.csv", bands)):
        with open(out / name, "w", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  tables: {len(meta)} fleets, {len(probes)} probes, {len(curves)} model-curve points, "
          f"{len(grid)} grid rows, {len(bands)} band rows")


def window_stats(fleets: list[Fleet]) -> list[dict]:
    out = []
    for fl in fleets:
        if not fl.ok:
            continue
        for m in fl.meas:
            b = band_at(fl.model, m["lam"], fl.policy, "exact")
            if not b:
                continue
            out.append({"fleet": fl.key, "u": m["u"], "n_held": m["n"], "f_lo_meas": m["f_lo"], "f_lo_model": b[0],
                        "f_hi_meas": m["f_hi"], "f_hi_model": b[1], "p_lo_meas": m["p_lo"], "p_lo_model": b[2],
                        "p_hi_meas": m["p_hi"], "p_hi_model": b[3], "band_meas": m["p_hi"] - m["p_lo"],
                        "band_model": b[3] - b[2]})
    return out


# ====================================================================== main ===========
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=E / "results/tables/fleet_routing")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    P.apply_style()

    fleets = []
    for key, label, root, globs in FLEETS:
        fl = Fleet(key, label, root, globs)
        if fl.ok:
            print(f"  {key:22s} runs {len(fl.runs)}  C {fl.C:7.2f} req/s  u* {fl.u_star:.3f}  held probes {len(fl.probes):4d}  "
                  f"loads {len(fl.meas)}  bands {len(fl.band)}")
        else:
            print(f"  {key:22s} no usable runs")
        fleets.append(fl)
    by = {f.key: f for f in fleets}

    fig_window_band([by[k] for k in CORE], out / "fr1_window_band__core.png",
                    "Feasible routing window and power band against load: model (exact solve) vs the open-loop sweeps")
    fig_window_band([by[k] for k in LADDER], out / "fr1b_window_band__ladder.png",
                    "Feasible routing window and power band against load: the 1:1 ladder and the Qwen pair at 1024/1024")
    s1 = fig_power_vs_f([by[k] for k in CORE], out / "fr2_power_vs_f__core.png",
                        "Fleet power against the routing fraction at several loads: measured probes (solid) and the model (dashed)")
    s2 = fig_power_vs_f([by[k] for k in LADDER], out / "fr2b_power_vs_f__ladder.png",
                        "Fleet power against the routing fraction: the 1:1 ladder and the Qwen pair at 1024/1024")
    fig_band_vs_load([by[k] for k in CORE + LADDER], out / "fr3_band_vs_load.png",
                     "Holdable band against offered load on every fleet: measured vs model, u* marked")
    ustar_rows = fig_ustar(fleets, out / "fr4_band_peak_vs_ustar.png")
    fig_scaling(fleets, out / "fr5_fleet_size.png")
    for fl in fleets:
        fig_plant_vs_load(fl, out / f"fr8_plant__{fl.key}.png")

    per_probe, per_fleet = parity_stats(fleets)
    win = window_stats(fleets)
    export_tables(fleets, out)
    with open(out / "fleet_probe_parity.csv", "w", newline="") as f:
        if per_probe:
            w = csv.DictWriter(f, fieldnames=list(per_probe[0].keys())); w.writeheader(); w.writerows(per_probe)
    with open(out / "fleet_parity_summary.csv", "w", newline="") as f:
        if per_fleet:
            w = csv.DictWriter(f, fieldnames=list(per_fleet[0].keys())); w.writeheader(); w.writerows(per_fleet)
    with open(out / "fleet_window_band.csv", "w", newline="") as f:
        if win:
            w = csv.DictWriter(f, fieldnames=list(win[0].keys())); w.writeheader(); w.writerows(win)
    with open(out / "fleet_band_peak.csv", "w", newline="") as f:
        if ustar_rows:
            w = csv.DictWriter(f, fieldnames=list(ustar_rows[0].keys())); w.writeheader(); w.writerows(ustar_rows)

    print("\n== fleet power parity over held probes (model from single-GPU calibration, exact solve)")
    for r in per_fleet:
        print(f"  {r['fleet']:22s} n {r['n_probes']:4d}  MAPE {r['power_mape_pct']:5.1f} %  median meas/pred {r['power_median_ratio']:.3f}  "
              f"within 5/10 % {100*r['power_within_5pct']:3.0f}/{100*r['power_within_10pct']:3.0f}  lit-large agree {100*r['lit_large_agree']:3.0f} % ({r['lit_n']})  "
              f"in-flight meas/pred {r['inflight_median_ratio']:.2f}")
    allp = per_probe
    if allp:
        ape = np.array([abs(r["pred_w"] - r["meas_w"]) / r["meas_w"] for r in allp])
        print(f"  ALL                    n {len(allp):4d}  MAPE {100*ape.mean():5.1f} %  within 5/10 % {100*np.mean(ape<=0.05):3.0f}/{100*np.mean(ape<=0.10):3.0f}")
    print("\n== window ends and band, model minus measured, per fleet (over the measured loads)")
    for fl in fleets:
        rows = [r for r in win if r["fleet"] == fl.key]
        if not rows:
            continue
        dlo = np.array([r["f_lo_model"] - r["f_lo_meas"] for r in rows]); dhi = np.array([r["f_hi_model"] - r["f_hi_meas"] for r in rows])
        db = np.array([r["band_model"] - r["band_meas"] for r in rows])
        print(f"  {fl.key:22s} loads {len(rows):2d}  f_lo {dlo.mean():+.3f} (|max| {np.abs(dlo).max():.3f})  f_hi {dhi.mean():+.3f} (|max| {np.abs(dhi).max():.3f})  "
              f"band {db.mean():+6.0f} W (|max| {np.abs(db).max():4.0f} W)  measured band {min(r['band_meas'] for r in rows):4.0f}-{max(r['band_meas'] for r in rows):4.0f} W")
    print("\n== band peak vs u*")
    for r in ustar_rows:
        print(f"  {r['fleet']:22s} u* {r['u_star']:.2f}  measured peak at u {r['measured_peak_u']:.2f} ({r['measured_peak_w']:.0f} W)  model peak at u {r['model_peak_u']:.2f} ({r['model_peak_w']:.0f} W)")
    print(f"\nwrote figures and tables under {out}")


if __name__ == "__main__":
    main()
