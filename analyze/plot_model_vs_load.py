#!/usr/bin/env python
"""How well does the analytical fleet model predict the routing knob's range and the power
range it spans, as the arrival rate varies? One row per fleet configuration. No GPU.

    python analyze/plot_model_vs_load.py \\
        --out results/figures/model_vs_load.png

For each configuration the model (lib/power_model.FleetModel, built exactly as
score_fleet_model.py builds it: single-GPU P(b) curves and step-time fits, capacity from the
fit) is evaluated on a fine grid of offered load and drawn as a band:

  (a) the routing fractions it calls sustainable (both pools inside its own capacity);
  (b) the fleet power range over those fractions, lowest .. highest (the knob's authority).

The band is the model's occupancy-vector solve (solver="exact"). The library's mean-field
default is not drawn: it fails to converge over most of the RTX window and on arXiv at
f <= 0.6, so it is not a prediction there.

Measured, at every load a configuration was swept at: (a) the lowest and highest f whose
queue HELD (score_fleet_model's rule: the probe started drained and its waiting list grew
< 0.5 req/s over the settled half), and (b) the lowest and highest fleet power over those same
probes. u is against the MEASURED fleet capacity (capacity_bisection) for both, so the model's own
capacity estimate shows up as where its window closes.

RTX 8000 rows use steady-state runs only (probes >= 180 s; 60 s probes are mid-ramp there).
Runs matching a row's globs (e.g. the H100 fleet_sweep_4gpu wide-range sweeps) join it automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
E = Path(__file__).resolve().parent.parent
H = E / "results"
RX = E / "results/rtx-8000-48gb"

# (label, results root, run-name globs) -- every run matched must share the configuration
CONFIGS = [
    ("H100 · llama31-8b ×3 + llama32-1b · ShareGPT · all flexible", H,
     ["*__fine_lam40", "*__fine_lam66.1", "*__fine_lam90", "*__rungs100flex_pack",
      "*llama31-8b-llama32-1b__sharegpt__*__wide_u*"]),
    ("H100 · llama31-8b ×3 + llama32-1b · ShareGPT · 80 % flexible", H,
     ["20260907_122224__3L1S__sharegpt__poisson__pack__c64-128"]),
    # fleet_sweep_4gpu's pack sweep only: the u = 0.5 control (20260916_002812) is `spread` with 60 s
    # probes, a different configuration, and mixing it in would solve the model with `spread`
    # for both.
    ("H100 · llama31-8b ×3 + llama32-1b · arXiv · all flexible", H,
     ["*llama31-8b-llama32-1b__arxiv_summarization__*__wide_u*"]),
    # the qwen pair over the whole load range (fleet_sweep_4gpu's protocol), pack
    ("H100 · qwen3-8b ×3 + qwen3-4b · ShareGPT · all flexible", H,
     ["*qwen3-8b-qwen3-4b__sharegpt__*__wide_u*"]),
    # (the single-load `*__pair_*_fgrid` sweeps are `spread`, which no result uses; they are
    #  left out -- every row here is `pack`)
    ("H100 · llama31-8b ×3 + llama32-1b · 1024 in / 1024 out", H,
     ["*__llama31-8b-llama32-1b__fixed_1024_1024__*__u10-80",
      "*__llama31-8b-llama32-1b__fixed_1024_1024__*__u90__fwin"]),
    # the rest of the 1:1 ladder (fleet_sweep_4gpu with a workload override)
    ("H100 · llama31-8b ×3 + llama32-1b · 128 in / 128 out", H,
     ["*llama31-8b-llama32-1b__fixed_128_128__*__wide_u*"]),
    ("H100 · llama31-8b ×3 + llama32-1b · 256 in / 256 out", H,
     ["*llama31-8b-llama32-1b__fixed_256_256__*__wide_u*"]),
    ("H100 · llama31-8b ×3 + llama32-1b · 512 in / 512 out", H,
     ["*llama31-8b-llama32-1b__fixed_512_512__*__wide_u*"]),
    ("H100 · qwen3-8b ×3 + qwen3-4b · 1024 in / 1024 out", H,
     ["*__qwen3-8b-qwen3-4b__fixed_1024_1024__*__u10-80",
      "*__qwen3-8b-qwen3-4b__fixed_1024_1024__*__u90__fwin"]),
    ("RTX 8000 · qwen3-8b ×5 + qwen3-4b ×3 · ShareGPT", RX, ["*__5L3S__pack__*"]),
    ("RTX 8000 · qwen3-8b ×4 + qwen3-4b ×4 · ShareGPT", RX, ["*__4L4S__pack__*"]),
    ("RTX 8000 · qwen3-8b ×6 + qwen3-4b ×2 · ShareGPT", RX, ["*__6L2S__pack__*"]),
]
STEADY_S = {"rtx-8000-48gb": 180.0}      # shortest probe that is steady state, per card
U_GRID = np.linspace(0.02, 1.05, 70)


def _f(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def runs_of(root: Path, globs: list[str]) -> list[Path]:
    """The config's runs: matched by name, 100 % flexible or as the first run says, steady
    probes only on cards that need them, one pair / workload / composition / caps."""
    from analyze import capacity_lookup
    seen, out = set(), []
    for g in globs:
        for d in sorted((root / "fleet_routing_sweep").glob(g)):
            if d in seen or not (d / "summary.json").exists() or not (d / "points.csv").exists():
                continue
            seen.add(d)
            out.append(d)
    if not out:
        return []
    hw = capacity_lookup.hardware_of(root)
    first = json.loads((out[0] / "summary.json").read_text())["args"]
    # The placement policy is part of the configuration: the model is solved with one policy
    # per row (the first run's), so a row must never mix `pack` and `spread` runs.
    key = lambda a: (a.get("large"), a.get("small"), a.get("workload"), int(a["n_large"]),
                     int(a.get("n_small") or 0), a.get("concurrency_large"),
                     a.get("concurrency_small"), str(a.get("class_split")), a.get("policy"))
    keep = []
    for d in out:
        a = json.loads((d / "summary.json").read_text())["args"]
        if key(a) != key(first):
            continue
        if float(a.get("point_s") or 0) < STEADY_S.get(hw, 0.0):
            continue
        keep.append(d)
    return keep


def model_for(root: Path, a: dict):
    """The fleet model for one configuration, as score_fleet_model.collect builds it."""
    from analyze.score_fleet_model import pool_from
    from lib import power_model as pm
    from lib import workloads
    curves = pm.load_curves(root / "power_curves.json")
    report = json.load(open(root / "power_curves_report.json"))
    res = {k: v["residence"] for k, v in report.items()
           if isinstance(v, dict) and v.get("residence", {}).get("ok")}
    wl_name = a["workload"]
    lk, sk = f"{a['large']}__{wl_name}", f"{a['small']}__{wl_name}"
    if lk not in curves or sk not in curves or lk not in res or sk not in res:
        return None
    wl = workloads.WORKLOADS[wl_name]
    fx = float(str(a.get("class_split", "0,0,1")).split(",")[2])
    return pm.FleetModel(
        large=pool_from(curves[lk], res[lk], wl, int(a["n_large"]), int(a["concurrency_large"]),
                        root, a["large"]),
        small=pool_from(curves[sk], res[sk], wl, int(a.get("n_small") or 0),
                        int(a["concurrency_small"]), root, a["small"]),
        flexible_share=fx)


def held_probes(run: Path) -> list[tuple[float, float, float]]:
    """(lambda, f, measured fleet power) for every probe whose queue held: it started drained
    and its waiting list grew < QUEUE_SLOPE_TOL over the settled half (score_fleet_model)."""
    from analyze.score_fleet_model import QUEUE_SLOPE_TOL, queue_slopes
    sl = queue_slopes(run)
    out = []
    for r in csv.DictReader((run / "points.csv").open()):
        lam, f, p = _f(r["offered_reqs_s"]), _f(r["f"]), _f(r["power_w"])
        slope = sl.get(f"L{lam:g}_f{f:g}", float("nan"))
        pre = str(r.get("drained", "")).lower() in ("true", "1")
        if pre and not (math.isfinite(slope) and slope > QUEUE_SLOPE_TOL) and math.isfinite(p):
            out.append((lam, f, p))
    return out


def measured(run: Path) -> list[dict]:
    """Per offered load: f and power ranges over the probes whose queue held."""
    by: dict[float, list] = {}
    for lam, f, p in held_probes(run):
        by.setdefault(lam, []).append((f, p))
    return [{"lam": lam, "f_lo": min(f for f, _ in g), "f_hi": max(f for f, _ in g),
             "p_lo": min(p for _, p in g), "p_hi": max(p for _, p in g), "n": len(g)}
            for lam, g in sorted(by.items()) if len(g) >= 2]


def band_at(fleet, lam: float, policy: str, solver: str):
    """(f_lo, f_hi, p_lo, p_hi) the model predicts at one offered load, or None.

    exact: the fractions inside its capacity at c_eff, every one of which solves.
    meanfield: reachable_band's rule -- power over the fractions its solve answers."""
    fleet.solver = solver
    fs = (fleet.feasible_fractions_exact(lam, grid=51, policy=policy) if solver == "exact"
          else fleet.feasible_fractions(lam, grid=51))
    ok = []
    for f in fs:
        q = fleet.power(lam, f, policy)["power_total_w"]
        if math.isfinite(q):
            ok.append((f, q))
    if not ok:
        return None
    return (min(f for f, _ in ok), max(f for f, _ in ok),
            min(q for _, q in ok), max(q for _, q in ok))


def errors(fleet, meas: list[dict], policy: str, solver: str) -> dict:
    """Model minus measured at the loads that were measured: f window ends, power ends."""
    e = []
    for m in meas:
        b = band_at(fleet, m["lam"], policy, solver)
        if b is not None:
            e.append((b[0] - m["f_lo"], b[1] - m["f_hi"], b[2] - m["p_lo"], b[3] - m["p_hi"],
                      (b[3] - b[2]) - (m["p_hi"] - m["p_lo"])))
    if not e:
        return {"n": 0, "of": len(meas)}
    a = np.array(e)
    return {"n": len(e), "of": len(meas), "f_lo": a[:, 0].mean(), "f_hi": a[:, 1].mean(),
            "floor": a[:, 2].mean(), "top": a[:, 3].mean(), "range": np.abs(a[:, 4]).mean(),
            "meas_range": float(np.mean([m["p_hi"] - m["p_lo"] for m in meas]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=E / "results/figures/model_vs_load.png")
    ap.add_argument("--platform", choices=("all", "h100", "rtx"), default="all",
                    help="draw only one card's fleet configurations")
    ap.add_argument("--select", default=None,
                    help="regular expression on the configuration label; keep only the rows "
                         "that match (the paper's six-fleet figure against the 1:1-ladder one)")
    args = ap.parse_args()
    import re
    import matplotlib.pyplot as plt
    from analyze import capacity_lookup
    from analyze import plotting as P
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    P.apply_style()

    rows = []
    for label, root, globs in CONFIGS:
        if (args.platform == "h100" and root != H) or (args.platform == "rtx" and root != RX):
            continue
        if args.select and not re.search(args.select, label):
            continue
        runs = runs_of(root, globs)
        if not runs:
            print(f"  skip {label}: no runs")
            continue
        a = json.loads((runs[0] / "summary.json").read_text())["args"]
        cap = capacity_lookup.fleet_capacity(runs[0])
        fleet = model_for(root, a)
        if cap is None or fleet is None:
            print(f"  skip {label}: capacity {cap is not None}, model {fleet is not None}")
            continue
        C = cap["fleet_reqs_s"]
        policy = a.get("policy", "pack")
        nan4 = (float("nan"),) * 4
        pred = [(u, *(band_at(fleet, u * C, policy, "exact") or nan4)) for u in U_GRID]
        meas = []
        for run in runs:
            for m in measured(run):
                m["u"] = m["lam"] / C
                meas.append(m)
        err = errors(fleet, meas, policy, "exact")
        # per held probe: exact-solve power minus measured, against the replicas it leaves dark
        fleet.solver = "exact"
        n_rep = fleet.large.n_replicas + fleet.small.n_replicas
        dark = []
        for run in runs:
            for lam, f, pw in held_probes(run):
                q = fleet.power(lam, f, policy)
                if math.isfinite(q["power_total_w"]):
                    dark.append((q["power_total_w"] - pw, n_rep - q["k_large"] - q["k_small"]))
        rows.append({"label": label, "policy": policy, "C": C, "pred": pred, "meas": meas,
                     "err": err, "n_runs": len(runs), "rtx": root == RX, "dark": dark,
                     "p0": float(fleet.large.curve.power(0))})
        last = max((q[0] for q in pred if math.isfinite(q[1])), default=float("nan"))
        ex = err
        print(f"  {label}: {len(runs)} runs, {len(meas)} measured loads; model window closes "
              f"at u {last:.2f}")
        if ex["n"]:
            print(f"      f window {ex['f_lo']:+.2f} / {ex['f_hi']:+.2f}; power range off "
                  f"{ex['range']:.0f} W of {ex['meas_range']:.0f} (floor {ex['floor']:+.0f}, top "
                  f"{ex['top']:+.0f})")

    n = len(rows)
    H_ROW, H_PAD = 2.3, 2.0
    fig, axes = plt.subplots(n, 2, figsize=(12.5, H_ROW * n + H_PAD), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1.25]})
    for i, r in enumerate(rows):
        u, flo, fhi, plo, phi = (np.array(x) for x in zip(*r["pred"]))
        ax, bx = axes[i]
        ax.fill_between(u, flo, fhi, color=P.CAT[0], alpha=0.30, lw=0, zorder=1)
        bx.fill_between(u, plo, phi, color=P.CAT[0], alpha=0.25, lw=0, zorder=1)
        bx.plot(u, plo, color=P.CAT[0], lw=1.4, zorder=2)
        bx.plot(u, phi, color=P.CAT[0], lw=1.4, zorder=2)
        for m in r["meas"]:
            ax.plot([m["u"], m["u"]], [m["f_lo"], m["f_hi"]], color=P.INK, lw=2.2,
                    solid_capstyle="butt", zorder=3)
            ax.plot([m["u"], m["u"]], [m["f_lo"], m["f_hi"]], "_", color=P.INK, ms=9, mew=2,
                    zorder=3)
            bx.plot([m["u"], m["u"]], [m["p_lo"], m["p_hi"]], color=P.INK, lw=2.2,
                    solid_capstyle="butt", zorder=3)
            bx.plot([m["u"], m["u"]], [m["p_lo"], m["p_hi"]], "_", color=P.INK, ms=9, mew=2,
                    zorder=3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 1.05)
        bx.set_xlim(0, 1.05)
        P.finish(ax, ylabel="f", xlabel="offered load u" if i == n - 1 else None)
        P.finish(bx, ylabel="fleet power (W)", xlabel="offered load u" if i == n - 1 else None)
        ax.set_title(r["label"] + f"  ·  {r['policy']}", loc="left", fontsize=8.6)
        ex = r["err"]
        if ex["n"]:
            loads = f"{ex['n']} measured load" + ("s" if ex["n"] > 1 else "")
            bx.set_title(f"{loads}: power-range error {ex['range']:.0f} W (measured range "
                         f"{ex['meas_range']:.0f} W) · floor {ex['floor']:+.0f} W, top "
                         f"{ex['top']:+.0f} W · window ends {ex['f_lo']:+.2f} / "
                         f"{ex['f_hi']:+.2f}",
                         loc="left", fontsize=7.6, color=P.INK_2)
        else:
            bx.set_title("no measured load inside the model's window", loc="left",
                         fontsize=7.6, color=P.INK_2)
    axes[0][0].set_title("(a) routing fractions both pools can sustain\n"
                         + axes[0][0].get_title(loc="left"), loc="left", fontsize=8.6)
    axes[0][1].set_title("(b) fleet power range over those fractions  (errors: model minus "
                         "measured)\n" + axes[0][1].get_title(loc="left"), loc="left",
                         fontsize=7.6, color=P.INK_2)
    handles = [Patch(color=P.CAT[0], alpha=0.35,
                     label="model, from single-GPU measurements only"),
               Line2D([], [], color=P.INK, lw=2.2, marker="_", ms=9, mew=2,
                      label="measured: lowest to highest over the probes whose queue held")]
    # Anchored in INCHES, between the footnote (bottom ~0.45 in) and the axes (tight_layout
    # keeps them above 1.05 in): a fixed figure fraction drifts up into the bottom row's
    # x-labels as rows are added.
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.2, frameon=False,
               bbox_to_anchor=(0.5, 0.55 / (H_ROW * n + H_PAD)))
    # the RTX bias against dark replicas, fitted over every held RTX probe drawn above. The
    # 14-20 W is nvidia-smi on the RTX 8000's dark cards in the fleet.
    dk = [d for r in rows if r["rtx"] for d in r["dark"]]
    p0 = next((r["p0"] for r in rows if r["rtx"]), float("nan"))
    if len(dk) >= 3:
        e, nd = np.array([d[0] for d in dk]), np.array([d[1] for d in dk], float)
        c1, c0 = np.polyfit(nd, e, 1)
        rtx_note = (f"RTX 8000: model minus measured = {c0:+.0f} W {c1:+.0f} W per dark replica "
                    f"({len(dk)} probes). The model charges a dark card its single-GPU idle power "
                    f"({p0:.0f} W); dark cards in the fleet draw 14–20 W.\n")
        print(f"  RTX dark-replica fit: {c0:+.1f} W {c1:+.1f} W/dark over {len(dk)} held probes")
    else:
        rtx_note = ""
    fig.text(0.01, 0.006,
             "u = offered load ÷ measured fleet capacity; the model's own capacity estimate is "
             "where its window closes. Window ends: model minus measured lowest / highest "
             "sustainable f (measured on a 0.1 grid).\n" + rtx_note.rstrip("\n"),
             fontsize=7.4, color=P.MUTED, va="bottom", ha="left")
    card = {"h100": "  (NVIDIA H100 80GB)", "rtx": "  (Quadro RTX 8000)"}.get(args.platform, "")
    fig.suptitle("Predicted and measured routing window and power range across offered load"
                 + card,
                 x=0.01, ha="left", fontsize=11)
    tot = H_ROW * n + H_PAD
    fig.tight_layout(rect=(0, 1.05 / tot, 1, 1 - 0.3 / tot))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    P.save(fig, args.out, tight=False)


if __name__ == "__main__":
    main()
