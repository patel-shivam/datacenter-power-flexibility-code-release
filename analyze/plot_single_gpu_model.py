#!/usr/bin/env python
"""Single-GPU model figures: predicted against measured, every prediction from the cell's
OWN calibration.

A cell is one (card, model, workload). Its calibration is the latest closed-loop
concurrency sweep (concurrency_sweep) of that cell: the TBT line tau(b) = t0 + t1*b (OLS of the mean
TBT on the achieved concurrency, exactly `fit_power_model.residence_coefficients`) and the
power profile P(b) (the same seven-form fit with every third level held out). Nothing is
transferred between workloads, models or cards.

From that calibration and the workload's two means (s = D + P/S, S = 2048) the script
predicts, and scores against measurement:

  step time      tau(b)                      concurrency_sweep levels, mean TBT
  residence      T(b) = s * tau(b)           concurrency_sweep levels (mean e2e), poisson_load per request
  throughput     mu(b) = b / T(b)            concurrency_sweep levels, requests retired per second
  capacity       C = c_eff / T(c_eff)        capacity_bisection bisections (re-judged, no floor)
  occupancy      b = lam*s*t0 / (1 - lam*s*t1)   capacity_bisection stable probes, poisson_load stable phases
  power          P(b) at the achieved b      concurrency_sweep levels;  P(b(lam)) on capacity_bisection / poisson_load
  drain          t0*max(q) + t1*sum(q)       poisson_load natural drains with oracle residuals
  KV ceiling     floor(K / l_bar), K/(P+D)   kv_ceiling_sweep sweeps past the memory limit

Levels that cannot describe a steady state are excluded from the residence, throughput
and step-time scores: a window shorter than two mean residence times (completions then
over-represent short requests), fewer than 10 completions, or a KV-pinned batch (KV use
>= 97 % or a standing waiting list) -- those are the KV figure's data, not a residence
point. Power is scored on every level that retired a request, as the fit was.

    python analyze/plot_single_gpu_model.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(E))
sys.path.insert(0, str(E / "analyze"))
import plotting as P  # noqa: E402
from lib import power_model as pm  # noqa: E402
from lib import workloads  # noqa: E402
from capacity_lookup import lookup, rejudge  # noqa: E402
import fit_power_model as fpm  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

S = 2048.0
ROOTS = [("h100-80gb", E / "results"),
         ("rtx-8000-48gb", E / "results/rtx-8000-48gb"),
         ("v100-32gb", E / "results/v100-32gb"),
         ("rtx-5090-32gb", E / "results/rtx-5090-32gb")]
HW_LABEL = {"h100-80gb": "H100 80 GB", "rtx-8000-48gb": "RTX 8000 48 GB (fp16)",
            "v100-32gb": "V100 32 GB (fp16)", "rtx-5090-32gb": "RTX 5090 32 GB"}
WL_ORDER = ["sharegpt", "humaneval", "fixed_128_128", "fixed_1024_1024", "fixed_2048_2048",
            "fixed_6144_1024", "arxiv_summarization"]
WL_LABEL = {"sharegpt": "ShareGPT", "humaneval": "HumanEval", "fixed_128_128": "128 in / 128 out",
            "fixed_256_256": "256 / 256", "fixed_512_512": "512 / 512",
            "fixed_1024_1024": "1024 in / 1024 out", "fixed_2048_2048": "2048 in / 2048 out",
            "fixed_6144_1024": "6144 in / 1024 out", "arxiv_summarization": "arXiv summarization",
            "azure_conv_2024": "Azure conversation 2024", "azure_code_2024": "Azure code 2024"}
WL_COLOR = dict(zip(WL_ORDER, P.CAT[:len(WL_ORDER)]))
TABLE_ONLY_WL = ["fixed_256_256", "fixed_512_512"]   # H100-only ladder rungs: scored, not drawn
# Measured for the datacenter simulation (the Azure-length cells): build_cells returns them for datacenter_sim,
# but this figure set leaves them out, so they do not enter the single-GPU tables or headline
# numbers.
DC_ONLY_WL = ("azure_conv_2024", "azure_code_2024")
for _w in DC_ONLY_WL:
    WL_COLOR.setdefault(_w, P.INK_2)
MODEL_ORDER = ["llama32-1b", "qwen3-4b", "llama31-8b", "qwen3-8b",
               "qwen3-1.7b", "qwen3-14b", "llama32-3b", "mistral-7b", "mistral-24b"]
MODEL_MARK = dict(zip(MODEL_ORDER, ["o", "s", "^", "D", "v", "P", "X", "*", "p"]))
MODEL_LABEL = {"llama32-1b": "Llama-3.2-1B", "qwen3-4b": "Qwen3-4B", "llama31-8b": "Llama-3.1-8B",
               "qwen3-8b": "Qwen3-8B", "qwen3-1.7b": "Qwen3-1.7B", "qwen3-14b": "Qwen3-14B",
               "llama32-3b": "Llama-3.2-3B", "mistral-7b": "Mistral-7B", "mistral-24b": "Mistral-24B"}
# freqknob: clock-capped profiles belong to a separate study, never a calibration
SKIP_RUN = ("kvceil", "smoke", "SOLO", "superseded", "freqknob")


# ====================================================================== calibration ====
@dataclass
class Cell:
    hw: str
    model: str
    wl: str
    run: Path
    t0: float            # s
    t1: float            # s per resident request
    r2_tbt: float
    s_len: float         # effective request length s = D + P/S
    curve: pm.PowerCurve | None
    heldout_rmse_w: float
    kv_pool: int | None
    idle_w: float
    measure_s: float
    levels: list[dict] = field(default_factory=list)
    degenerate: bool = False

    def tau(self, b):
        return self.t0 + self.t1 * np.asarray(b, float)

    def T(self, b):
        return self.s_len * self.tau(b)

    def mu(self, b):
        b = np.asarray(b, float)
        return np.where(b > 0, b / self.T(np.maximum(b, 1e-9)), 0.0)

    def occupancy(self, lam):
        """Single-replica fixed point of mu(b) = lam; inf above the pole."""
        a, bb = self.s_len * self.t0, self.s_len * self.t1
        return np.where(lam * bb < 1.0, lam * a / np.maximum(1.0 - lam * bb, 1e-12), np.inf)

    def power(self, b):
        return self.curve.power(b) if self.curve is not None else np.full_like(np.asarray(b, float), np.nan)


def _f(x, default=float("nan")):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# A Poisson phase's window must see the in-flight count filled: a window that opens while the count
# is still filling (a warm-up shorter than the residence, e.g. long fixed shapes on the V100 and RTX
# cards) measures the ramp, and occupancy reads low. Phases whose window saw less than this share of
# the steady count are left out of the occupancy, power-from-rate and per-request residence points
# (their drains stand: the drain prediction is conditioned on the queue actually seen at the stop).
RAMP_FILLED_MIN = 0.95


def ramp_filled(T: float, w0: float, W: float) -> float:
    """Mean share of the steady in-flight count over a window [w0, w0 + W] when the count fills
    from empty as 1 - exp(-t / T), T the phase's mean residence."""
    if not (math.isfinite(T) and T > 0 and math.isfinite(w0) and math.isfinite(W) and W > 0):
        return float("nan")
    return 1.0 - (T / W) * (math.exp(-w0 / T) - math.exp(-(w0 + W) / T))


def latest_concurrency_runs(root: Path) -> dict[tuple[str, str], Path]:
    """(model, workload) -> latest calibration sweep (kv-ceiling / smoke runs excluded)."""
    out: dict[tuple[str, str], Path] = {}
    base = root / "concurrency_sweep"
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir()):
        if any(k in d.name for k in SKIP_RUN) or not (d / "levels.csv").exists():
            continue
        try:
            meta = json.loads((d / "summary.json").read_text())["meta"]
        except (ValueError, KeyError, OSError):
            continue
        if (meta.get("idle") or {}).get("idle_power_w") is None:
            continue
        rows = list(csv.DictReader(open(d / "levels.csv")))
        names = {(r.get("workload") or "").split("(", 1)[0].strip() for r in rows}
        for wl in names:
            if wl and wl in workloads.WORKLOADS:
                out[(meta["model_slug"], wl)] = d          # sorted -> later run wins
    return out


def flag_levels(rows: list[dict], measure_s: float) -> list[dict]:
    out = []
    for r in rows:
        n = int(_f(r.get("n_completed"), 0))
        if n <= 0:
            continue
        b = _f(r.get("concurrency_achieved"))
        e2e_s = _f(r.get("e2e_ms_mean")) / 1e3
        short = (not math.isfinite(e2e_s)) or n < 10 or measure_s < 2.0 * e2e_s
        # KV-pinned: the pool is full, or a STANDING waiting list. The closed-loop client
        # resubmits on completion, so at high c a few requests always sit in the waiting
        # state for one scheduling step (1-3 % of c on the RTX 8000 with KV at 6-30 %);
        # that is scheduling latency, not memory, so the waiting threshold scales with c.
        c_req = _f(r.get("concurrency_requested"), 0.0)
        kv_bound = (_f(r.get("kv_usage_max"), 0.0) >= 0.97
                    or _f(r.get("num_waiting_mean"), 0.0) > max(0.5, 0.05 * c_req))
        out.append({"c_req": _f(r.get("concurrency_requested")), "b": b, "n": n,
                    "tbt_ms": _f(r.get("tbt_ms_mean")), "e2e_s": e2e_s,
                    "thr": _f(r.get("throughput_reqs_s")), "kv_max": _f(r.get("kv_usage_max"), 0.0),
                    "waiting": _f(r.get("num_waiting_mean"), 0.0), "P": _f(r.get("avg_power_w")),
                    "short": bool(short), "kv_bound": bool(kv_bound),
                    "valid": bool(not short and not kv_bound)})
    return out


def build_cells(hw: str, root: Path, verbose=False) -> dict[tuple[str, str], Cell]:
    cells: dict[tuple[str, str], Cell] = {}
    for (model, wl), run in latest_concurrency_runs(root).items():
        try:
            parts = list(fpm.load_levels(run))
        except (ValueError, OSError) as e:
            print(f"  skip {run.name}: {e}")
            continue
        for meta, name, b, p, rows in parts:
            if name != wl:
                continue
            res = fpm.residence_coefficients(rows)
            if not res.get("ok") or len(rows) < 4:
                # a TBT line through three levels is not a calibration
                continue
            curve, held = None, float("nan")
            if len(rows) >= 6:
                cmp = pm.fit_all_forms(b, p, p_idle_w=p[0], holdout_every=3)
                if cmp.get("best"):
                    curve = pm.PowerCurve.from_dict(cmp["forms"][cmp["best"]]["curve"])
                    held = _f(cmp["forms"][cmp["best"]].get("held_out", {}).get("rmse_w"))
            w = workloads.get(wl)
            cells[(model, wl)] = Cell(
                hw=hw, model=model, wl=wl, run=run, t0=res["t0_s"], t1=res["t1_s"],
                r2_tbt=res["r2"], s_len=float(w.mean_decode) + float(w.mean_prefill) / S,
                curve=curve, heldout_rmse_w=held,
                kv_pool=(meta.get("kv") or {}).get("kv_pool_tokens"),
                idle_w=float(p[0]), measure_s=float(meta.get("measure_s", 60.0)),
                levels=flag_levels(rows, float(meta.get("measure_s", 60.0))))
            if verbose:
                c = cells[(model, wl)]
                print(f"  {hw:14s} {model:11s} {wl:20s} t0 {c.t0*1e3:6.3f} ms t1 {c.t1*1e3:7.4f} ms "
                      f"R2 {c.r2_tbt:.4f}  P(b) {curve.form if curve else '-':10s} held-out "
                      f"{held:5.1f} W  levels {sum(l['valid'] for l in c.levels)}/{len(c.levels)} valid")
    # A degenerate intercept: a cell whose t0 is far below the same model's t0 on its other
    # tasks (Llama-3.2-1B on arXiv on the V100: 0.46 ms against 5-6 ms) fits its own levels
    # (R2 0.994) but its line is not a decode time near b = 0, so the occupancy fixed point,
    # which is proportional to t0 at low load, is meaningless there. Flagged, not scored
    # for occupancy and power-from-rate; everything measured at the sweep's own b is kept.
    by_model: dict[str, list[float]] = {}
    for (model, wl), c in cells.items():
        by_model.setdefault(model, []).append(c.t0)
    for (model, wl), c in cells.items():
        others = [t for t in by_model[model] if t != c.t0]
        c.degenerate = bool(others) and c.t0 < 0.25 * float(np.median(others))
        if c.degenerate:
            print(f"  degenerate calibration: {hw} {model} {wl} t0 {c.t0*1e3:.2f} ms against a median "
                  f"{1e3*float(np.median(others)):.2f} ms on the model's other tasks -- not used for the fixed point")
    return cells


# ====================================================================== data sources ===
def capacity_cells(root: Path) -> dict[tuple[str, str, int], dict]:
    out: dict[tuple[str, str, int], dict] = {}
    base = root / "capacity_bisection"
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir()):
        f = d / "capacity.json"
        if not f.exists() or "smoke" in d.name.lower():   # shakeout runs are never a measurement
            continue
        try:
            j = json.loads(f.read_text())
        except ValueError:
            continue
        wl = (j.get("workload") or {}).get("name")
        if not wl or not j.get("probes"):
            continue
        key = (j["model_slug"], wl, int(j.get("max_num_seqs", -1)))
        judged = []
        prev_residual = False
        for p in j["probes"]:
            stable = bool(p.get("stable")) if j.get("stability_rule") else rejudge(p)
            # A probe that started on the previous probe's leftover backlog (the reset
            # before it left requests waiting or running) runs with its running count
            # pinned at the cap while that backlog drains: its stability verdict is
            # still sign-correct (a draining queue means the rate is below capacity),
            # but its running_mean and power are not the steady state of that rate.
            judged.append({**p, "stable": stable, "inherited": prev_residual})
            prev_residual = bool((p.get("reset_residual_waiting") or 0) > 0
                                 or (p.get("reset_residual_running") or 0) > 0)
        # A bisection whose LOWEST diverging probe ran on inherited load bounds the
        # capacity with a verdict the backlog made. Such a file's capacity is not used.
        bad = [q for q in judged if not q["stable"]]
        first_bad = min(bad, key=lambda q: q["rate_reqs_s"]) if bad else None
        corrupt = bool(first_bad is not None and first_bad.get("inherited"))
        if corrupt:
            print(f"  capacity bisection corrupted by an inherited backlog: {d.name} "
                  f"(lowest diverging probe {first_bad['rate_reqs_s']} req/s) -- not scored")
        out[key] = {"run": d, "probes": judged, "kv_pool": (j.get("kv") or {}).get("kv_pool_tokens"),
                    "probe_s": j.get("probe_s"), "warmup_s": j.get("warmup_s"),
                    "rule": bool(j.get("stability_rule")), "corrupt": corrupt}
    return out


def poisson_runs(root: Path) -> list[dict]:
    out = []
    base = root / "poisson_load"
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir()):
        f = d / "summary.json"
        if not f.exists() or "smoke" in d.name:
            continue
        try:
            j = json.loads(f.read_text())
        except ValueError:
            continue
        reqs = list(csv.DictReader(open(d / "requests.csv"))) if (d / "requests.csv").exists() else []
        out.append({"run": d, "meta": j["meta"], "phases": j["phases"], "requests": reqs})
    return out


def kvceil_points(root: Path, cells: dict) -> list[dict]:
    """Observed KV plateau per (model, workload) from kv_ceiling_sweep sweeps and incidental KV-bound levels."""
    pts = []
    base = root / "concurrency_sweep"
    if not base.is_dir():
        return pts
    seen = set()
    for d in sorted(base.iterdir(), reverse=True):          # latest first
        if "kvceil" not in d.name or not (d / "levels.csv").exists():
            continue
        try:
            meta = json.loads((d / "summary.json").read_text())["meta"]
        except (ValueError, KeyError, OSError):
            continue
        rows = list(csv.DictReader(open(d / "levels.csv")))
        for wl in {(r.get("workload") or "") for r in rows}:
            if wl not in workloads.WORKLOADS or (meta["model_slug"], wl) in seen:
                continue
            lv = [r for r in rows if r.get("workload") == wl]
            b = np.array([_f(r["concurrency_achieved"]) for r in lv])
            c = np.array([_f(r["concurrency_requested"]) for r in lv])
            kv = np.array([_f(r.get("kv_usage_max"), 0.0) for r in lv])
            if not len(b):
                continue
            i = int(np.nanargmax(b))
            reached = c.max() >= 1.1 * b[i] and kv[i] >= 0.9
            w = workloads.get(wl)
            K = (meta.get("kv") or {}).get("kv_pool_tokens")
            if not K:
                continue
            seen.add((meta["model_slug"], wl))
            pts.append({"model": meta["model_slug"], "wl": wl, "K": K,
                        "observed": float(b[i]), "c_req_at": float(c[i]), "kv_at": float(kv[i]),
                        "c_req_max": float(c.max()), "reached": bool(reached),
                        "pred_dw": K / w.decode_weighted_context,
                        "pred_peak": K / (w.mean_prefill + w.mean_decode), "source": d.name})
    # incidental: calibration sweeps whose top levels pinned the batch with KV full
    for (model, wl), cell in cells.items():
        if (model, wl) in seen or not cell.kv_pool:
            continue
        bound = [l for l in cell.levels if l["kv_max"] >= 0.97 and l["c_req"] >= 1.1 * l["b"]]
        if not bound:
            continue
        top = max(bound, key=lambda l: l["b"])
        w = workloads.get(wl)
        pts.append({"model": model, "wl": wl, "K": cell.kv_pool, "observed": top["b"],
                    "c_req_at": top["c_req"], "kv_at": top["kv_max"],
                    "c_req_max": max(l["c_req"] for l in cell.levels), "reached": True,
                    "pred_dw": cell.kv_pool / w.decode_weighted_context,
                    "pred_peak": cell.kv_pool / (w.mean_prefill + w.mean_decode),
                    "source": cell.run.name + " (calibration sweep)"})
    return pts


# ====================================================================== offline drain ==
_DECODE_DIST: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _decode_distribution(wl_name: str) -> tuple[np.ndarray, np.ndarray]:
    """(output lengths, length-biased weights) of a workload, sampled once and cached."""
    if wl_name not in _DECODE_DIST:
        w = workloads.get(wl_name)
        if w.kind == "fixed":
            D, wts = np.array([float(w.decode)]), np.array([1.0])
        else:
            pairs = w.sample(50_000, random.Random(0))
            D = np.array([d for _, d in pairs], float)
            wts = D / D.sum()
        _DECODE_DIST[wl_name] = (D, wts)
    return _DECODE_DIST[wl_name]


def offline_drain(cell: Cell, n_inflight: int, trials: int = 400, seed: int = 0) -> float:
    """The paper's offline estimate: residuals are length-biased draws from the output-length
    distribution at a uniformly random generation position; median over trials of
    t0*max(q) + t1*sum(q)."""
    if n_inflight <= 0:
        return 0.0
    D, wts = _decode_distribution(cell.wl)
    nr = np.random.default_rng(seed)
    out = np.empty(trials)
    for t in range(trials):
        d = nr.choice(D, size=n_inflight, p=wts)
        q = np.ceil(nr.uniform(size=n_inflight) * d)
        out[t] = cell.t0 * q.max() + cell.t1 * q.sum()
    return float(np.median(out))


# ====================================================================== scoring ========
def stats(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    if not len(x):
        return {"n": 0}
    r = y / x
    ape = np.abs(x - y) / y
    return {"n": int(len(x)), "median_ratio": float(np.median(r)),
            "p10_ratio": float(np.percentile(r, 10)), "p90_ratio": float(np.percentile(r, 90)),
            "mape_pct": float(100 * ape.mean()), "within_10pct": float(np.mean(ape <= 0.10)),
            "within_20pct": float(np.mean(ape <= 0.20))}


# ====================================================================== drawing ========
def legend_handles(models_present, wls_present):
    hw = [Patch(facecolor=WL_COLOR[w], edgecolor="none", label=WL_LABEL[w])
          for w in WL_ORDER if w in wls_present]
    hm = [Line2D([], [], marker=MODEL_MARK[m], color=P.INK_2, linestyle="none", markersize=6,
                 label=MODEL_LABEL.get(m, m)) for m in MODEL_ORDER if m in models_present]
    return hw + hm


def _figure_legend(fig, handles, title):
    ncol = min(6, max(1, math.ceil(len(handles) / 2)))
    rows = math.ceil(len(handles) / ncol)
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.06 + 0.055 * rows, 1, 0.95))


def parity(points: list[dict], *, title: str, xlabel: str, ylabel: str, path: Path,
           log: bool = True, hollow_key: str | None = None, hollow_label: str = "",
           exclude_key: str | None = None, point_alpha: float = 0.9, size: float = 28.0) -> dict:
    """One panel per card; y = x line with a 0.8-1.25x band. Hollow markers are drawn but
    excluded from the panel score when `exclude_key` (default: the hollow key) is set."""
    exclude_key = exclude_key or hollow_key
    hws = [h for h, _ in ROOTS if any(p["hw"] == h for p in points)]
    if not hws:
        print(f"  (no points for {path.name})")
        return {}
    fig, axes = plt.subplots(1, len(hws), figsize=(3.9 * len(hws) + 0.4, 4.6), squeeze=False)
    out_stats = {}
    for ax, hw in zip(axes[0], hws):
        pts = [p for p in points if p["hw"] == hw]
        xs = np.array([p["x"] for p in pts]); ys = np.array([p["y"] for p in pts])
        lo = np.nanmin(np.concatenate([xs, ys])) * 0.8
        hi = np.nanmax(np.concatenate([xs, ys])) * 1.25
        if not np.isfinite(lo) or lo <= 0:
            lo = 1e-3
        ax.fill_between([lo, hi], [0.8 * lo, 0.8 * hi], [1.25 * lo, 1.25 * hi],
                        color=P.GRID, alpha=0.6, linewidth=0, zorder=0)
        ax.plot([lo, hi], [lo, hi], color=P.MUTED, linewidth=1.0, zorder=1)
        # one scatter call per (model, workload, hollow) group: the per-request figure
        # has tens of thousands of points and a call per point would take minutes
        groups: dict[tuple, list] = {}
        for p in pts:
            hollow = bool(hollow_key and p.get(hollow_key))
            groups.setdefault((p["model"], p["wl"], hollow), []).append(p)
        for (model, wl, hollow), g in groups.items():
            ax.scatter([p["x"] for p in g], [p["y"] for p in g], s=size,
                       marker=MODEL_MARK.get(model, "o"),
                       facecolor="none" if hollow else WL_COLOR.get(wl, P.MUTED),
                       edgecolor=WL_COLOR.get(wl, P.MUTED), linewidth=0.9,
                       alpha=point_alpha, zorder=3)
        scored = [p for p in pts if not (exclude_key and p.get(exclude_key))]
        st = stats([p["x"] for p in scored], [p["y"] for p in scored])
        out_stats[hw] = st
        txt = (f"n = {st['n']}\nmedian meas/pred {st['median_ratio']:.2f}\n"
               f"MAPE {st['mape_pct']:.0f} %" if st.get("n") else "no points")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
                color=P.INK_2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        P.finish(ax, xlabel=xlabel, ylabel=ylabel if ax is axes[0][0] else None,
                 title=HW_LABEL.get(hw, hw), logx=log, logy=log)
    handles = legend_handles({p["model"] for p in points}, {p["wl"] for p in points})
    if hollow_key:
        handles.append(Line2D([], [], marker="o", color=P.INK_2, markerfacecolor="none",
                              linestyle="none", markersize=6, label=hollow_label))
    _figure_legend(fig, handles, title)
    P.save(fig, path, tight=False)
    return out_stats


def ratio_vs_utilization(points: list[dict], *, title: str, path: Path, hollow_key: str = "kv_bound",
                         hollow_label: str = "KV-pinned (b >= 0.9 K / l_bar)") -> None:
    """Measured / predicted against the offered load as a fraction of the MEASURED capacity:
    where the fixed point is trustworthy and where it is not."""
    pts = [p for p in points if math.isfinite(p.get("u", float("nan"))) and p["x"] > 0 and p["y"] > 0]
    hws = [h for h, _ in ROOTS if any(p["hw"] == h for p in pts)]
    if not hws:
        print(f"  (no points for {path.name})")
        return
    fig, axes = plt.subplots(1, len(hws), figsize=(3.9 * len(hws) + 0.4, 4.2), squeeze=False)
    for ax, hw in zip(axes[0], hws):
        sel = [p for p in pts if p["hw"] == hw]
        ax.axhspan(0.8, 1.25, color=P.GRID, alpha=0.6, linewidth=0, zorder=0)
        ax.axhline(1.0, color=P.MUTED, linewidth=1.0, zorder=1)
        groups: dict[tuple, list] = {}
        for p in sel:
            groups.setdefault((p["model"], p["wl"], bool(p.get(hollow_key))), []).append(p)
        for (model, wl, hollow), g in groups.items():
            ax.scatter([p["u"] for p in g], [p["y"] / p["x"] for p in g], s=26,
                       marker=MODEL_MARK.get(model, "o"),
                       facecolor="none" if hollow else WL_COLOR.get(wl, P.MUTED),
                       edgecolor=WL_COLOR.get(wl, P.MUTED), linewidth=0.9, zorder=3)
        for lo_u, hi_u in ((0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.5)):
            g = [p for p in sel if lo_u <= p["u"] < hi_u and not p.get(hollow_key)]
            if g:
                r = np.array([p["y"] / p["x"] for p in g])
                print(f"    {path.stem} {hw:14s} u in [{lo_u:.2f},{hi_u:.2f}): n {len(g):3d}  median ratio "
                      f"{np.median(r):.2f}  p10-p90 {np.percentile(r, 10):.2f}-{np.percentile(r, 90):.2f}")
        ax.set_xlim(0, 1.15)
        P.finish(ax, xlabel="offered load / measured capacity", ylabel="measured / predicted" if ax is axes[0][0] else None,
                 title=HW_LABEL.get(hw, hw), logy=True)
    handles = legend_handles({p["model"] for p in pts}, {p["wl"] for p in pts})
    handles.append(Line2D([], [], marker="o", color=P.INK_2, markerfacecolor="none", linestyle="none",
                          markersize=6, label=hollow_label))
    _figure_legend(fig, handles, title)
    P.save(fig, path, tight=False)


# ====================================================================== TBT law figure =
CORE_MODELS = ("llama32-1b", "qwen3-4b", "llama31-8b", "qwen3-8b")


def tbt_law_figures(cells_by_hw: dict, out: Path) -> list[dict]:
    """Per card: the fitted lines tau(b) = t0 + t1 b with the measured mean TBT of every
    level on them (one panel per model, one line per workload), and right beside them the
    t0 and t1 bars across workloads. Returns the per-(card, model) transfer numbers."""
    rows = []
    for hw, cells in cells_by_hw.items():
        bar_models = [m for m in MODEL_ORDER
                      if sum(1 for (mm, wl) in cells if mm == m and wl in WL_ORDER) >= 2]
        # line panels: the core models, or every model with >= 3 workloads; the bars take all
        line_models = [m for m in bar_models if m in CORE_MODELS] or \
                      [m for m in bar_models if sum(1 for (mm, wl) in cells if mm == m and wl in WL_ORDER) >= 3] or \
                      bar_models
        if not bar_models:
            continue
        n = len(line_models)
        bar_w = max(1.2, 0.3 * len(bar_models))
        fig, axes = plt.subplots(1, n + 2, figsize=(3.2 * n + 2 * 2.6 * bar_w / 1.2 + 0.6, 3.9), squeeze=False,
                                 gridspec_kw={"width_ratios": [1.0] * n + [bar_w, bar_w]})
        axes = axes[0]
        for ax, m in zip(axes[:n], line_models):
            ys_all = []
            for wl in WL_ORDER:
                c = cells.get((m, wl))
                if c is None:
                    continue
                pts = [(l["b"], l["tbt_ms"], l["valid"]) for l in c.levels
                       if l["b"] > 0 and math.isfinite(l["tbt_ms"])]
                if not pts:
                    continue
                bmax = max(b for b, _, _ in pts)
                bb = np.linspace(0.0, bmax * 1.04, 60)
                ax.plot(bb, 1e3 * c.tau(bb), color=WL_COLOR[wl], linewidth=1.4, zorder=2)
                ys_all += [t for _, t, _ in pts]
                for want, face in ((True, WL_COLOR[wl]), (False, "none")):
                    sel = [(b, t) for b, t, v in pts if v == want]
                    if sel:
                        ax.scatter([b for b, _ in sel], [t for _, t in sel], s=22,
                                   marker=MODEL_MARK.get(m, "o"), facecolor=face,
                                   edgecolor=WL_COLOR[wl], linewidth=0.9, zorder=3)
            # a slow card's arXiv line (t1 ~ 10 ms) would squash every other workload on a
            # linear axis: use a log TBT axis when the panel spans more than ~15x
            wide = bool(ys_all) and max(ys_all) / max(min(ys_all), 1e-9) > 15
            ax.set_xlim(left=0)
            if not wide:
                ax.set_ylim(bottom=0)
            P.finish(ax, xlabel="resident requests b",
                     ylabel="mean TBT (ms)" if ax is axes[0] else None,
                     title=MODEL_LABEL.get(m, m) + (" (log TBT)" if wide else ""), logy=wide)
        wls = [wl for wl in WL_ORDER if any((m, wl) in cells for m in bar_models)]
        width = 0.8 / max(len(wls), 1)
        for ax, key, label, logy in ((axes[n], "t0", "t0 (ms)", False),
                                     (axes[n + 1], "t1", "t1 (ms per resident request)", True)):
            for j, wl in enumerate(wls):
                xs = [i + (j - (len(wls) - 1) / 2) * width for i, m in enumerate(bar_models) if (m, wl) in cells]
                ys = [1e3 * getattr(cells[(m, wl)], key) for m in bar_models if (m, wl) in cells]
                ax.bar(xs, ys, width=width * 0.95, color=WL_COLOR[wl], linewidth=0, zorder=2)
            ax.set_xticks(range(len(bar_models)))
            ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in bar_models], rotation=30, ha="right", fontsize=7.5)
            P.finish(ax, ylabel=label, title=f"{key} by workload", logy=logy)
        models = bar_models
        handles = [Patch(facecolor=WL_COLOR[w], edgecolor="none", label=WL_LABEL[w]) for w in wls]
        handles.append(Line2D([], [], marker="o", color=P.INK_2, markerfacecolor="none", linestyle="none",
                              markersize=6, label="level excluded from the steady-state scores"))
        _figure_legend(fig, handles, f"Step time law on the {HW_LABEL.get(hw, hw)}: tau(b) = t0 + t1 b "
                                     f"(lines) and the measured mean TBT of every closed-loop level")
        P.save(fig, out / f"f8_tbt_law__{hw}.png", tight=False)
        for m in models:
            cs = [cells[(m, wl)] for wl in WL_ORDER + TABLE_ONLY_WL if (m, wl) in cells]
            t0s = [1e3 * c.t0 for c in cs]
            t1s = [1e3 * c.t1 for c in cs]
            rows.append({"hw": hw, "model": m, "n_workloads": len(cs),
                         "t0_min_ms": min(t0s), "t0_max_ms": max(t0s),
                         "t0_spread_pct": 100 * (max(t0s) / min(t0s) - 1),
                         "t1_min_ms": min(t1s), "t1_max_ms": max(t1s),
                         "t1_ratio": max(t1s) / min(t1s),
                         "r2_min": min(c.r2_tbt for c in cs), "r2_max": max(c.r2_tbt for c in cs)})
    with open(out / "single_gpu_tbt_law.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["hw"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    return rows


F9_SKIP_WL = ("fixed_6144_1024",)


def tbt_core_figures(cells_by_hw: dict, out: Path) -> None:
    """Per card: one panel per core model with its TBT lines across the tasks (colour =
    task, marker = model, linear axes, steady-state levels only, no 6144/1024) and beside
    them the t1 of those models with the task on the x axis (bars in the task's colour,
    the model's marker on top; linear y)."""
    for hw, cells in cells_by_hw.items():
        models = [m for m in CORE_MODELS if any((m, wl) in cells for wl in WL_ORDER)]
        wls = [wl for wl in WL_ORDER if wl not in F9_SKIP_WL and any((m, wl) in cells for m in models)]
        if not models or not wls:
            continue
        n = len(models)
        fig, axes = plt.subplots(1, n + 1, figsize=(3.1 * n + 4.6, 4.2), squeeze=False,
                                 gridspec_kw={"width_ratios": [1.0] * n + [1.5]})
        axes = axes[0]
        axb = axes[n]
        for ax, m in zip(axes[:n], models):
            for wl in wls:
                c = cells.get((m, wl))
                if c is None:
                    continue
                # steady-state levels only (the fit used every level; the line is drawn
                # over the range the shown levels cover)
                pts = [(l["b"], l["tbt_ms"]) for l in c.levels
                       if l["b"] > 0 and math.isfinite(l["tbt_ms"]) and l["valid"]]
                if not pts:
                    continue
                bmax = max(b for b, _ in pts)
                bb = np.linspace(0.0, bmax * 1.04, 60)
                ax.plot(bb, 1e3 * c.tau(bb), color=WL_COLOR[wl], linewidth=1.2, zorder=2)
                ax.scatter([b for b, _ in pts], [t for _, t in pts], s=24,
                           marker=MODEL_MARK.get(m, "o"), facecolor=WL_COLOR[wl],
                           edgecolor=WL_COLOR[wl], linewidth=0.9, zorder=3)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            P.finish(ax, xlabel="resident requests b",
                     ylabel="mean TBT (ms)" if ax is axes[0] else None,
                     title=MODEL_LABEL.get(m, m))
        width = 0.8 / len(models)
        for i, wl in enumerate(wls):
            for j, m in enumerate(models):
                c = cells.get((m, wl))
                if c is None:
                    continue
                x = i + (j - (len(models) - 1) / 2) * width
                y = 1e3 * c.t1
                axb.bar(x, y, width=width * 0.92, color=WL_COLOR[wl], linewidth=0, zorder=2)
                axb.scatter([x], [y], s=26, marker=MODEL_MARK.get(m, "o"), facecolor=P.SURFACE,
                            edgecolor=P.INK, linewidth=0.8, zorder=4)
        axb.set_xticks(range(len(wls)))
        axb.set_xticklabels([WL_LABEL[w] for w in wls], rotation=30, ha="right", fontsize=8)
        axb.set_ylim(bottom=0)
        P.finish(axb, ylabel="t1 (ms per resident request)", title="t1 by task, the four core models")
        handles = [Patch(facecolor=WL_COLOR[w], edgecolor="none", label=WL_LABEL[w]) for w in wls]
        handles += [Line2D([], [], marker=MODEL_MARK[m], color=P.INK_2, linestyle="none", markersize=6,
                           label=MODEL_LABEL[m]) for m in models]
        _figure_legend(fig, handles, f"Step time across tasks on the {HW_LABEL.get(hw, hw)}: "
                                     f"tau(b) = t0 + t1 b per task (steady-state levels), and t1 by task")
        P.save(fig, out / f"f9_tbt_core__{hw}.png", tight=False)


# ====================================================================== markdown =======
FIG_LABEL = {"tbt": "F7 step time", "residence_level": "F1b residence (closed-loop levels)",
             "residence_request": "F1a residence (per request, Poisson)",
             "throughput_level": "F3a throughput (closed loop, window-corrected)",
             "capacity": "F3b capacity", "occupancy": "F2 occupancy (Poisson)", "drain": "F4 drain",
             "kv_ceiling": "F5 KV ceiling", "power_level": "F6a power at b",
             "power_rate": "F6b power from rate"}


def emit_markdown_tables(out: Path) -> None:
    """The stats, calibration and workload tables as Markdown, for pasting into the report."""
    import pandas as pd

    st = pd.read_csv(out / "single_gpu_stats.csv")
    cal = pd.read_csv(out / "single_gpu_calibration.csv")
    wl = pd.read_csv(out / "workload_lengths.csv")
    lines = ["# Tables generated by plot_single_gpu_model.py", ""]
    hw_order = [h for h, _ in ROOTS]
    for fig in FIG_LABEL:
        sub = st[(st.figure == fig) & (st.workload.isna())]
        if sub.empty:
            continue
        lines += [f"## {FIG_LABEL[fig]}", "",
                  "| card | n | median meas/pred | p10–p90 | MAPE | within 10 % / 20 % |",
                  "|---|---|---|---|---|---|"]
        for hw in hw_order:
            r = sub[sub.hw == hw]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(f"| {HW_LABEL.get(hw, hw)} | {int(r.n)} | {r.median_ratio:.3f} | "
                         f"{r.p10_ratio:.2f}–{r.p90_ratio:.2f} | {r.mape_pct:.1f} % | "
                         f"{100*r.within_10pct:.0f} % / {100*r.within_20pct:.0f} % |")
        lines.append("")
        # per-workload rows for the H100 (the card with every workload)
        subw = st[(st.figure == fig) & (st.workload.notna()) & (st.hw == "h100-80gb")]
        if not subw.empty:
            lines += ["H100 by workload:", "", "| workload | n | median | p10–p90 | MAPE |", "|---|---|---|---|---|"]
            for w in WL_ORDER + TABLE_ONLY_WL:
                r = subw[subw.workload == w]
                if r.empty:
                    continue
                r = r.iloc[0]
                lines.append(f"| {WL_LABEL.get(w, w)} | {int(r.n)} | {r.median_ratio:.3f} | "
                             f"{r.p10_ratio:.2f}–{r.p90_ratio:.2f} | {r.mape_pct:.1f} % |")
            lines.append("")
    lines += ["## Calibration cells", "",
              "| card | cells | t0 (ms) | t1 (ms/request) | R² of tau(b) | P(b) held-out RMSE (cells fitted) |",
              "|---|---|---|---|---|---|"]
    for hw in hw_order:
        c = cal[cal.hw == hw]
        if c.empty:
            continue
        fitted = c[c.heldout_rmse_w.notna()]
        lines.append(f"| {HW_LABEL.get(hw, hw)} | {len(c)} | {c.t0_ms.min():.2f}–{c.t0_ms.max():.2f} | "
                     f"{c.t1_ms.min():.3f}–{c.t1_ms.max():.3f} | {c.r2_tbt.min():.3f}–{c.r2_tbt.max():.3f} | "
                     f"{fitted.heldout_rmse_w.min():.1f}–{fitted.heldout_rmse_w.max():.1f} W ({len(fitted)}) |")
    lines += ["", "Per cell:", "", "| card | model | workload | t0 (ms) | t1 (ms) | R² | P(b) form | held-out RMSE (W) | steady levels |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in cal.sort_values(["hw", "model", "workload"]).iterrows():
        lines.append(f"| {HW_LABEL.get(r.hw, r.hw)} | {MODEL_LABEL.get(r.model, r.model)} | {WL_LABEL.get(r.workload, r.workload)} | "
                     f"{r.t0_ms:.3f} | {r.t1_ms:.4f} | {r.r2_tbt:.4f} | {r.form if isinstance(r.form, str) else '—'} | "
                     f"{'—' if pd.isna(r.heldout_rmse_w) else f'{r.heldout_rmse_w:.1f}'} | {int(r.levels_valid)}/{int(r.levels_total)} |")
    lines += ["", "## Workloads", "",
              "| workload | rows | prompt P: mean ± std (p50, p99) | output D: mean ± std (p50, p99) | corr(P, D) | l_bar | P+D | s |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in wl.iterrows():
        corr = "—" if pd.isna(r.corr_PD) else f"{r.corr_PD:+.2f}"
        lines.append(f"| {WL_LABEL.get(r.workload, r.workload)} | {r.n_rows} | {r.mean_P:.0f} ± {r.std_P:.0f} ({r.p50_P:.0f}, {r.p99_P:.0f}) | "
                     f"{r.mean_D:.0f} ± {r.std_D:.0f} ({r.p50_D:.0f}, {r.p99_D:.0f}) | {corr} | {r.l_bar:.0f} | {r.P_plus_D:.0f} | {r.s:.1f} |")
    (out / "single_gpu_tables.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'single_gpu_tables.md'}")


# ====================================================================== main ===========
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=E / "results/tables/single_gpu")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    P.apply_style()

    all_points: list[dict] = []
    all_stats: list[dict] = []
    cells_by_hw: dict[str, dict] = {}
    for hw, root in ROOTS:
        if not root.is_dir():
            continue
        print(f"\n== {hw}: calibration cells")
        cells_by_hw[hw] = {k: c for k, c in build_cells(hw, root, verbose=args.verbose).items()
                           if k[1] not in DC_ONLY_WL}

    def add(fig, hw, cell_or_model, wl, x, y, **extra):
        model = cell_or_model.model if isinstance(cell_or_model, Cell) else cell_or_model
        p = {"figure": fig, "hw": hw, "model": model, "wl": wl, "x": float(x), "y": float(y), **extra}
        all_points.append(p)
        return p

    drawable = lambda wl: wl in WL_ORDER  # noqa: E731

    # ---------------- F7 step time, F1b residence, F3a throughput, F6a power (concurrency_sweep) ----
    pts_tbt, pts_res, pts_thr, pts_pow = [], [], [], []
    for hw, cells in cells_by_hw.items():
        for (model, wl), c in cells.items():
            for l in c.levels:
                b = l["b"]
                if b <= 0:
                    continue
                pp = add("power_level", hw, c, wl, c.power(b), l["P"], b=b, kv_bound=l["kv_bound"], short=l["short"])
                if drawable(wl) and math.isfinite(pp["x"]):
                    pts_pow.append(pp)
                if not l["valid"]:
                    continue
                pt = add("tbt", hw, c, wl, 1e3 * c.tau(b), l["tbt_ms"], b=b)
                pr = add("residence_level", hw, c, wl, c.T(b), l["e2e_s"], b=b)
                # concurrency_sweep counts a completion only if the request was ALSO submitted inside
                # the window, so in steady state it misses the fraction T/W that was
                # submitted before the window opened: a measurement rule, corrected here
                # with the measured residence (W/(W - T)); the raw value is kept alongside.
                W = c.measure_s
                corr = W / (W - l["e2e_s"]) if W > l["e2e_s"] else float("nan")
                pm_ = add("throughput_level", hw, c, wl, c.mu(b), l["thr"] * corr, b=b,
                          y_raw=l["thr"], window_corr=corr)
                if drawable(wl):
                    pts_tbt.append(pt); pts_res.append(pr); pts_thr.append(pm_)

    # ---------------- capacity_bisection: occupancy (F2), capacity (F3b), power from rate (F6b) ------
    pts_occ, pts_cap, pts_prate = [], [], []
    ramp_rows = []           # Poisson probes / phases whose window opened on a filling replica
    for hw, root in ROOTS:
        cells = cells_by_hw.get(hw) or {}
        for (model, wl, cap), e in capacity_cells(root).items():
            c = cells.get((model, wl))
            if c is None:
                continue
            w = workloads.get(wl)
            K = e["kv_pool"] or c.kv_pool
            c_kv = K / w.decode_weighted_context if K else float("inf")
            c_eff = min(cap, int(c_kv)) if K else cap
            meas = lookup(root, model, wl, cap)
            cap_meas = float(meas["reqs_s"]) if meas and meas.get("reqs_s") else float("nan")
            if e.get("corrupt") and meas and meas.get("source") == e["run"].name:
                cap_meas = float("nan")          # the latest file is the corrupted one
            if math.isfinite(cap_meas):
                pc = add("capacity", hw, c, wl, float(c.mu(c_eff)), cap_meas, cap=cap, c_eff=c_eff)
                if drawable(wl):
                    pts_cap.append(pc)
            for p in e["probes"]:
                if not p["stable"] or p.get("inherited") or c.degenerate:
                    continue
                lam = _f(p.get("realized_rate"), _f(p.get("rate_reqs_s")))
                bm = _f(p.get("running_mean"))
                if not (math.isfinite(lam) and math.isfinite(bm)) or bm <= 0:
                    continue
                bh = float(c.occupancy(lam))
                clamped = not math.isfinite(bh) or bh > cap
                bh_plot = min(bh, cap) if math.isfinite(bh) else cap
                # The same ramp can bias a probe, but a probe records no residence of its own, so
                # its fill can only be ESTIMATED with the model's T(b) -- and the model must not
                # decide which data score it. Recorded as a diagnostic, still scored; the cure is
                # re-measuring such probes with a longer warm-up.
                filled = ramp_filled(float(c.T(bh_plot)), _f(e.get("warmup_s")), _f(e.get("probe_s")))
                if math.isfinite(filled) and filled < RAMP_FILLED_MIN:
                    ramp_rows.append({"hw": hw, "model": model, "workload": wl, "run": e["run"].name,
                                      "source": "capacity_bisection", "phase": f"probe {p.get('rate_reqs_s')}",
                                      "rate_reqs_s": lam, "residence_s": round(float(c.T(bh_plot)), 2),
                                      "residence_from": "model T(b) at the predicted occupancy",
                                      "warmup_s": e.get("warmup_s"), "window_s": e.get("probe_s"),
                                      "filled": round(filled, 3), "left_out": False})
                # a probe whose resident count sits at the KV ceiling is preempting and
                # re-prefilling: outside the model (the paper's c = min(c, K / l_bar) says
                # so), drawn hollow and left out of the score
                kv_bound = bm >= 0.9 * c_kv
                u = lam / cap_meas if math.isfinite(cap_meas) and cap_meas > 0 else float("nan")
                po = add("occupancy", hw, c, wl, bh_plot, bm, lam=lam, cap=cap, clamped=clamped,
                         kv_bound=kv_bound, hollow=(clamped or kv_bound), u=u, source="capacity_bisection")
                if drawable(wl):
                    pts_occ.append(po)
                pw = _f(p.get("avg_power_w"))
                if math.isfinite(pw) and c.curve is not None:
                    pr = add("power_rate", hw, c, wl, float(c.power(bh_plot)), pw, lam=lam, clamped=clamped,
                             kv_bound=kv_bound, hollow=(clamped or kv_bound), u=u, source="capacity_bisection")
                    if drawable(wl):
                        pts_prate.append(pr)

    # ---------------- poisson_load: per-request residence (F1a), occupancy, power, drains (F4) -
    pts_req, pts_drain = [], []
    for hw, root in ROOTS:
        cells = cells_by_hw.get(hw) or {}
        for r in poisson_runs(root):
            model = r["meta"]["model_slug"]
            cap = int(r["meta"]["max_num_seqs"])
            by_phase = {}
            for q in r["requests"]:
                by_phase.setdefault(int(q["phase"]), []).append(q)
            for ph in r["phases"]:
                wl = ph["workload"]
                c = cells.get((model, wl))
                if c is None:
                    continue
                filled = ramp_filled(_f(ph.get("e2e_s_mean")), _f(ph.get("warmup_s")), _f(ph.get("measure_s")))
                ramp = bool(ph.get("stable")) and math.isfinite(filled) and filled < RAMP_FILLED_MIN
                if ramp:
                    ramp_rows.append({"hw": hw, "model": model, "workload": wl, "run": r["run"].name,
                                      "source": "poisson_load", "phase": ph["phase"], "rate_reqs_s": ph.get("rate_reqs_s"),
                                      "residence_s": ph.get("e2e_s_mean"), "residence_from": "measured (phase e2e)",
                                      "warmup_s": ph.get("warmup_s"), "window_s": ph.get("measure_s"),
                                      "filled": round(filled, 3), "left_out": True})
                lam = _f(ph.get("realized_rate_reqs_s"), _f(ph.get("rate_reqs_s")))
                bm = _f(ph.get("running_mean"))
                ref = ((r["meta"].get("rate_reference") or {}).get(wl) or {})
                cap_meas = _f(ref.get("reqs_s")) if str(ref.get("provenance", "")).startswith("measured") else float("nan")
                if not math.isfinite(cap_meas):
                    m2 = lookup(root, model, wl, cap)
                    cap_meas = float(m2["reqs_s"]) if m2 and m2.get("reqs_s") else float("nan")
                u = lam / cap_meas if math.isfinite(cap_meas) and cap_meas > 0 else float("nan")
                K = (r["meta"].get("kv") or {}).get("kv_pool_tokens") or c.kv_pool
                c_kv = K / workloads.get(wl).decode_weighted_context if K else float("inf")
                if ph.get("stable") and math.isfinite(bm) and bm > 0 and not c.degenerate and not ramp:
                    bh = float(c.occupancy(lam))
                    clamped = not math.isfinite(bh) or bh > cap
                    bh_plot = min(bh, cap) if math.isfinite(bh) else cap
                    kv_bound = bm >= 0.9 * c_kv
                    po = add("occupancy", hw, c, wl, bh_plot, bm, lam=lam, cap=cap, clamped=clamped,
                             kv_bound=kv_bound, hollow=(clamped or kv_bound), u=u, source="poisson_load")
                    if drawable(wl):
                        pts_occ.append(po)
                    pw = _f(ph.get("avg_power_w"))
                    if math.isfinite(pw) and c.curve is not None:
                        pr = add("power_rate", hw, c, wl, float(c.power(bh_plot)), pw, lam=lam, clamped=clamped,
                                 kv_bound=kv_bound, hollow=(clamped or kv_bound), u=u, source="poisson_load")
                        if drawable(wl):
                            pts_prate.append(pr)
                    # per request: predicted with the request's own length at the phase's mean occupancy
                    w0 = ph.get("warmup_s", 0.0)
                    w1 = w0 + ph.get("measure_s", 0.0)
                    tau = float(c.tau(bm))
                    for q in by_phase.get(int(ph["phase"]), []):
                        ts, td = _f(q["t_submit"]), _f(q["t_done"])
                        if not (w0 <= ts < w1) or not math.isfinite(td):
                            continue
                        s_i = _f(q["D"]) + _f(q["P"]) / S
                        pq = add("residence_request", hw, c, wl, s_i * tau, td - ts, lam=lam, b=bm)
                        if drawable(wl):
                            pts_req.append(pq)
                for d in ph.get("drains", []):
                    if d.get("timed_out") or d.get("n_waiting", 0) > 0 or d.get("n_prefill_pending", 0) > 0:
                        continue
                    if d.get("n_running", 0) <= 0 or _f(d.get("drain_s")) <= 0:
                        continue
                    pred = c.t0 * d["q_max"] + c.t1 * d["q_sum"]
                    pdx = add("drain", hw, c, wl, pred, d["drain_s"], n=d["n_running"], q_max=d["q_max"],
                              q_sum=d["q_sum"], lam=lam, offline=False)
                    off = offline_drain(c, int(d["n_running"]))
                    pox = add("drain", hw, c, wl, off, d["drain_s"], n=d["n_running"], lam=lam, offline=True)
                    if drawable(wl):
                        pts_drain.append(pdx); pts_drain.append(pox)

    # ---------------- kv_ceiling_sweep: KV ceiling (F5) --------------------------------------------
    pts_kv = []
    for hw, root in ROOTS:
        cells = cells_by_hw.get(hw) or {}
        for k in kvceil_points(root, cells):
            if not k["reached"]:
                print(f"  kv ceiling not reached: {hw} {k['model']} {k['wl']} (top c {k['c_req_max']:.0f}, "
                      f"best b {k['observed']:.1f}, kv {k['kv_at']:.2f})")
                continue
            p1 = add("kv_ceiling", hw, k["model"], k["wl"], k["pred_dw"], k["observed"], K=k["K"], peak=False,
                     source=k["source"])
            p2 = add("kv_ceiling", hw, k["model"], k["wl"], k["pred_peak"], k["observed"], K=k["K"], peak=True,
                     source=k["source"])
            if drawable(k["wl"]):
                pts_kv.append(p1); pts_kv.append(p2)

    # ---------------- figures -------------------------------------------------------------
    figs = {}
    tbt_core_figures(cells_by_hw, out)
    for r in tbt_law_figures(cells_by_hw, out):
        print(f"  tbt law {r['hw']:14s} {r['model']:11s} {r['n_workloads']} workloads: t0 {r['t0_min_ms']:.2f}-{r['t0_max_ms']:.2f} ms "
              f"(spread {r['t0_spread_pct']:.0f} %), t1 {r['t1_min_ms']:.3f}-{r['t1_max_ms']:.3f} ms (x{r['t1_ratio']:.1f}), "
              f"R2 {r['r2_min']:.3f}-{r['r2_max']:.3f}")
    figs["step_time"] = parity(pts_tbt, title="Step time: tau(b) = t0 + t1 b against the measured mean TBT",
                               xlabel="predicted TBT (ms)", ylabel="measured TBT (ms)",
                               path=out / "f7_step_time_parity.png")
    figs["residence_level"] = parity(pts_res, title="Mean residence: T(b) = s tau(b) against the measured mean request latency (closed loop)",
                                     xlabel="predicted residence (s)", ylabel="measured residence (s)",
                                     path=out / "f1b_residence_levels.png")
    figs["residence_request"] = parity(pts_req, title="Per-request residence under Poisson arrivals: s_i tau(b) against the measured latency",
                                       xlabel="predicted residence (s)", ylabel="measured residence (s)",
                                       path=out / "f1a_residence_requests.png", point_alpha=0.25, size=9)
    figs["occupancy"] = parity(pts_occ, title="Occupancy under Poisson arrivals: the fixed point of mu(b) = lambda against the mean running count",
                               xlabel="predicted resident requests", ylabel="measured resident requests",
                               path=out / "f2_occupancy.png", hollow_key="hollow", exclude_key="kv_bound",
                               hollow_label="fixed point above the cap (clamped to c), or KV-pinned")
    ratio_vs_utilization(pts_occ, title="Occupancy error against load: measured / predicted resident count as the offered rate approaches the measured capacity",
                         path=out / "f2b_occupancy_vs_load.png")
    figs["throughput_level"] = parity(pts_thr, title="Completion rate: mu(b) = b / T(b) against the measured throughput (closed loop)",
                                      xlabel="predicted req/s", ylabel="measured req/s",
                                      path=out / "f3a_throughput_levels.png")
    figs["capacity"] = parity(pts_cap, title="Capacity: C = c_eff / T(c_eff) against the open-loop bisection (queue-stable rate)",
                              xlabel="predicted capacity (req/s)", ylabel="measured capacity (req/s)",
                              path=out / "f3b_capacity.png")
    figs["drain"] = parity(pts_drain, title="Natural drain: t0 max(q) + t1 sum(q) with oracle residuals against the measured time to empty",
                           xlabel="predicted drain (s)", ylabel="measured drain (s)",
                           path=out / "f4_drain.png", hollow_key="offline",
                           hollow_label="offline estimate (length-biased residuals)")
    figs["kv_ceiling"] = parity(pts_kv, title="KV-limited concurrency: K / l_bar (filled) and K / (P + D) (hollow) against the observed resident plateau",
                                xlabel="predicted sequences", ylabel="observed resident sequences",
                                path=out / "f5_kv_ceiling.png", hollow_key="peak", hollow_label="peak-aware bound K / (P + D)")
    figs["power_level"] = parity(pts_pow, title="Power profile: fitted P(b) at the achieved concurrency against the measured mean power",
                                 xlabel="predicted power (W)", ylabel="measured power (W)",
                                 path=out / "f6a_power_levels.png", log=False)
    figs["power_rate"] = parity(pts_prate, title="Power from the arrival rate alone: P(b(lambda)) against the measured mean power under Poisson arrivals",
                                xlabel="predicted power (W)", ylabel="measured power (W)",
                                path=out / "f6b_power_from_rate.png", log=False, hollow_key="hollow",
                                exclude_key="kv_bound",
                                hollow_label="fixed point above the cap (clamped to c), or KV-pinned")

    # ---------------- tables ------------------------------------------------------------
    # the per-request residence points are hundreds of thousands of rows: they go to a
    # gzipped file of their own so the main points table stays small enough to commit
    import gzip
    keys = sorted({k for p in all_points for k in p})
    with open(out / "single_gpu_points.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for p in all_points:
            if p["figure"] != "residence_request":
                wr.writerow(p)
    with gzip.open(out / "single_gpu_requests.csv.gz", "wt", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for p in all_points:
            if p["figure"] == "residence_request":
                wr.writerow(p)
    rows = []
    for fig_name in ("tbt", "residence_level", "residence_request", "throughput_level", "capacity",
                     "occupancy", "drain", "kv_ceiling", "power_level", "power_rate"):
        for hw, _ in ROOTS:
            # KV-pinned probes are outside the occupancy model; KV-pinned closed-loop
            # LEVELS are legitimate operating points for the power fit (the fit used them)
            sel = [p for p in all_points if p["figure"] == fig_name and p["hw"] == hw
                   and not p.get("offline") and not p.get("peak")
                   and not (fig_name in ("occupancy", "power_rate") and p.get("kv_bound"))]
            st = stats([p["x"] for p in sel], [p["y"] for p in sel])
            if st.get("n"):
                rows.append({"figure": fig_name, "hw": hw, **st})
            for wl in WL_ORDER + TABLE_ONLY_WL:
                sw = [p for p in sel if p["wl"] == wl]
                stw = stats([p["x"] for p in sw], [p["y"] for p in sw])
                if stw.get("n"):
                    rows.append({"figure": fig_name, "hw": hw, "workload": wl, **stw})
    with open(out / "single_gpu_stats.csv", "w", newline="") as f:
        keys = ["figure", "hw", "workload", "n", "median_ratio", "p10_ratio", "p90_ratio", "mape_pct",
                "within_10pct", "within_20pct"]
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in keys})
    # Poisson windows that opened on a filling replica (RAMP_FILLED_MIN): poisson_load phases, judged on
    # their measured residence and left out; capacity_bisection probes, estimated with the model and kept
    with open(out / "single_gpu_ramp_windows.csv", "w", newline="") as f:
        keys = ["hw", "model", "workload", "run", "source", "phase", "rate_reqs_s", "residence_s",
                "residence_from", "warmup_s", "window_s", "filled", "left_out"]
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(ramp_rows)
    for src in ("poisson_load", "capacity_bisection"):
        by_hw = {}
        for x in ramp_rows:
            if x["source"] == src:
                by_hw[x["hw"]] = by_hw.get(x["hw"], 0) + 1
        print(f"  {src} windows that saw < {RAMP_FILLED_MIN:.0%} of the steady in-flight count "
              f"({'left out' if src == 'poisson_load' else 'kept, model-estimated'}): {sum(by_hw.values())} {by_hw}")

    # calibration table
    with open(out / "single_gpu_calibration.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["hw", "model", "workload", "run", "t0_ms", "t1_ms", "r2_tbt", "s", "p_idle_w",
                     "form", "heldout_rmse_w", "kv_pool_tokens", "levels_valid", "levels_total"])
        for hw, cells in cells_by_hw.items():
            for (model, wl), c in sorted(cells.items()):
                wr.writerow([hw, model, wl, c.run.name, f"{c.t0*1e3:.4f}", f"{c.t1*1e3:.5f}",
                             f"{c.r2_tbt:.4f}", f"{c.s_len:.2f}", f"{c.idle_w:.1f}",
                             c.curve.form if c.curve else "", f"{c.heldout_rmse_w:.2f}", c.kv_pool,
                             sum(l["valid"] for l in c.levels), len(c.levels)])

    # per-level table: every closed-loop level of every cell with its flags and the cell's
    # fitted values at its achieved b (the paper figures draw the TBT lines and P(b) from this)
    with open(out / "single_gpu_levels.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["hw", "model", "workload", "run", "c_req", "b", "n", "tbt_ms", "e2e_s",
                     "thr_reqs_s", "P_w", "kv_max", "waiting", "short", "kv_bound", "valid",
                     "tau_fit_ms", "T_fit_s", "mu_fit_reqs_s", "P_fit_w", "measure_s", "degenerate"])
        for hw, cells in cells_by_hw.items():
            for (model, wl), c in sorted(cells.items()):
                for l in c.levels:
                    b = l["b"]
                    wr.writerow([hw, model, wl, c.run.name, l["c_req"], f"{b:.3f}", l["n"],
                                 f"{l['tbt_ms']:.4f}", f"{l['e2e_s']:.4f}", f"{l['thr']:.4f}",
                                 f"{l['P']:.2f}", f"{l['kv_max']:.4f}", f"{l['waiting']:.3f}",
                                 l["short"], l["kv_bound"], l["valid"],
                                 f"{1e3 * float(c.tau(b)):.4f}", f"{float(c.T(b)):.4f}",
                                 f"{float(c.mu(b)):.4f}",
                                 f"{float(c.power(b)):.2f}" if c.curve is not None else "",
                                 c.measure_s, c.degenerate])
    # the fitted P(b) profile of every cell sampled on a b grid, for drawing the curves
    with open(out / "single_gpu_pb_curves.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["hw", "model", "workload", "form", "b", "P_fit_w"])
        for hw, cells in cells_by_hw.items():
            for (model, wl), c in sorted(cells.items()):
                if c.curve is None or not c.levels:
                    continue
                bmax = max(l["b"] for l in c.levels)
                # dense where a log axis looks: 0, then log-spaced from 1 to the top level
                bgrid = np.unique(np.concatenate([[0.0], np.geomspace(1.0, max(bmax, 1.0), 60)]))
                for b in bgrid:
                    wr.writerow([hw, model, wl, c.curve.form, f"{b:.3f}", f"{float(c.power(b)):.2f}"])

    # workload table
    with open(out / "workload_lengths.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["workload", "n_rows", "mean_P", "std_P", "p50_P", "p99_P", "mean_D", "std_D", "p50_D",
                     "p99_D", "corr_PD", "l_bar", "P_plus_D", "s"])
        for wl in WL_ORDER + TABLE_ONLY_WL:
            w = workloads.get(wl)
            if w.kind == "fixed":
                Pv, Dv = np.array([float(w.prefill)]), np.array([float(w.decode)])
                n = "-"
            else:
                df = workloads.load_trace(w.trace)
                Pv = np.minimum(df["num_prefill_tokens"].to_numpy(float), w.max_prefill or np.inf)
                Dv = np.minimum(df["num_decode_tokens"].to_numpy(float), w.max_decode or np.inf)
                n = len(df)
            corr = float(np.corrcoef(Pv, Dv)[0, 1]) if len(Pv) > 1 and Pv.std() > 0 and Dv.std() > 0 else float("nan")
            wr.writerow([wl, n, f"{Pv.mean():.1f}", f"{Pv.std():.1f}", f"{np.median(Pv):.0f}",
                         f"{np.percentile(Pv, 99):.0f}", f"{Dv.mean():.1f}", f"{Dv.std():.1f}",
                         f"{np.median(Dv):.0f}", f"{np.percentile(Dv, 99):.0f}", f"{corr:.2f}",
                         f"{w.decode_weighted_context:.1f}", f"{w.mean_prefill + w.mean_decode:.1f}",
                         f"{w.mean_decode + w.mean_prefill / S:.2f}"])

    emit_markdown_tables(out)
    print("\n== scores (measured / predicted)")
    for r in rows:
        if "workload" in r:
            continue
        print(f"  {r['figure']:18s} {r['hw']:14s} n {r['n']:5d}  median {r['median_ratio']:.3f}  "
              f"p10-p90 {r['p10_ratio']:.2f}-{r['p90_ratio']:.2f}  MAPE {r['mape_pct']:5.1f} %  "
              f"within 10/20 % {100*r['within_10pct']:3.0f}/{100*r['within_20pct']:3.0f}")
    print(f"\nwrote tables and figures under {out}")


if __name__ == "__main__":
    main()
