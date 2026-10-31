#!/usr/bin/env python
"""The analytical fleet model at datacenter scale: a simulation, not a measurement.

    python analyze/datacenter_sim.py
    python analyze/datacenter_sim.py --only week,shares
    python analyze/datacenter_sim.py --selfcheck

Writes CSV tables to results/tables/datacenter/, which the datacenter figures in paper_figures/ read.
No GPU. The public inputs it reads (an LLM-serving arrival trace and grid load) are fetched and
reduced to small CSVs by analyze/fetch_datacenter_inputs.py.

The model is the analytical model in lib/power_model.py, built exactly as
analyze/plot_model_vs_load.model_for builds it -- single-GPU power curves and step-time lines,
measured per-replica capacities -- with the replica count raised from the 4-8 GPUs it was
validated on to 10^4-10^5.

Why the scale-up is mostly arithmetic
-------------------------------------
Under packing a pool carrying lambda lights k = ceil(lambda / C) replicas: k-1 at the cap c,
one holding the remainder r, the rest idle at P(0). Its power is therefore

    P_pool = n P(0) + (k-1) [P(c) - P(0)] + [P(r) - P(0)]
           = n P(0) + lambda * e  +  (at most one replica's swing),   e = [P(c) - P(0)] / C,

where e is the energy one request adds above idle on a full replica. On 4 GPUs the last
term is the rung structure, steps of one replica's swing; on 10^4 it is 10^-4 of the pool.
So at scale fleet power is linear in the traffic each pool carries, and moving d-lambda of
flexible traffic from the large pool to the small one changes it by
d-lambda * (e_large - e_small). Every MW below is that product. The code still evaluates the
exact closed form, remainder replica included, not the linearization.

What carries over from the GPU runs, and what does not
-------------------------------------------------------
Carried over (each validated on 4 and 8 H100s): per-replica power at an occupancy, the
packing rule's occupancy vector, each pool's capacity, the routing window both pools can
serve, and the time a drop takes (the predicted drain + ~3 s in the measured drops).
Assumed, not validated: a router that packs across 10^4 replicas; non-GPU IT power (constant
per GPU) and cooling (a PUE) for facility-level numbers; ShareGPT request shapes (a trace
supplies only the arrival rate over time); single-GPU replicas of 1-8B models.

Scenarios (--only; all run by default): week (the reference site's week of load), shares (the
week and a 4-hour event at every flexible share), step (the measured 1-s drop and return),
availability and availability_detail (GPUs that leave the inference pools: lent to training or
failed), programs (what grid programs require against the drop time), site_70b24b_azure (the
same tables for the second site, Llama-3.1-70B -> Mistral-24B, at the Azure chat and code
request lengths).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

E = Path(__file__).resolve().parent.parent
R = E / "results"
OUT = R / "tables" / "datacenter"
INPUTS = E / "data" / "datacenter"

PAIRS = {"llama": ("llama31-8b", "llama32-1b"), "qwen": ("qwen3-8b", "qwen3-4b")}
CAP_L, CAP_S = 64, 128          # the pack caps every fleet run used (capacity_bisection capacities at these)
# Ceiling on each pool's utilization against its MEASURED capacity (for a full-model pool, the lower of the
# measured capacity and the model's own, past which the closed form is undefined). At 1.0 each pool may run at
# its full capacity, as the routing section's measured windows do: the GPU runs held an empty queue up to
# u = 0.95 and diverged at 1.05. Within a 5-minute step the busiest minute runs about 5 % above the step's mean
# (9 % at p90); a ceiling of 0.9 would keep it within u = 0.95.
U_POOL = 1.0
# Where the non-flexible (firm) queries go. "capacity": split between the two pools in the ratio of the pools'
# measured capacities, n_L C_L : n_S C_S -- a fixed assignment that no routing decision moves, so a share s of
# flexible queries leaves (1 - s) of the traffic spread over both pools as load balancing would. Only the
# flexible queries are routed: the fraction f of them goes to the small pool. "large": every firm query on the
# large model, an alternative convention kept for comparison.
FIRM_SPLIT = "capacity"
# How normal operation places the FLEXIBLE queries. "capacity": like the firm ones, in the ratio of the pools'
# measured capacities -- every query load-balanced by capacity, each pool at the site's utilization -- so every
# flexible share has the same baseline, and a power cut moves flexible queries from there to the small pool;
# where that split does not fit the window it is clipped to the nearest fraction that does. "large": as many as
# possible on the large model (f_lo), an alternative convention kept for comparison.
BASE_ROUTING = "capacity"


def capacity_share_small(fl) -> float:
    """The small pool's share of the fleet's measured capacity, n_S C_S / (n_L C_L + n_S C_S)."""
    cap_s = max(fl.n_small, 0) * fl.c_small
    total = fl.n_large * fl.c_large + cap_s
    return cap_s / total if total > 0 else 0.0


def base_fraction(fl, f_lo, f_hi) -> np.ndarray:
    """f in normal operation: the small pool's capacity share (BASE_ROUTING = "capacity") or f_lo ("large"),
    inside the window; NaN where the window is empty."""
    f_lo = np.asarray(f_lo, dtype=float)
    if BASE_ROUTING == "large":
        return f_lo
    return np.clip(np.full(f_lo.shape, capacity_share_small(fl)), f_lo, f_hi)


def firm_small_share(n_large: int, c_large: float, n_small: int, c_small: float) -> float:
    """The fraction of the firm queries the small pool carries: its share of the fleet's measured capacity
    under FIRM_SPLIT = "capacity", none under "large"."""
    cap_s = max(n_small, 0) * c_small
    if FIRM_SPLIT == "large" or cap_s <= 0:
        return 0.0
    return cap_s / (n_large * c_large + cap_s)


def routing_window(lam, s, phi: float, lim_l: float, lim_s: float) -> tuple[np.ndarray, np.ndarray]:
    """[f_lo, f_hi], the fractions of the flexible queries on the small pool that keep both pools inside their
    limits, with the firm queries split phi : 1 - phi (small : large). Small pool: lam [(1 - s) phi + s f] <=
    lim_s; large pool: lam [(1 - s)(1 - phi) + s (1 - f)] <= lim_l. NaN where no f does. The flexible share s
    may vary over time (an array as long as lam)."""
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    s = np.broadcast_to(np.asarray(s, dtype=float), lam.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = np.clip(1.0 - (lim_l / lam - (1.0 - s) * (1.0 - phi)) / s, 0.0, None)
        hi = np.clip((lim_s / lam - (1.0 - s) * phi) / s, None, 1.0)
        fits = ((lam * (1.0 - phi) <= lim_l * (1 + 1e-12))              # s = 0: nothing to route, the split must fit
                & (lam * phi <= lim_s * (1 + 1e-12)))
        f_lo = np.where(s > 0, lo, np.where(fits, 0.0, np.nan))
        f_hi = np.where(s > 0, hi, np.where(fits, 0.0, np.nan))
    # both pools exactly at their limits leave a window of zero width, which rounding can turn inside out
    touch = (f_lo > f_hi) & (f_lo - f_hi <= 1e-9)
    mid = 0.5 * (f_lo + f_hi)
    f_lo, f_hi = np.where(touch, mid, f_lo), np.where(touch, mid, f_hi)
    bad = ~(f_lo <= f_hi) | (lam <= 0)
    f_lo[bad] = np.nan
    f_hi[bad] = np.nan
    return f_lo, f_hi


# ======================================================================================
# one pool, closed form, vectorized
# ======================================================================================

def pool_closed_form(pool, lam) -> dict:
    """Power, lit replicas and requests in flight of one packed pool at offered rates ``lam``.

    The exact solve of PoolModel.power_from_rate_exact without the bisection: k-1 replicas
    at c_eff, one at r = lam_r A / (1 - lam_r B), the rest at P(0). Each replica's power is
    read at round(b), as the exact solve reads it. NaN past the model's own pool capacity.
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    A, B, c, n = pool.residence_a_s, pool.residence_b_s, float(pool.c_eff), pool.n_replicas
    out = {"power_w": np.full(lam.shape, np.nan), "lit": np.full(lam.shape, np.nan),
           "inflight": np.full(lam.shape, np.nan)}
    if n <= 0:
        out["power_w"][:] = 0.0
        out["lit"][:] = 0.0
        out["inflight"][:] = 0.0
        return out
    C1 = c / (A + B * c)
    p0 = float(pool.curve.power(0))
    pc = float(pool.curve.power(int(round(c))))
    ok = (lam >= 0) & (lam <= n * C1 * (1 + 1e-12))
    zero = ok & (lam <= 0)
    pos = ok & (lam > 0)
    k = np.clip(np.ceil(lam[pos] / C1 - 1e-12), 1, n)
    lam_r = lam[pos] - (k - 1) * C1
    r = lam_r * A / (1.0 - lam_r * B)
    pr = pool.curve.power(np.round(r))
    out["power_w"][pos] = (k - 1) * pc + pr + (n - k) * p0
    out["lit"][pos] = (k - 1) + (np.round(r) > 0)
    out["inflight"][pos] = (k - 1) * c + r
    out["power_w"][zero] = n * p0
    out["lit"][zero] = 0.0
    out["inflight"][zero] = 0.0
    return out


def per_request_energy(pool) -> dict:
    """The constants of the linear law: idle watts, full-replica watts, requests/s at the cap,
    and e = [P(c) - P(0)] / C, the energy one request adds above idle (J)."""
    A, B, c = pool.residence_a_s, pool.residence_b_s, float(pool.c_eff)
    C1 = c / (A + B * c)
    p0 = float(pool.curve.power(0))
    pc = float(pool.curve.power(int(round(c))))
    return {"p_idle_w": p0, "p_full_w": pc, "c_eff": int(c), "reqs_s_full": C1,
            "e_inc_j": (pc - p0) / C1, "e_total_j": pc / C1}


# ======================================================================================
# a two-pool fleet of any size
# ======================================================================================

@dataclass
class DCFleet:
    pair: str
    workload: str
    n_large: int
    n_small: int
    flexible_share: float
    large: object               # lib.power_model.PoolModel
    small: object
    c_large: float              # measured per-replica capacity at CAP_L, req/s (capacity_bisection)
    c_small: float
    gpus_large: int = 1         # GPUs per replica: n_large / n_small count replicas (a 70B instance is four)
    gpus_small: int = 1

    @property
    def n_gpus(self) -> int:
        return self.n_large * self.gpus_large + self.n_small * self.gpus_small

    @property
    def capacity(self) -> float:
        """Measured fleet capacity, n_L C_L + n_S C_S: what u is a fraction of (as on the GPU)."""
        return self.n_large * self.c_large + self.n_small * self.c_small

    def _limit(self, pool, c_meas: float) -> float:
        """Requests/s one pool may carry: U_POOL of the lower of its measured capacity and the
        model's own (the closed form is undefined past the latter)."""
        A, B, c = pool.residence_a_s, pool.residence_b_s, float(pool.c_eff)
        return U_POOL * pool.n_replicas * min(c_meas, c / (A + B * c))

    @property
    def firm_small_share(self) -> float:
        return firm_small_share(self.n_large, self.c_large, self.n_small, self.c_small)

    def window(self, lam) -> tuple[np.ndarray, np.ndarray]:
        """[f_lo, f_hi]: the routing fractions both pools serve inside U_POOL. NaN if none."""
        lim_l = self._limit(self.large, self.c_large)
        lim_s = self._limit(self.small, self.c_small) if self.n_small else 0.0
        return routing_window(lam, self.flexible_share, self.firm_small_share, lim_l, lim_s)

    def split(self, lam, f) -> tuple[np.ndarray, np.ndarray]:
        """(large, small) req/s: the firm queries in the FIRM_SPLIT ratio, the flexible ones by f."""
        lam = np.asarray(lam, dtype=float)
        s = self.flexible_share
        small = lam * ((1.0 - s) * self.firm_small_share + s * np.asarray(f, dtype=float))
        return lam - small, small

    def power(self, lam, f) -> dict:
        lam_l, lam_s = self.split(lam, f)
        a = pool_closed_form(self.large, lam_l)
        b = pool_closed_form(self.small, lam_s)
        return {"lam_large": np.atleast_1d(lam_l), "lam_small": np.atleast_1d(lam_s),
                "p_large_w": a["power_w"], "p_small_w": b["power_w"],
                "p_total_w": a["power_w"] + b["power_w"],
                "lit_large": a["lit"], "lit_small": b["lit"],
                "inflight_large": a["inflight"], "inflight_small": b["inflight"]}


_CACHE: dict = {}


def build_fleet(pair: str, workload: str, n_large: int, n_small: int,
                flexible_share: float = 1.0) -> DCFleet:
    """The fleet model for one (pair, workload, composition), from the GPU calibrations."""
    from analyze import capacity_lookup
    from analyze.plot_model_vs_load import model_for
    large, small = PAIRS[pair]
    key = (pair, workload, n_large, n_small)
    fm = _CACHE.get(key)
    if fm is None:
        a = {"large": large, "small": small, "workload": workload, "n_large": n_large,
             "n_small": n_small, "concurrency_large": CAP_L, "concurrency_small": CAP_S,
             "class_split": "0,0,1"}
        fm = model_for(R, a)
        if fm is None:
            raise SystemExit(f"no calibration for {pair} on {workload}")
        _CACHE[key] = fm
    cl = capacity_lookup.lookup(R, large, workload, CAP_L)
    cs = capacity_lookup.lookup(R, small, workload, CAP_S)
    if cl is None or cs is None:
        raise SystemExit(f"no measured capacity for {pair} on {workload}")
    return DCFleet(pair, workload, n_large, n_small, float(flexible_share), fm.large, fm.small,
                   float(cl["reqs_s"]), float(cs["reqs_s"]))


def composition(n_gpus: int, small_share: float) -> tuple[int, int]:
    n_small = int(round(n_gpus * small_share))
    return n_gpus - n_small, n_small


# ======================================================================================
# scenario: how fast -- the measured drop-and-return, normalized
# ======================================================================================

STEP_ARM = "modelol"           # the analytical model, open loop: the controller at scale


def step_response() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The measured shape of a drop and its return under the open-loop model, per workload.

    Each drop is normalized by its OWN measured levels, y = (P - P_pre) / (P_hold - P_pre),
    with P_pre the mean over the 20 s before the drop and P_hold the mean over t = 30-49 s of the
    50 s hold: 0 before, 1 once down, back to 0 after the return at t = 50 s. That isolates
    timing from level accuracy (1.9 % on average); normalizing by the commanded nominal
    instead would fold arXiv's -334 W pre-drop offset into the shape. At scale every replica
    the router stops feeding drains exactly as one does on 4 GPUs -- in parallel, not in
    turn -- so the fleet's response has this shape and only its height scales. The median
    over drops is what the datacenter figures draw; p10-p90 is the spread between drops.
    Drops shallower than 100 W are left out: their shape is mostly noise."""
    t = pd.read_csv(R / "power_drops/drop_traces.csv.gz")
    m = t[(t.arm == STEP_ARM) & (t.reached.astype(str).str.lower() == "true")].copy()
    key = ["run", "event"]
    pre = m[(m.t_rel >= -20) & (m.t_rel < 0)].groupby(key).measured_w.mean().rename("p_pre")
    hold = m[(m.t_rel >= 30) & (m.t_rel < 50)].groupby(key).measured_w.mean().rename("p_hold")
    m = m.join(pre, on=key).join(hold, on=key)
    m["y"] = (m.measured_w - m.p_pre) / (m.p_hold - m.p_pre)
    m = m[np.isfinite(m.y) & ((m.p_pre - m.p_hold) > 100)]
    rows = []
    for wl, g in m.groupby("workload"):
        for tr, h in g.groupby("t_rel"):
            rows.append({"workload": wl, "t_rel_s": tr, "y_p50": h.y.median(),
                         "y_p10": h.y.quantile(0.1), "y_p90": h.y.quantile(0.9),
                         "n_drops": h[["run", "event"]].drop_duplicates().shape[0]})
    shape = pd.DataFrame(rows)
    summ = []
    for wl, g in shape.groupby("workload"):
        down = g[(g.t_rel_s >= 0) & (g.t_rel_s < 50) & (g.y_p50 >= 0.9)].t_rel_s
        back = g[(g.t_rel_s >= 50) & (g.y_p50 <= 0.1)].t_rel_s
        summ.append({"workload": wl, "n_drops": int(g.n_drops.max()),
                     "t_down90_s": float(down.min()) if len(down) else np.nan,
                     "t_back90_s": float(back.min() - 50) if len(back) else np.nan})
    drains = pd.read_csv(R / "power_drops/predicted_drain.csv")
    drains = drains[drains.model == PAIRS["llama"][0]][["workload", "drain_p50_s"]]
    return shape, pd.DataFrame(summ).merge(drains, on="workload", how="left")


# ======================================================================================
# the reference datacenter, and facility power
# ======================================================================================

# One fleet every time-series scenario uses, so their numbers compose. Each choice is the one
# the GPU runs validated; the sensitivity tables vary them one at a time.
REF = dict(
    pair="llama", workload="sharegpt",
    n_gpus=10_000,          # ~1,250 eight-GPU servers; results scale linearly (x10 = 100k GPUs)
    small_share=0.25,       # the validated 3 large : 1 small composition (3L1S, 6L2S)
    flexible_share=0.5,     # half the queries labelled flexible (sensitivity 0.25 .. 1.0)
    u_peak=0.6,             # the busiest 5 minutes of the trace at 60 % of measured capacity
)
# Facility power = PUE x (GPU power + non-GPU IT power). Neither term is measured here: the
# non-GPU share (CPUs, memory, NICs, fans) is held constant per GPU, so it adds MW to the base
# but nothing to the flexible MW, and cooling is taken to follow heat, so the flexible MW at
# the meter is PUE x the GPU MW. Both values are stated assumptions, varied in
# facility_sensitivity().
PUE = 1.2
OTHER_W_PER_GPU = 300.0


def facility_mw(gpu_w, n_gpus: int, pue: float = PUE, other_w: float = OTHER_W_PER_GPU):
    return pue * (np.asarray(gpu_w, dtype=float) + n_gpus * other_w) / 1e6


def ref_fleet(**kw) -> DCFleet:
    p = {**REF, **kw}
    nl, ns = composition(p["n_gpus"], p["small_share"])
    return build_fleet(p["pair"], p["workload"], nl, ns, p["flexible_share"])


def load_profile(name: str = "azure_conv_2024_week") -> pd.DataFrame:  # noqa: D401
    """A normalized arrival-rate profile: columns t (UTC timestamps) and shape (peak = 1).

    Written by analyze/fetch_datacenter_inputs.py from a public serving trace; only the
    rate's variation over time is used -- request shapes stay the calibrated workload's."""
    p = INPUTS / f"{name}.csv"
    if not p.exists():
        raise SystemExit(f"{p} missing: run analyze/fetch_datacenter_inputs.py first")
    d = pd.read_csv(p, parse_dates=["t"])
    return d[["t", "shape"]]


def envelope(fl: DCFleet, lam) -> dict:
    """At each rate: the routing window, normal operation (f_base, BASE_ROUTING: every query split by the pools'
    capacities) and the floor (f_hi: every flexible query the small pool can take); p_ceiling_w is the most power
    the window allows (f_lo: as much as possible on the large model)."""
    f_lo, f_hi = fl.window(lam)
    f_base = base_fraction(fl, f_lo, f_hi)
    hi = fl.power(lam, np.nan_to_num(f_base))
    lo = fl.power(lam, np.nan_to_num(f_hi))
    top = fl.power(lam, np.nan_to_num(f_lo))
    ok = np.isfinite(f_lo)
    nan = lambda x: np.where(ok, x, np.nan)
    return {"f_lo": f_lo, "f_hi": f_hi, "f_base": f_base, "p_base_w": nan(hi["p_total_w"]),
            "p_floor_w": nan(lo["p_total_w"]), "p_ceiling_w": nan(top["p_total_w"]),
            "lit_large_base": nan(hi["lit_large"]), "lit_large_floor": nan(lo["lit_large"]),
            "util_small_floor": nan(lo["lam_small"] / max(fl.n_small * fl.c_small, 1e-9)),
            "util_large_base": nan(hi["lam_large"] / max(fl.n_large * fl.c_large, 1e-9))}


# ======================================================================================
# scenario: a week of real traffic
# ======================================================================================

def week_table(profile: pd.DataFrame, fleet: DCFleet | None = None, **kw) -> pd.DataFrame:
    """The reference datacenter over a traced week: baseline power, the floor routing can
    reach, and the flexible MW between them, at the trace's resolution. `fleet` replaces the
    reference fleet (the traffic stays u_peak x that fleet's measured capacity x the profile)."""
    p = {**REF, **kw}
    fl = fleet if fleet is not None else ref_fleet(**kw)
    lam = p["u_peak"] * fl.capacity * profile["shape"].to_numpy()
    env = envelope(fl, lam)
    out = pd.DataFrame({"t": profile["t"], "u": lam / fl.capacity, "lam_reqs_s": lam,
                        "f_lo": env["f_lo"], "f_hi": env["f_hi"], "f_base": env["f_base"],
                        "gpu_base_mw": env["p_base_w"] / 1e6, "gpu_floor_mw": env["p_floor_w"] / 1e6,
                        "lit_large_base": env["lit_large_base"], "lit_large_floor": env["lit_large_floor"],
                        "util_small_floor": env["util_small_floor"],
                        "util_large_base": env["util_large_base"]})
    out["gpu_flex_mw"] = out.gpu_base_mw - out.gpu_floor_mw
    out["fac_base_mw"] = facility_mw(out.gpu_base_mw * 1e6, fl.n_gpus)
    out["fac_floor_mw"] = facility_mw(out.gpu_floor_mw * 1e6, fl.n_gpus)
    out["fac_flex_mw"] = out.fac_base_mw - out.fac_floor_mw
    out["flex_frac_fac"] = out.fac_flex_mw / out.fac_base_mw
    meta = {"pair": fl.pair, "workload": fl.workload, "n_gpus": fl.n_gpus,
            "small_share": fl.n_small * fl.gpus_small / fl.n_gpus, "flexible_share": fl.flexible_share,
            "u_peak": p["u_peak"]}
    for k, v in meta.items():
        out[k] = v
    return out


def week_sensitivity(profile: pd.DataFrame) -> pd.DataFrame:
    """One REF parameter at a time: what the week's flexible MW depends on."""
    grid = [("reference", {})]
    grid += [(f"flexible_share={s}", {"flexible_share": s}) for s in (0.25, 0.75, 1.0)]
    grid += [(f"u_peak={u}", {"u_peak": u}) for u in (0.4, 0.8)]
    grid += [(f"small_share={x}", {"small_share": x}) for x in (0.125, 0.375, 0.5)]
    grid += [("pair=qwen", {"pair": "qwen"})]
    grid += [(f"workload={w}", {"workload": w}) for w in ("arxiv_summarization", "fixed_1024_1024")]
    grid += [("trace=azure code (weekday business hours)", {"_profile": "azure_code_2024_week"})]
    rows = []
    for label, kw in grid:
        prof = load_profile(kw.pop("_profile")) if "_profile" in kw else profile
        w = week_table(prof, **kw)
        ok = w.gpu_flex_mw.notna()
        rows.append({"case": label, "feasible_frac": float(ok.mean()),
                     "gpu_base_mean_mw": w.gpu_base_mw.mean(), "gpu_flex_mean_mw": w.gpu_flex_mw.mean(),
                     "gpu_flex_min_mw": w.gpu_flex_mw.min(), "gpu_flex_max_mw": w.gpu_flex_mw.max(),
                     "fac_base_mean_mw": w.fac_base_mw.mean(), "fac_flex_mean_mw": w.fac_flex_mw.mean(),
                     "fac_flex_min_mw": w.fac_flex_mw.min(),
                     "flex_frac_fac_mean": w.flex_frac_fac.mean(), "flex_frac_fac_min": w.flex_frac_fac.min(),
                     "gpu_flex_frac_mean": (w.gpu_flex_mw / w.gpu_base_mw).mean()})
    return pd.DataFrame(rows)


def facility_sensitivity(profile: pd.DataFrame) -> pd.DataFrame:
    """The facility-level flexible share under other PUE / non-GPU assumptions (GPU MW fixed)."""
    w = week_table(profile)
    rows = []
    for pue in (1.1, 1.2, 1.4):
        for other in (0.0, 150.0, 300.0, 575.0):
            base = facility_mw(w.gpu_base_mw * 1e6, REF["n_gpus"], pue, other)
            flex = pue * w.gpu_flex_mw
            rows.append({"pue": pue, "other_w_per_gpu": other, "fac_base_mean_mw": base.mean(),
                         "fac_flex_mean_mw": flex.mean(), "flex_frac_mean": (flex / base).mean()})
    return pd.DataFrame(rows)


# ======================================================================================
# scenario: a demand-response event on the week's busiest day
# ======================================================================================

EVENT_HOURS_LOCAL = (16, 20)    # a 4-hour summer-peak window, US Eastern (EEA / load-management hours)
EVENT_TZ = "America/New_York"


def busiest_day(local: pd.Series, shape):
    """The local calendar day with the highest mean load among the days the profile covers in full: a UTC week
    seen in US Eastern starts and ends with part-days, and a part-day missing its quiet evening reads busier."""
    s = pd.Series(np.asarray(shape, dtype=float), index=local.dt.date.to_numpy())
    n = s.groupby(level=0).size()
    return s[s.index.isin(n.index[n == n.max()])].groupby(level=0).mean().idxmax()


def event_table(profile: pd.DataFrame, depth: str | float = "max", fleet: DCFleet | None = None,
                **kw) -> tuple[pd.DataFrame, dict]:
    """The busiest day of the week with a 4-hour event at the local evening peak.

    Outside the event the fleet runs in normal operation (f_base, BASE_ROUTING). Inside it either drops to the
    floor (depth="max") or holds a cut of `depth` x baseline facility power, set by inverting
    the model -- at scale P(f) is linear, so the inverse is a straight line, and the cut is
    clipped to the floor when the window cannot reach it. `fleet` replaces the reference fleet."""
    p = {**REF, **kw}
    fl = fleet if fleet is not None else ref_fleet(**kw)
    d = profile.copy()
    d["local"] = d.t.dt.tz_convert(EVENT_TZ) if d.t.dt.tz is not None else d.t.dt.tz_localize("UTC").dt.tz_convert(EVENT_TZ)
    day = busiest_day(d.local, d["shape"])
    d = d[d.local.dt.date == day].reset_index(drop=True)
    lam = p["u_peak"] * fl.capacity * d["shape"].to_numpy()
    env = envelope(fl, lam)
    hr = d.local.dt.hour + d.local.dt.minute / 60
    in_ev = (hr >= EVENT_HOURS_LOCAL[0]) & (hr < EVENT_HOURS_LOCAL[1])
    base_fac = facility_mw(env["p_base_w"], fl.n_gpus)
    floor_fac = facility_mw(env["p_floor_w"], fl.n_gpus)
    if depth == "max":
        f_ev = env["f_hi"]
    else:
        want_fac = base_fac * (1 - float(depth))
        # GPU watts to remove, then f along the (linear) P(f) between normal operation and f_hi
        cut_gpu = (base_fac - want_fac) * 1e6 / PUE
        span = env["p_base_w"] - env["p_floor_w"]
        frac = np.clip(cut_gpu / np.where(span > 0, span, np.nan), 0, 1)
        f_ev = env["f_base"] + frac * (env["f_hi"] - env["f_base"])
    f = np.where(in_ev, f_ev, env["f_base"])
    q = fl.power(lam, f)
    out = pd.DataFrame({"t": d.t, "local": d.local, "hour_local": hr, "in_event": in_ev,
                        "u": lam / fl.capacity, "f": f, "f_lo": env["f_lo"], "f_hi": env["f_hi"],
                        "f_base": env["f_base"],
                        "gpu_mw": q["p_total_w"] / 1e6, "gpu_base_mw": env["p_base_w"] / 1e6,
                        "gpu_floor_mw": env["p_floor_w"] / 1e6,
                        "fac_mw": facility_mw(q["p_total_w"], fl.n_gpus), "fac_base_mw": base_fac,
                        "fac_floor_mw": floor_fac,
                        "small_frac_of_queries": q["lam_small"] / lam,
                        "lit_large": q["lit_large"], "lit_small": q["lit_small"]})
    ev = out[out.in_event]
    dt_h = (out.t.diff().dt.total_seconds().median() or 300) / 3600
    base_q = fl.split(lam, env["f_base"])[1]                    # queries on the small model anyway
    moved = lam * fl.flexible_share * (f - env["f_base"])
    summ = {"day": str(day), "depth": depth, "event_hours_local": EVENT_HOURS_LOCAL,
            "fac_base_event_mean_mw": float(ev.fac_base_mw.mean()),
            "fac_event_mean_mw": float(ev.fac_mw.mean()),
            "fac_shed_mean_mw": float((ev.fac_base_mw - ev.fac_mw).mean()),
            "fac_shed_min_mw": float((ev.fac_base_mw - ev.fac_mw).min()),
            "shed_frac_mean": float(((ev.fac_base_mw - ev.fac_mw) / ev.fac_base_mw).mean()),
            "shed_mwh": float((ev.fac_base_mw - ev.fac_mw).sum() * dt_h),
            "queries_moved_frac_of_day": float(moved.sum() / lam.sum()),
            "queries_moved_frac_of_event": float(moved[in_ev].sum() / lam[in_ev].sum()),
            "queries_on_small_anyway_frac_of_day": float(base_q.sum() / lam.sum()),
            "u_event_mean": float(ev.u.mean()), "u_day_max": float(out.u.max())}
    return out, summ


# ======================================================================================
# scenario: curtailment-enabled headroom (Norris et al., Duke 2025) and what routing covers
# ======================================================================================

CURTAIL_RATES = (0.0025, 0.005, 0.01)


def load_grid(ba: str, year: int) -> pd.Series:
    """Hourly balancing-authority demand (MW), UTC index, from EIA-930 (fetch_datacenter_inputs)."""
    p = INPUTS / f"eia930_{ba}_{year}.csv.gz"
    if not p.exists():
        raise SystemExit(f"{p} missing: run analyze/fetch_datacenter_inputs.py first")
    d = pd.read_csv(p, parse_dates=["t"])
    s = d.set_index("t")["demand_mw"].astype(float)
    return s[~s.index.duplicated()].sort_index()


def curtailment(load: pd.Series, x_mw: float, ceiling: float, dc_shape=None) -> pd.Series:
    """MW the new load must give up each hour to keep the total at or below `ceiling`.

    `dc_shape` is the new load's own hourly level as a fraction of its nameplate x_mw (1.0 =
    flat, the Duke assumption)."""
    lvl = x_mw * (1.0 if dc_shape is None else dc_shape)
    return (load + lvl - ceiling).clip(lower=0).clip(upper=lvl)


def headroom(load: pd.Series, rate: float, ceiling: float | None = None, dc_shape=None) -> dict:
    """Largest new load whose curtailed energy is `rate` of what it would otherwise use.

    The method of Norris et al. (Duke Nicholas Institute, 2025) on one year of hourly demand,
    with the year's own peak as the ceiling unless one is given."""
    ceiling = float(load.max()) if ceiling is None else ceiling
    use = lambda x: float((x * (1.0 if dc_shape is None else dc_shape) * np.ones(len(load))).sum())
    lo, hi = 0.0, float(load.max())
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        frac = curtailment(load, mid, ceiling, dc_shape).sum() / use(mid)
        lo, hi = (mid, hi) if frac <= rate else (lo, mid)
    x = lo
    c = curtailment(load, x, ceiling, dc_shape)
    on = c > 0
    runs = (on != on.shift()).cumsum()[on]
    dur = runs.value_counts()
    depth = c[on] / (x * (1.0 if dc_shape is None else np.asarray(dc_shape)[on.to_numpy()]))
    return {"rate": rate, "headroom_mw": x, "ceiling_mw": ceiling, "hours": int(on.sum()),
            "events": int(len(dur)), "mean_event_h": float(dur.mean()) if len(dur) else 0.0,
            "max_event_h": int(dur.max()) if len(dur) else 0,
            "depth_mean": float(depth.mean()) if len(depth) else 0.0,
            "depth_p50": float(depth.median()) if len(depth) else 0.0,
            "depth_p90": float(depth.quantile(0.9)) if len(depth) else 0.0,
            "hours_depth_le_25pct": int((depth <= 0.25).sum()),
            "hours_depth_le_50pct": int((depth <= 0.5).sum())}


def dc_hourly_shape(week: pd.DataFrame, index: pd.DatetimeIndex, tz: str) -> tuple[np.ndarray, np.ndarray]:
    """The reference datacenter's facility load and flexible MW, each as a fraction of its
    weekly peak, mapped onto a grid year's hours by (local weekday, local hour)."""
    w = week.copy()
    loc = w.t.dt.tz_convert(tz)
    w["wd"], w["hr"] = loc.dt.weekday, loc.dt.hour
    peak = w.fac_base_mw.max()
    g = w.groupby(["wd", "hr"]).agg(load=("fac_base_mw", "mean"), flex=("fac_flex_mw", "mean")) / peak
    li = index.tz_convert(tz)
    key = pd.MultiIndex.from_arrays([li.weekday, li.hour])
    return g.load.reindex(key).to_numpy(), g.flex.reindex(key).to_numpy()


GRID_TZ = {"PJM": "America/New_York", "ERCO": "America/Chicago"}


def curtail_tables(week: pd.DataFrame, years=(2024, 2025)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per balancing authority and year: Duke's headroom for a flat load (the reproduction), and
    for an inference load shaped like the reference datacenter, with the share of the required
    curtailment the routing lever alone delivers (hours, and MWh) at each rate."""
    rows, hourly = [], []
    for ba, tz in GRID_TZ.items():
        for yr in years:
            try:
                load = load_grid(ba, yr)
            except SystemExit:
                continue
            shape, flex = dc_hourly_shape(week, load.index, tz)
            for rate in CURTAIL_RATES:
                flat = headroom(load, rate)
                dc = headroom(load, rate, dc_shape=shape)
                x = dc["headroom_mw"]
                need = curtailment(load, x, dc["ceiling_mw"], shape)
                can = x * flex
                on = need > 0
                covered_h = int(((need <= can + 1e-9) & on).sum())
                rows.append({"ba": ba, "year": yr, "rate": rate,
                             "flat_headroom_mw": flat["headroom_mw"], "flat_hours": flat["hours"],
                             "flat_events": flat["events"], "flat_mean_event_h": flat["mean_event_h"],
                             "flat_max_event_h": flat["max_event_h"],
                             "flat_depth_mean": flat["depth_mean"], "flat_depth_p90": flat["depth_p90"],
                             "flat_hours_half_online_frac": flat["hours_depth_le_50pct"] / max(flat["hours"], 1),
                             "dc_hours_depth_le_25pct_frac": dc["hours_depth_le_25pct"] / max(dc["hours"], 1),
                             "dc_hours_half_online_frac": dc["hours_depth_le_50pct"] / max(dc["hours"], 1),
                             "dc_headroom_mw": x, "dc_hours": dc["hours"], "dc_events": dc["events"],
                             "dc_mean_event_h": dc["mean_event_h"], "dc_depth_mean": dc["depth_mean"],
                             "dc_depth_p50": dc["depth_p50"], "dc_depth_p90": dc["depth_p90"],
                             "routing_covers_hours_frac": covered_h / max(int(on.sum()), 1),
                             "routing_covers_mwh_frac": float(np.minimum(need, can)[on].sum() / max(need.sum(), 1e-9)),
                             "residual_mwh": float((need - np.minimum(need, can)).clip(lower=0).sum()),
                             "residual_hours": int(((need - can) > 1e-6).sum()),
                             "peak_load_mw": float(load.max()), "mean_load_mw": float(load.mean())})
                if rate == 0.005:
                    hh = pd.DataFrame({"t": load.index, "ba": ba, "year": yr, "load_mw": load.to_numpy(),
                                       "dc_mw": x * shape, "need_mw": need.to_numpy(), "routing_mw": can})
                    hourly.append(hh[hh.need_mw > 0])
    return pd.DataFrame(rows), (pd.concat(hourly, ignore_index=True) if hourly else pd.DataFrame())


# ======================================================================================
# scenario: how fast against what grid rules require -- and the ramp limits that bind at scale
# ======================================================================================

SITE_GPUS = (10_000, 100_000, 1_000_000)


def programs_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The curated program requirements (data/datacenter/grid_programs.csv, sources inside) beside
    the measured routing response, and the time a full cut takes under each large-load ramp rule.

    A cut of X MW completes no sooner than the drain allows (the measured 90 % time in step_summary.csv) and
    no sooner than the ramp rule allows: X / rate for a MW-per-minute cap, 1 / rate of the range for
    ERCOT's PCLR (10-20 % of the flexible range per minute, whatever its size). Site sizes scale
    the reference week's mean flexible MW linearly (half / all queries flexible)."""
    pr = pd.read_csv(INPUTS / "grid_programs.csv")
    step = pd.read_csv(OUT / "step_summary.csv")
    lo, hi = float(step.t_down90_s.min()), float(step.t_down90_s.max())
    pr["routing_down90_s_min"], pr["routing_down90_s_max"] = lo, hi
    pr["routing_meets_response"] = pr.response_s >= hi
    w5 = pd.read_csv(OUT / "week.csv")
    w10 = pd.read_csv(OUT / "week_flex100.csv")
    rows = []
    sharegpt = float(step.set_index("workload").loc["sharegpt", "t_down90_s"])
    for n in SITE_GPUS:
        for s, w in ((0.5, w5), (1.0, w10)):
            x = float(w.fac_flex_mw.mean()) * n / REF["n_gpus"]
            xmax = float(w.fac_flex_mw.max()) * n / REF["n_gpus"]
            rows.append({"n_gpus": n, "flexible_share": s, "flex_mean_mw": x, "flex_max_mw": xmax,
                         "drain_limited_s": sharegpt,
                         "spp_20mw_min_s": max(sharegpt, xmax / 20.0 * 60),
                         "miso_30mw_min_s": max(sharegpt, xmax / 30.0 * 60),
                         "ercot_pclr_fastest_s": max(sharegpt, 60 / 0.20),
                         "ercot_pclr_slowest_s": max(sharegpt, 60 / 0.10),
                         "ercot_10mw_per_5s_repeated_s": max(sharegpt, xmax / 10.0 * 5)})
    return pr, pd.DataFrame(rows)


# ======================================================================================
# GPUs that leave -- lent to training, failed, in maintenance
# ======================================================================================

AVAIL_FRACTIONS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5)
LEND_FRACTION = 0.4                 # GPUs lent to training outside the weekday 08:00-20:00 Eastern window


def envelope_fixed_traffic(fl: DCFleet, lam: np.ndarray) -> pd.DataFrame:
    """The routing envelope of a given fleet at a given absolute rate -- the traffic does not shrink
    when GPUs leave, so u rises."""
    env = envelope(fl, lam)
    base = facility_mw(env["p_base_w"], fl.n_gpus)
    floor = facility_mw(env["p_floor_w"], fl.n_gpus)
    return pd.DataFrame({"u": lam / fl.capacity, "fac_base_mw": base, "fac_floor_mw": floor,
                         "fac_flex_mw": base - floor, "f_lo": env["f_lo"], "f_hi": env["f_hi"],
                         "f_base": env["f_base"]})


def availability_tables(profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(a) The reference week when only a fraction a of the inference GPUs is available (both
    pools alike: lent to training, failed or in maintenance) and the traffic is unchanged.
    (b) A week in which 40 % of the GPUs serve training outside weekday 08:00-20:00 Eastern."""
    rows = []
    out_b = []
    for s in (REF["flexible_share"], 1.0):
        full = ref_fleet(flexible_share=s)
        lam = REF["u_peak"] * full.capacity * profile["shape"].to_numpy()
        for a in AVAIL_FRACTIONS:
            fl = build_fleet(REF["pair"], REF["workload"], int(round(a * full.n_large)),
                             int(round(a * full.n_small)), s)
            e = envelope_fixed_traffic(fl, lam)
            ok = e.fac_flex_mw.notna()
            rows.append({"flexible_share": s, "available_frac": a, "n_gpus": fl.n_gpus,
                         "served_frac_of_week": float(ok.mean()),
                         "u_peak_on_available": float(np.nanmax(e.u)),
                         "fac_flex_mean_mw": float(e.fac_flex_mw[ok].mean()) if ok.any() else np.nan,
                         "fac_flex_min_mw": float(e.fac_flex_mw[ok].min()) if ok.any() else np.nan,
                         "fac_base_mean_mw": float(e.fac_base_mw[ok].mean()) if ok.any() else np.nan})
        # (b) lending outside the weekday daytime window
        lent = build_fleet(REF["pair"], REF["workload"], int(round((1 - LEND_FRACTION) * full.n_large)),
                           int(round((1 - LEND_FRACTION) * full.n_small)), s)
        loc = profile.t.dt.tz_convert(EVENT_TZ)
        day = (loc.dt.weekday < 5) & (loc.dt.hour >= 8) & (loc.dt.hour < 20)
        e_full = envelope_fixed_traffic(full, lam)
        e_lent = envelope_fixed_traffic(lent, lam)
        pick = lambda col: np.where(day.to_numpy(), e_full[col], e_lent[col])
        out_b.append(pd.DataFrame({"t": profile.t, "flexible_share": s, "inference_gpus": np.where(day, full.n_gpus, lent.n_gpus),
                                   "u": pick("u"), "fac_base_mw": pick("fac_base_mw"),
                                   "fac_flex_mw": pick("fac_flex_mw"),
                                   "fac_flex_mw_always_full": e_full.fac_flex_mw}))
    return pd.DataFrame(rows), pd.concat(out_b, ignore_index=True)


AVAIL_GRID = tuple(round(1.0 - 0.025 * k, 3) for k in range(21))     # 100 % .. 50 % of the GPUs, smooth curves


def availability_detail_tables(profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GPUs that leave (availability_tables) at every flexible share, on a fine grid of GPUs available (both
    pools alike, traffic unchanged).
    (a) The week's room on the reference site: its mean (an interval the remaining GPUs cannot serve counts
    as no room, as in the share scenarios), its firm part (available 95 % of the week), and how much of the
    week is still served. (b) The routing window [f_lo, f_hi] at the week's busiest, median and quietest
    5 minutes: the share of the flexible queries the small model must take to keep today's service, and
    the most it can take. The router can move the queries between the two; the room is that share of the
    flexible traffic times the energy each moved query saves."""
    shape = profile["shape"].to_numpy()
    moments = {"busiest": float(np.max(shape)), "median": float(np.median(shape)),
               "quietest": float(np.min(shape))}
    rows, win = [], []
    for s in FLEX_SHARES:
        full = ref_fleet(flexible_share=s)
        lam = REF["u_peak"] * full.capacity * shape
        mean_full = None
        for a in AVAIL_GRID:
            fl = build_fleet(REF["pair"], REF["workload"], int(round(a * full.n_large)),
                             int(round(a * full.n_small)), s)
            e = envelope_fixed_traffic(fl, lam)
            ok = e.fac_flex_mw.notna()
            f = e.fac_flex_mw.fillna(0.0)
            mean = float(f.mean())
            mean_full = mean if mean_full is None else mean_full
            rows.append({"flexible_share": s, "available_frac": a, "n_gpus": fl.n_gpus,
                         "served_frac_of_week": float(ok.mean()),
                         "u_peak_on_available": float(lam.max() / fl.capacity),
                         "fac_flex_mean_mw": mean, "fac_flex_p5_mw": float(f.quantile(0.05)),
                         "fac_flex_kept_of_full": mean / mean_full if mean_full else np.nan,
                         "fac_base_mean_mw": float(e.fac_base_mw[ok].mean()) if ok.any() else np.nan})
            for name, sh in moments.items():
                lam_m = REF["u_peak"] * full.capacity * sh
                lo, hi = fl.window(lam_m)
                win.append({"flexible_share": s, "available_frac": a, "moment": name, "lam_reqs_s": lam_m,
                            "u_on_available": lam_m / fl.capacity, "f_lo": float(lo[0]), "f_hi": float(hi[0])})
    return pd.DataFrame(rows), pd.DataFrame(win)


REMOVAL_RULES = ("both in proportion", "8B pool only", "1B pool only")


def removal_by_pool_table(profile: pd.DataFrame) -> pd.DataFrame:
    """Which GPUs leave: a share r of the site's GPUs removed from both pools in proportion (the rule of
    availability_tables), from the 8B pool only, or from the 1B pool only -- the reference week, traffic
    unchanged, unserved steps counted as no reduction. Which pool is cheaper depends on the flexible share: at
    half flexible the 1B pool has spare capacity, at all flexible it is the pool the moved queries fill."""
    rows = []
    for s in (REF["flexible_share"], 1.0):
        full = ref_fleet(flexible_share=s)
        lam = REF["u_peak"] * full.capacity * profile["shape"].to_numpy()
        nl, ns = full.n_large, full.n_small
        for r in (round(0.05 * k, 2) for k in range(11)):
            k = int(round(r * (nl + ns)))
            shapes = {"both in proportion": (nl - int(round(r * nl)), ns - int(round(r * ns))),
                      "8B pool only": (nl - k, ns), "1B pool only": (nl, ns - k)}
            for rule in REMOVAL_RULES:
                l, sm = shapes[rule]
                if l <= 0 or sm <= 0:
                    continue
                e = envelope_fixed_traffic(build_fleet(REF["pair"], REF["workload"], l, sm, s), lam)
                f = e.fac_flex_mw.fillna(0.0)
                rows.append({"flexible_share": s, "removed_frac": r, "rule": rule, "n_large": l, "n_small": sm,
                             "served_frac_of_week": float(e.fac_flex_mw.notna().mean()),
                             "fac_flex_mean_mw": float(f.mean()), "fac_flex_p5_mw": float(f.quantile(0.05))})
    return pd.DataFrame(rows)


# ======================================================================================
# every flexible share at once: what the main datacenter figures draw as a gradient
# ======================================================================================

FLEX_SHARES = tuple(round(0.1 * k, 1) for k in range(1, 11))     # 10 % .. 100 % of queries flexible


def shares_tables(profile: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The traced week, the 4-hour event and the curtailment coverage at every flexible share
    10-100 %, each the reference scenario with only `flexible_share` changed -- so the figures
    can show how each result grows with the share instead of two cases of it."""
    weeks, events, ev_summ, curt, curt_h = [], [], [], [], []
    # Today's power does not depend on the share wherever the site can serve its traffic, so the
    # new load the grid sees is the same at every share (the all-flexible week, feasible
    # throughout); only what routing can shed changes, and an interval a low share cannot serve
    # sheds nothing. (Averaging over the feasible intervals only would flatter the low shares.)
    today = week_table(profile, flexible_share=1.0).fac_base_mw.to_numpy()
    for s in FLEX_SHARES:
        w = week_table(profile, flexible_share=s)
        weeks.append(w[["t", "u", "fac_base_mw", "fac_floor_mw", "fac_flex_mw", "flex_frac_fac",
                        "util_small_floor"]].assign(flexible_share=s))
        tab, summ = event_table(profile, "max", flexible_share=s)
        events.append(tab[["t", "local", "hour_local", "in_event", "fac_mw", "fac_base_mw",
                           "fac_floor_mw"]].assign(flexible_share=s))
        ev_summ.append({"flexible_share": s, **summ})
        tab_c, hourly = curtail_tables(w.assign(fac_base_mw=today, fac_flex_mw=w.fac_flex_mw.fillna(0.0)))
        curt.append(tab_c.assign(flexible_share=s))
        if len(hourly):
            curt_h.append(hourly[(hourly.ba == "PJM") & (hourly.year == 2025)].assign(flexible_share=s))
    return {"week_by_share": pd.concat(weeks, ignore_index=True),
            "event_by_share": pd.concat(events, ignore_index=True),
            "event_summary_by_share": pd.DataFrame(ev_summ),
            "curtailment_by_share": pd.concat(curt, ignore_index=True),
            "curtailment_hours_by_share": (pd.concat(curt_h, ignore_index=True) if curt_h else pd.DataFrame())}


# ======================================================================================
# other model pairs: the second site (Llama-3.1-70B -> Mistral-24B) at the Azure request lengths
# ======================================================================================
#
# WHERE THE SAVING COMES FROM. A busy H100 draws 320-600 W whatever model it serves (at the cap
# on ShareGPT: 1B 324 W, 8B 536 W, 24B 600 W, 70B 518 W per card), but it completes 69, 22, 10
# and 2.5 requests/s. Energy per request is busy power over THROUGHPUT, so routing a flexible
# query to a smaller model saves energy because the smaller model needs far fewer GPU-seconds
# for it -- the large pool darkens whole replicas -- not because the smaller model's GPU draws
# less. A pair's throughput ratio per GPU, not its power ratio, sets the room.
#
# The pools here are built from the same single-instance calibrations as everything above
# (plot_single_gpu_model.build_cells: the fitted P(b), the step-time line, the KV pool), on the
# H100 root or, for the 70B served as one instance over four H100s, its own root. A capacity
# comes from capacity_bisection where it was measured at the cap and is the pool's own c / T(c) otherwise
# (recorded per tier, so a figure can say which).

GPUS_PER_REPLICA = {"llama31-70b": 4}
CAL_ROOTS = {"llama31-70b": ("h100-80gb-tp4", "h100-80gb-tp4")}    # (hardware slug, results subdir)
SMALL_CAP_MODELS = ("llama32-1b", "llama32-3b", "qwen3-1.7b", "qwen3-4b")   # cap 128, as the fleet runs
MODEL_LABEL = {"llama32-1b": "Llama-3.2-1B", "llama32-3b": "Llama-3.2-3B", "llama31-8b": "Llama-3.1-8B",
               "mistral-7b": "Mistral-7B", "mistral-24b": "Mistral-24B", "qwen3-4b": "Qwen3-4B",
               "qwen3-8b": "Qwen3-8B", "qwen3-14b": "Qwen3-14B", "llama31-70b": "Llama-3.1-70B (4 GPUs)"}
_CELLS: dict = {}


def tier_cap(model: str) -> int:
    return 128 if model in SMALL_CAP_MODELS else 64


@dataclass
class Tier:
    """One model's pool: the calibrated PoolModel, GPUs per replica and the capacity its limit uses."""
    model: str
    workload: str
    pool: object
    gpus_per_replica: int
    c_meas: float                # per-replica req/s the limit uses (measured where it exists)
    capacity_source: str
    t0_s: float
    t1_s: float
    label: str = ""

    @property
    def n_replicas(self) -> int:
        return int(self.pool.n_replicas)

    @property
    def n_gpus(self) -> int:
        return self.n_replicas * self.gpus_per_replica

    @property
    def own_capacity(self) -> float:
        c = float(self.pool.c_eff)
        return c / (self.pool.residence_a_s + self.pool.residence_b_s * c)

    @property
    def limit(self) -> float:
        return U_POOL * self.n_replicas * min(self.c_meas, self.own_capacity)

    def with_replicas(self, n: int) -> "Tier":
        import dataclasses
        return dataclasses.replace(self, pool=dataclasses.replace(self.pool, n_replicas=int(max(n, 0))))

    def constants(self) -> dict:
        e = per_request_energy(self.pool)
        g = self.gpus_per_replica
        return {"model": self.model, "label": self.label or MODEL_LABEL.get(self.model, self.model),
                "gpus_per_replica": g, "pack_cap": int(self.pool.pack_cap), "c_eff": int(self.pool.c_eff),
                "reqs_s_per_replica": min(self.c_meas, self.own_capacity), "capacity_source": self.capacity_source,
                "reqs_s_per_gpu": min(self.c_meas, self.own_capacity) / g,
                "p_idle_w_per_gpu": e["p_idle_w"] / g, "p_full_w_per_gpu": e["p_full_w"] / g,
                "e_inc_j": e["e_inc_j"], "e_total_j": e["e_total_j"], "t0_ms": 1e3 * self.t0_s, "t1_ms": 1e3 * self.t1_s}


def make_tier(model: str, workload: str, n_replicas: int) -> Tier:
    """A pool of `n_replicas` instances of `model` on `workload`, from its own calibration."""
    import contextlib
    import io
    from analyze import capacity_lookup
    from analyze.plot_single_gpu_model import build_cells
    from analyze.score_fleet_model import pool_from
    from lib import workloads
    hw, sub = CAL_ROOTS.get(model, ("h100-80gb", ""))
    root = R / sub if sub else R
    if hw not in _CELLS:
        with contextlib.redirect_stdout(io.StringIO()):          # build_cells narrates every dropped level
            _CELLS[hw] = build_cells(hw, root)
    c = _CELLS[hw].get((model, workload))
    if c is None or c.curve is None:
        raise SystemExit(f"no single-instance calibration for {model} on {workload} ({hw})")
    cap = tier_cap(model)
    pool = pool_from(c.curve, {"t0_s": c.t0, "t1_s": c.t1}, workloads.get(workload), int(n_replicas), cap, root, model)
    meas = capacity_lookup.lookup(root, model, workload, cap)
    own = float(pool.c_eff) / (pool.residence_a_s + pool.residence_b_s * float(pool.c_eff))
    if not meas:
        source = "model c/T(c) (no capacity_bisection at this cap yet)"
    elif own < float(meas["reqs_s"]):                    # the limit takes the lower of the two (Tier.limit)
        source = f"measured (capacity_bisection) {float(meas['reqs_s']):.2f} req/s; limit uses the model's lower {own:.2f}"
    else:
        source = "measured (capacity_bisection)"
    return Tier(model, workload, pool, GPUS_PER_REPLICA.get(model, 1),
                float(meas["reqs_s"]) if meas else own, source, c.t0, c.t1)


PROJECTED_GPUS, PROJECTED_BATCH = 8, 512


def predicted_drain_s(model: str, workload: str = "sharegpt") -> float:
    """The offline drain of one full replica at its cap: t0 max(q) + t1 sum(q) over length-biased
    residuals (the predictor, at a median ratio of 0.97-1.08 against measured drains)."""
    from analyze.plot_single_gpu_model import offline_drain
    tier = make_tier(model, workload, 1)
    hw = CAL_ROOTS.get(model, ("h100-80gb", ""))[0]
    return float(offline_drain(_CELLS[hw][(model, workload)], int(tier.pool.c_eff)))


@dataclass
class Measured:
    """One replica of `model` on `workload`, from measurements only."""
    model: str
    workload: str
    gpus: int
    c_reqs_s: float          # capacity at the datacenter cap (capacity_bisection)
    p_cap_w: float           # power the replica drew at that capacity (capacity_bisection's stable probe)
    p_idle_w: float          # idle power with the model loaded (concurrency_sweep)
    source: str

    @property
    def e_j(self) -> float:
        """Energy a request adds above idle on a full replica."""
        return (self.p_cap_w - self.p_idle_w) / self.c_reqs_s

    @property
    def reqs_s_per_gpu(self) -> float:
        return self.c_reqs_s / self.gpus


def measured_cell(model: str, workload: str) -> Measured:
    import json as _json
    from analyze import capacity_lookup
    from analyze.plot_single_gpu_model import latest_concurrency_runs
    hw, sub = CAL_ROOTS.get(model, ("h100-80gb", ""))
    root = R / sub if sub else R
    cap = tier_cap(model)
    meas = capacity_lookup.lookup(root, model, workload, cap)
    if meas is None:
        raise SystemExit(f"no capacity_bisection capacity for {model} on {workload} at cap {cap} ({hw})")
    j = _json.loads((root / "capacity_bisection" / meas["source"] / "capacity.json").read_text())
    run = latest_concurrency_runs(root).get((model, workload))
    if run is None or j.get("power_at_capacity_w") is None:
        raise SystemExit(f"no concurrency_sweep idle or capacity_bisection power at capacity for {model} on {workload} ({hw})")
    idle = _json.loads((run / "summary.json").read_text())["meta"]["idle"]["idle_power_w"]
    return Measured(model, workload, GPUS_PER_REPLICA.get(model, 1), float(meas["reqs_s"]),
                    float(j["power_at_capacity_w"]), float(idle), f"{meas['source']} / {run.name}")


# ---- the three main experiments (the week, the 4-hour event, GPUs that leave) on a second pair: 70B -> 24B ----
# The reference site's recipe with only the models changed: 10,000 GPUs, REF small_share of them on the small
# model, ShareGPT shapes, the busiest 5 minutes at u_peak of the site's measured capacity, each pool under
# U_POOL. Llama-3.1-70B runs as one instance over four GPUs, so its 7,500 GPUs are 1,875 replicas; Mistral-24B
# runs on one. No fleet of either was run on GPUs: the pools are their single-instance calibrations.

SITE2 = {"large": "llama31-70b", "small": "mistral-24b", "tag": "70b24b",
         "label": "Llama-3.1-70B (4 GPUs) -> Mistral-24B"}
STEP_ONSET_S = 3.0        # the measured drop's first seconds (router reaction, the 1 Hz meter): not stretched
RETURN_ONSET_S = 1.0


def stretched_kernel(k: float, workload: str = "sharegpt") -> tuple[pd.DataFrame, dict]:
    """The measured drop and return under the model (step_response, the 8B -> 1B pair on four H100s) with
    everything after the first STEP_ONSET_S (RETURN_ONSET_S) seconds stretched k-fold: a drop lands in the
    darkened replicas' drain time plus a near-constant overhead, so a k-times longer drain stretches
    the drain part and leaves the overhead. A projection, not a measurement: it ignores that the small pool
    fills and drains on its own clock (the 24B's 13 s against the 1B's 2 s)."""
    s = pd.read_csv(OUT / "step_response.csv")
    g = s[s.workload == workload].sort_values("t_rel_s")
    t = g.t_rel_s.to_numpy(dtype=float)
    src = t.copy()
    d = (t > STEP_ONSET_S) & (t < 50.0)
    src[d] = np.minimum(STEP_ONSET_S + (t[d] - STEP_ONSET_S) / k, 50.0)    # k < 1 must not read the return
    r = t >= 50.0 + RETURN_ONSET_S
    src[r] = 50.0 + RETURN_ONSET_S + (t[r] - 50.0 - RETURN_ONSET_S) / k
    out = pd.DataFrame({"workload": workload, "t_rel_s": t})
    for col in ("y_p50", "y_p10", "y_p90"):
        out[col] = np.interp(src, t, g[col].to_numpy(dtype=float))
    down = out[(out.t_rel_s >= 0) & (out.t_rel_s < 50) & (out.y_p50 >= 0.9)].t_rel_s
    back = out[(out.t_rel_s >= 50) & (out.y_p50 <= 0.1)].t_rel_s
    summ = {"workload": workload, "stretch": k, "t_down90_s": float(down.min()) if len(down) else np.nan,
            "t_back90_s": float(back.min() - 50) if len(back) else np.nan}
    return out, summ


def site2_counts() -> tuple[int, int]:
    """SITE2's replica counts on the reference recipe (REF small_share of REF n_gpus on the small model)."""
    gl, gs = GPUS_PER_REPLICA.get(SITE2["large"], 1), GPUS_PER_REPLICA.get(SITE2["small"], 1)
    small_gpus = int(round(REF["n_gpus"] * REF["small_share"]))
    return (REF["n_gpus"] - small_gpus) // gl, small_gpus // gs


def site_tables(profile: pd.DataFrame, make, nl0: int, ns0: int, consts_fn, drain_large_s: float) -> dict[str, pd.DataFrame]:
    """The three main experiments on a two-pool site, table for table as the reference site's: the week and
    the 4-hour event at every flexible share, the GPUs-available grid with the routing window at three
    moments, the night-lending week, which pool the GPUs leave, the 25 % event target, the pools' constants
    and the event's 1-s kernel. `make(share, n_large, n_small)` builds the fleet (DCFleet or LinearFleet,
    replica counts)."""
    shape = profile["shape"].to_numpy()
    full = make(1.0, nl0, ns0)
    lam = REF["u_peak"] * full.capacity * shape            # one traffic for every share and availability
    moments = {"busiest": float(np.max(shape)), "median": float(np.median(shape)), "quietest": float(np.min(shape))}
    weeks, events, ev_summ, avail, win = [], [], [], [], []
    for s in FLEX_SHARES:
        fl = make(s, nl0, ns0)
        w = week_table(profile, fleet=fl)
        weeks.append(w[["t", "u", "fac_base_mw", "fac_floor_mw", "fac_flex_mw", "flex_frac_fac",
                        "util_small_floor"]].assign(flexible_share=s))
        tab, summ = event_table(profile, "max", fleet=fl)
        events.append(tab[["t", "local", "hour_local", "in_event", "fac_mw", "fac_base_mw",
                           "fac_floor_mw"]].assign(flexible_share=s))
        ev_summ.append({"flexible_share": s, **summ})
        mean_full = None
        for a in AVAIL_GRID:
            fa = make(s, int(round(a * nl0)), int(round(a * ns0)))
            e = envelope_fixed_traffic(fa, lam)
            ok = e.fac_flex_mw.notna()
            f = e.fac_flex_mw.fillna(0.0)
            mean = float(f.mean())
            mean_full = mean if mean_full is None else mean_full
            avail.append({"flexible_share": s, "available_frac": a, "n_gpus": fa.n_gpus,
                          "served_frac_of_week": float(ok.mean()), "u_peak_on_available": float(lam.max() / fa.capacity),
                          "fac_flex_mean_mw": mean, "fac_flex_p5_mw": float(f.quantile(0.05)),
                          "fac_flex_kept_of_full": mean / mean_full if mean_full else np.nan,
                          "fac_base_mean_mw": float(e.fac_base_mw[ok].mean()) if ok.any() else np.nan})
            for name, sh in moments.items():
                lam_m = REF["u_peak"] * full.capacity * sh
                lo, hi = fa.window(lam_m)
                win.append({"flexible_share": s, "available_frac": a, "moment": name, "lam_reqs_s": lam_m,
                            "u_on_available": lam_m / fa.capacity, "f_lo": float(lo[0]), "f_hi": float(hi[0])})
    _, s25 = event_table(profile, 0.25, fleet=make(1.0, nl0, ns0))
    ev_summ.append({"flexible_share": 1.0, **s25})
    loc = profile.t.dt.tz_convert(EVENT_TZ)
    day = ((loc.dt.weekday < 5) & (loc.dt.hour >= 8) & (loc.dt.hour < 20)).to_numpy()
    lending, pools = [], []
    for s in (REF["flexible_share"], 1.0):
        fl_full = make(s, nl0, ns0)
        fl_lent = make(s, int(round((1.0 - LEND_FRACTION) * nl0)), int(round((1.0 - LEND_FRACTION) * ns0)))
        e_full, e_lent = envelope_fixed_traffic(fl_full, lam), envelope_fixed_traffic(fl_lent, lam)
        pick = lambda col: np.where(day, e_full[col], e_lent[col])
        lending.append(pd.DataFrame({"t": profile.t, "flexible_share": s,
                                     "inference_gpus": np.where(day, fl_full.n_gpus, fl_lent.n_gpus),
                                     "u": pick("u"), "u_full": e_full.u.to_numpy(),
                                     "fac_base_mw": pick("fac_base_mw"), "fac_flex_mw": pick("fac_flex_mw"),
                                     "fac_flex_mw_always_full": e_full.fac_flex_mw.to_numpy()}))
        nl, ns, gl = fl_full.n_large, fl_full.n_small, fl_full.gpus_large
        for rr in (round(0.05 * k, 2) for k in range(11)):
            kg = int(round(rr * fl_full.n_gpus))
            rules = {"both in proportion": (nl - int(round(rr * nl)), ns - int(round(rr * ns))),
                     "70B pool only": (nl - int(round(kg / gl)), ns), "24B pool only": (nl, ns - kg)}
            for rule, (l, sm) in rules.items():
                if l <= 0 or sm <= 0:
                    continue
                e = envelope_fixed_traffic(make(s, l, sm), lam)
                f = e.fac_flex_mw.fillna(0.0)
                pools.append({"flexible_share": s, "removed_frac": rr, "rule": rule, "n_large": l, "n_small": sm,
                              "served_frac_of_week": float(e.fac_flex_mw.notna().mean()),
                              "fac_flex_mean_mw": float(f.mean()), "fac_flex_p5_mw": float(f.quantile(0.05))})
    consts = pd.DataFrame(consts_fn(full)).assign(site_capacity_reqs_s=full.capacity, lam_peak_reqs_s=float(lam.max()),
                                                  lam_mean_reqs_s=float(lam.mean()), n_gpus=full.n_gpus)
    kern, ksum = stretched_kernel(float(drain_large_s / predicted_drain_s(PAIRS["llama"][0])))
    return {"week_by_share": pd.concat(weeks, ignore_index=True),
            "event_by_share": pd.concat(events, ignore_index=True),
            "event_summary_by_share": pd.DataFrame(ev_summ),
            "availability_by_share": pd.DataFrame(avail), "availability_window": pd.DataFrame(win),
            "availability_by_pool": pd.DataFrame(pools), "week_lending": pd.concat(lending, ignore_index=True),
            "site_constants": consts, "step_response": kern, "step_summary": pd.DataFrame([ksum])}


# ---- the same site at the Azure trace's own request lengths (chat, code) ---------------------------------
# At Azure's long prompts vLLM's 2,048-token prefill budget caps the batch near 2048*D/(P+D), so the step-time
# line does not describe a full replica; the pools are then the linear law on measured constants, as in
# pairs_shapes. Chat lengths ride the chat trace's timing, code lengths the code trace's.

AZURE_TIMING = {"azure_conv_2024": "azure_conv_2024_week", "azure_code_2024": "azure_code_2024_week"}
AZURE_TAG = {"azure_conv_2024": "chat", "azure_code_2024": "code"}


@dataclass
class LinearFleet:
    """Two pools by the linear law on measured constants (`Measured` cells): a pool of n replicas carrying lam
    req/s draws n P(0) + lam e, e = [P(c) - P(0)] / C, and may carry U_POOL n C (measured C). The interface is
    DCFleet's (window / split / power / capacity / n_gpus), so week_table, event_table and the availability
    code run on it unchanged."""
    pair: str
    workload: str
    n_large: int
    n_small: int
    flexible_share: float
    large: "Measured"
    small: "Measured"

    @property
    def gpus_large(self) -> int:
        return self.large.gpus

    @property
    def gpus_small(self) -> int:
        return self.small.gpus

    @property
    def c_large(self) -> float:
        return self.large.c_reqs_s

    @property
    def c_small(self) -> float:
        return self.small.c_reqs_s

    @property
    def n_gpus(self) -> int:
        return self.n_large * self.gpus_large + self.n_small * self.gpus_small

    @property
    def capacity(self) -> float:
        return self.n_large * self.c_large + self.n_small * self.c_small

    @property
    def firm_small_share(self) -> float:
        return firm_small_share(self.n_large, self.c_large, self.n_small, self.c_small)

    def window(self, lam) -> tuple[np.ndarray, np.ndarray]:
        lim_l, lim_s = U_POOL * self.n_large * self.c_large, U_POOL * self.n_small * self.c_small
        return routing_window(lam, self.flexible_share, self.firm_small_share, lim_l, lim_s)

    def split(self, lam, f) -> tuple[np.ndarray, np.ndarray]:
        """(large, small) req/s: the firm queries in the FIRM_SPLIT ratio, the flexible ones by f."""
        lam = np.asarray(lam, dtype=float)
        s = self.flexible_share
        small = lam * ((1.0 - s) * self.firm_small_share + s * np.asarray(f, dtype=float))
        return lam - small, small

    def power(self, lam, f) -> dict:
        lam_l, lam_s = (np.atleast_1d(x).astype(float) for x in self.split(lam, f))

        def pool(m, n, x):
            p = np.where(x <= n * m.c_reqs_s * (1 + 1e-12), n * m.p_idle_w + x * m.e_j, np.nan)
            return p, np.where(np.isfinite(p), np.ceil(np.maximum(x, 0.0) / m.c_reqs_s - 1e-12), np.nan)

        pl, kl = pool(self.large, self.n_large, lam_l)
        ps, ks = pool(self.small, self.n_small, lam_s)
        nan = np.full(lam_l.shape, np.nan)
        return {"lam_large": lam_l, "lam_small": lam_s, "p_large_w": pl, "p_small_w": ps, "p_total_w": pl + ps,
                "lit_large": kl, "lit_small": ks, "inflight_large": nan, "inflight_small": nan}


def capacity_running(model: str, workload: str) -> float:
    """Requests resident on one replica at its measured capacity: capacity_bisection's stable probe at that rate."""
    import json as _json
    from analyze import capacity_lookup
    hw, sub = CAL_ROOTS.get(model, ("h100-80gb", ""))
    root = R / sub if sub else R
    meas = capacity_lookup.lookup(root, model, workload, tier_cap(model))
    j = _json.loads((root / "capacity_bisection" / meas["source"] / "capacity.json").read_text())
    at = [p for p in j.get("probes", []) if p.get("stable") and abs(p["rate_reqs_s"] - j["capacity_reqs_s"]) < 1e-9]
    return float(at[-1]["running_mean"])


def azure_drain_s(model: str, workload: str) -> float:
    """Projected drain of a full replica at `workload`'s lengths: the offline predictor (t0 max q + t1 sum q over
    length-biased residuals of that workload's output lengths) with the model's DECODE step time -- its ShareGPT
    cell's t0, t1, since a replica that stops admitting only decodes -- and the resident count measured at
    capacity. Longer contexts make the real decode step somewhat slower, so this leans short."""
    import dataclasses
    from analyze.plot_single_gpu_model import offline_drain
    make_tier(model, "sharegpt", 1)
    hw = CAL_ROOTS.get(model, ("h100-80gb", ""))[0]
    cell = dataclasses.replace(_CELLS[hw][(model, "sharegpt")], wl=workload)
    return float(offline_drain(cell, int(round(capacity_running(model, workload)))))


def _linear_consts(full: LinearFleet) -> list[dict]:
    rows = []
    for role, m, n in (("large", full.large, full.n_large), ("small", full.small, full.n_small)):
        rows.append({"role": role, "model": m.model, "replicas": int(n), "gpus_per_replica": m.gpus,
                     "gpus": int(n) * m.gpus, "capacity_measured_reqs_s": m.c_reqs_s, "capacity_model_reqs_s": np.nan,
                     "limit_reqs_s": U_POOL * n * m.c_reqs_s, "p_idle_w_per_gpu": m.p_idle_w / m.gpus,
                     "p_full_w_per_gpu": m.p_cap_w / m.gpus, "e_inc_j": m.e_j,
                     "running_at_capacity": capacity_running(m.model, full.workload),
                     "drain_full_replica_s": azure_drain_s(m.model, full.workload)})
    return rows


def site2_azure_tables(workload: str) -> dict[str, pd.DataFrame]:
    """The three main experiments on SITE2 at the Azure trace's own `workload` lengths, on that trace's timing."""
    nl0, ns0 = site2_counts()
    cl, cs = measured_cell(SITE2["large"], workload), measured_cell(SITE2["small"], workload)
    make = lambda s, nl, ns: LinearFleet(f"{SITE2['large']} -> {SITE2['small']}", workload, int(nl), int(ns),
                                         float(s), cl, cs)
    return site_tables(load_profile(AZURE_TIMING[workload]), make, nl0, ns0, _linear_consts,
                       azure_drain_s(SITE2["large"], workload))


SCENARIOS = ("week", "step", "programs", "shares", "availability", "availability_detail", "site_70b24b_azure")


def run(only: set[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "step" in only:
        shape, summ = step_response()
        shape.to_csv(OUT / "step_response.csv", index=False)
        summ.to_csv(OUT / "step_summary.csv", index=False)
        print(summ.round(2).to_string(index=False))
    if only & {"week", "event"}:
        prof = load_profile()
    if "week" in only:
        w = week_table(prof)
        w.to_csv(OUT / "week.csv", index=False)
        week_table(prof, flexible_share=1.0).to_csv(OUT / "week_flex100.csv", index=False)
        sens = week_sensitivity(prof)
        sens.to_csv(OUT / "week_sensitivity.csv", index=False)
        facility_sensitivity(prof).to_csv(OUT / "facility_sensitivity.csv", index=False)
        print(sens.round(3).to_string(index=False))
    if "programs" in only:
        pr, ramps = programs_tables()
        pr.to_csv(OUT / "grid_programs.csv", index=False)
        pd.read_csv(INPUTS / "ramp_limits.csv").to_csv(OUT / "ramp_limits.csv", index=False)
        ramps.to_csv(OUT / "ramp_times.csv", index=False)
        print(pr[["label", "response_s", "duration_s", "routing_meets_response"]].to_string(index=False))
        print(ramps.round(1).to_string(index=False))
    if "availability" in only:
        a, b = availability_tables(load_profile())
        a.to_csv(OUT / "availability.csv", index=False)
        b.to_csv(OUT / "week_lending.csv", index=False)
        print(a.round(3).to_string(index=False))
        for s, g in b.groupby("flexible_share"):
            print(f"lending s={s}: inference flex mean {g.fac_flex_mw.mean():.2f} MW vs always-full "
                  f"{g.fac_flex_mw_always_full.mean():.2f}; GPU-hours lent {((g.inference_gpus.max() - g.inference_gpus) * 5 / 60).sum():,.0f}")
    if "site_70b24b_azure" in only:
        for wl, tag in AZURE_TAG.items():
            tabs = site2_azure_tables(wl)
            for name, d in tabs.items():
                d.to_csv(OUT / f"{name}_{SITE2['tag']}_{tag}.csv", index=False)
            print(f"== {wl}")
            print(tabs["site_constants"].round(3).T.to_string())
            w = tabs["week_by_share"]
            print(w.groupby("flexible_share").agg(flex_mean_mw=("fac_flex_mw", lambda x: x.fillna(0).mean()),
                                                  unserved=("fac_flex_mw", lambda x: x.isna().mean()),
                                                  frac_mean=("flex_frac_fac", "mean")).round(3).to_string())
            print(tabs["event_summary_by_share"][["flexible_share", "depth", "day", "fac_base_event_mean_mw",
                                                  "fac_shed_mean_mw", "shed_frac_mean"]].round(3).to_string(index=False))
            print(tabs["step_summary"].round(2).to_string(index=False))
    if "availability_detail" in only:
        a, w = availability_detail_tables(load_profile())
        a.to_csv(OUT / "availability_by_share.csv", index=False)
        w.to_csv(OUT / "availability_window.csv", index=False)
        half = lambda d: d[np.isclose(d.flexible_share, REF["flexible_share"])]
        print(half(a).round(3).to_string(index=False))
        print(half(w).round(3).to_string(index=False))
        p = removal_by_pool_table(load_profile())
        p.to_csv(OUT / "availability_by_pool.csv", index=False)
        for s, g in p.groupby("flexible_share"):
            print(f"flexible {s:.0%}, week-mean reduction (MW) by removal rule:")
            print(g.pivot_table(index="removed_frac", columns="rule", values="fac_flex_mean_mw").round(2).to_string())
    if "shares" in only:
        tabs = shares_tables(load_profile())
        for name, t in tabs.items():
            t.to_csv(OUT / f"{name}.csv", index=False)
        w = tabs["week_by_share"]
        print(w.groupby("flexible_share").agg(flex_mean_mw=("fac_flex_mw", "mean"),
                                              flex_min_mw=("fac_flex_mw", "min"),
                                              frac_mean=("flex_frac_fac", "mean")).round(3).to_string())
        c = tabs["curtailment_by_share"]
        print(c[np.isclose(c.rate, 0.005)].pivot_table(index="flexible_share", columns=["ba", "year"],
                                                        values="routing_covers_mwh_frac").round(3).to_string())


def _selfcheck() -> None:
    """The closed form must reproduce the library's exact solve, at 4 GPUs and at 400."""
    for pair in PAIRS:
        for wl in ("sharegpt", "arxiv_summarization"):
            for nl, ns in ((3, 1), (300, 100)):
                fl = build_fleet(pair, wl, nl, ns)
                from lib import power_model as pm
                fm = pm.FleetModel(large=fl.large, small=fl.small, flexible_share=1.0,
                                   solver="exact")
                lam_max = 0.85 * fl.capacity
                worst = 0.0
                for lam in np.linspace(0.05, 1, 9) * lam_max:
                    f_lo, f_hi = (x[0] for x in fl.window(lam))
                    if not math.isfinite(f_lo):
                        continue
                    for f in np.linspace(f_lo, f_hi, 5):
                        ref = fm.power(lam, float(f), "pack")["power_total_w"]
                        got = fl.power(lam, f)["p_total_w"][0]
                        if math.isfinite(ref):
                            worst = max(worst, abs(got - ref) / ref)
                assert worst < 1e-6, (pair, wl, nl, ns, worst)
    # A pool past its capacity is NaN, an empty pool idles every replica.
    fl = build_fleet("llama", "sharegpt", 3, 1)
    big = pool_closed_form(fl.large, [0.0, 1e9])
    assert big["power_w"][0] == 3 * float(fl.large.curve.power(0)) and math.isnan(big["power_w"][1])
    # The window: f_lo rises from 0 once the large pool alone cannot carry the load.
    f_lo, f_hi = fl.window(np.array([10.0, 0.7 * fl.capacity]))
    assert f_lo[0] == 0.0 and f_hi[0] == 1.0 and f_lo[1] > 0.0
    # The linear fleet: its room is the moved queries times the energy gap, and its window's edges put the large
    # (f_lo) and the small (f_hi) pool exactly at U_POOL.
    lg = Measured("L", "w", 4, 7.0, 2700.0, 480.0, "stub")
    sm = Measured("S", "w", 1, 6.0, 680.0, 110.0, "stub")
    lf = LinearFleet("L -> S", "w", 100, 50, 0.5, lg, sm)
    lam = np.array([0.3, 0.6, 0.75]) * lf.capacity
    lo, hi = lf.window(lam)
    room = lf.power(lam, lo)["p_total_w"] - lf.power(lam, hi)["p_total_w"]
    assert np.allclose(room, lam * 0.5 * (hi - lo) * (lg.e_j - sm.e_j))
    sp_lo, sp_hi = lf.power(lam, lo), lf.power(lam, hi)
    busy = lo > 0
    assert np.allclose(sp_lo["lam_large"][busy], U_POOL * 100 * lg.c_reqs_s)
    assert np.allclose(sp_hi["lam_small"][hi < 1], U_POOL * 50 * sm.c_reqs_s)
    assert math.isnan(lf.power(np.array([1e9]), np.array([0.0]))["p_total_w"][0])
    # The firm split: with no flexible query on the small pool, the firm half of a 50 %-flexible fleet sits on the
    # two pools in the ratio of their measured capacities ...
    global FIRM_SPLIT
    fl = build_fleet("llama", "sharegpt", 300, 100, 0.5)
    lam = np.array([0.2, 0.5, 0.8, 0.95]) * fl.capacity
    phi = 100 * fl.c_small / fl.capacity
    lam_l, lam_s = fl.split(lam, np.zeros(lam.shape))
    assert FIRM_SPLIT == "capacity" and np.isclose(fl.firm_small_share, phi)
    assert np.allclose(lam_s, 0.5 * lam * phi) and np.allclose(lam_l, 0.5 * lam * (1 - phi) + 0.5 * lam)
    # ... the window's edges still put a pool exactly at its limit ...
    lo, hi = fl.window(lam)
    lim_l, lim_s = fl._limit(fl.large, fl.c_large), fl._limit(fl.small, fl.c_small)
    sl, ss = fl.split(lam, lo), fl.split(lam, hi)
    assert np.allclose(sl[0][lo > 0], lim_l) and np.allclose(ss[1][hi < 1], lim_s)
    # ... and "large" gives back the closed form of every firm query on the large model.
    keep, FIRM_SPLIT = FIRM_SPLIT, "large"
    try:
        lo, hi = fl.window(lam)
        with np.errstate(invalid="ignore", divide="ignore"):
            old_lo, old_hi = np.clip((1 - lim_l / lam) / 0.5, 0, None), np.clip(lim_s / (lam * 0.5), None, 1)
        ok = old_lo <= old_hi
        assert np.allclose(lo[ok], old_lo[ok]) and np.allclose(hi[ok], old_hi[ok]) and np.isnan(lo[~ok]).all()
    finally:
        FIRM_SPLIT = keep
    # With the capacity split every flexible share serves the reference site's busiest 5 minutes on every GPU.
    for s in FLEX_SHARES:
        r = ref_fleet(flexible_share=s)
        assert np.isfinite(r.window(np.array([REF["u_peak"] * r.capacity]))[0][0]), s
    # Normal operation (BASE_ROUTING = "capacity"): every query split by capacity, so both pools run at the site's
    # utilization and the baseline is the same at every flexible share; "large" gives back f_lo.
    global BASE_ROUTING
    assert BASE_ROUTING == "capacity"
    ref1 = ref_fleet(flexible_share=1.0)
    lam = np.array([0.15, 0.35, 0.6]) * ref1.capacity
    e1 = envelope(ref1, lam)
    q1 = ref1.power(lam, e1["f_base"])
    assert np.allclose(q1["lam_small"] / (ref1.n_small * ref1.c_small), lam / ref1.capacity)
    assert np.allclose(q1["lam_large"] / (ref1.n_large * ref1.c_large), lam / ref1.capacity)
    for s in FLEX_SHARES:
        assert np.allclose(envelope(ref_fleet(flexible_share=s), lam)["p_base_w"], e1["p_base_w"]), s
    keep, BASE_ROUTING = BASE_ROUTING, "large"
    try:
        e = envelope(ref_fleet(flexible_share=0.5), lam)
        assert np.allclose(e["f_base"], e["f_lo"]) and np.allclose(e["p_base_w"], e["p_ceiling_w"])
    finally:
        BASE_ROUTING = keep
    print("datacenter_sim selfcheck ok: closed form == exact solve at 4 and 400 GPUs; "
          "NaN past capacity; window edges; linear fleet room = moved x energy gap; "
          "firm queries in the capacity ratio, 'large' = every firm query on the large model, every share serves the reference peak; "
          "normal operation splits every query by capacity (one baseline for every share), 'large' = f_lo")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=",".join(SCENARIOS),
                    help=f"comma-separated subset of {SCENARIOS}")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--inputs", type=Path, default=None,
                    help="directory of the reduced public inputs (default data/datacenter)")
    ap.add_argument("--out", type=Path, default=None,
                    help="where the tables go (default results/tables/datacenter)")
    args = ap.parse_args()
    global INPUTS, OUT
    if args.inputs is not None:
        INPUTS = args.inputs
    if args.out is not None:
        OUT = args.out
    if args.selfcheck:
        _selfcheck()
        return
    run(set(args.only.split(",")))


if __name__ == "__main__":
    main()
