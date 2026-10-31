"""The analytical power model: one GPU, then a pool, then a two-tier fleet.

Coordinate
----------
Power is modelled as a function of **b, the number of sequences the model is actually
running** -- ``scheduler.running``, not running+waiting, not tokens/s, not arrival
rate. That choice is the whole model:

  * A queued request draws no power. Counting it makes a saturated replica and a
    backlogged one look identical.
  * Tokens/s is an *output* of the batch, not a description of it. Two workloads at
    the same tok/s with different batch geometry draw different power, and a router
    dispatches sequences anyway -- it cannot dispatch a token rate.
  * b is what the packing rule controls directly, so a model in b is a model in the
    thing the knob actually moves.

The single-GPU law
------------------
    P(b) = P_idle + (P_busy - P_idle) * g(b),    g(0) = 0,  g concave, g(inf) -> 1

``P_idle`` is weights-resident-but-no-work (measured, not fitted). ``P_busy`` is the
high-concurrency asymptote. ``g`` is concave because the marginal sequence adds less
power than the one before it: at small b the GPU is latency-bound and each added
sequence fills idle SM time, while at large b the SMs are already saturated and the
extra sequence only makes each step longer.

**The concavity is not a curve-fitting detail -- it is the mechanism.** For concave g
and a fixed total in-flight N, Jensen's inequality says sum_i g(b_i) is *maximized*
when the b_i are equal. So spreading N sequences evenly over the pool draws the most
power and packing them onto the fewest replicas draws the least. That is precisely why
first-fit packing is a power lever, and it is testable: run pack and spread at matched
N, and the model predicts the sign and the size of the gap in advance.

Candidate forms for g are fitted and compared rather than assumed; see ``FORMS``. The
one non-obvious candidate is ``jump_mm``, which adds a discontinuity at b=0->1: a GPU
with *any* work on it pays a fixed wake-up cost before concurrency matters at all.
Whether that term is real is an empirical question this suite settles per model.

What this model does NOT claim
------------------------------
It predicts the **mean** power at a given state. It is structurally silent about the
*variance* -- the power swing. That is not an oversight: with g concave, feeding a mean
occupancy into g over-predicts the mean of g by exactly the Jensen gap, and that gap is
the swing. A model in expectation can locate the centre of the band and cannot give its
width. Anything about excursion has to be measured.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

# ======================================================================================
# g(b): the concave occupancy -> power shape
# ======================================================================================

def _g_mm(b, b50):
    """Michaelis-Menten / hyperbolic. Concave for all b>0. One parameter: the
    concurrency at which half the dynamic range is reached."""
    return b / (b + b50)


def _g_exp(b, b0):
    """Saturating exponential. Concave everywhere; approaches 1 faster than mm."""
    return 1.0 - np.exp(-b / b0)


def _g_pow(b, bsat, alpha):
    """Clipped power law. Concave for alpha < 1; reaches exactly 1 at b = bsat and
    stays there, which matches a plant that genuinely saturates rather than
    asymptoting."""
    return np.clip((b / bsat) ** alpha, 0.0, 1.0)


def _g_hill(b, b50, alpha):
    """Hill. alpha < 1 concave, alpha > 1 sigmoid. Included so the fit can *reject*
    concavity if the data says the curve has an S-shape."""
    bb = np.power(np.maximum(b, 0.0), alpha)
    return bb / (bb + b50 ** alpha)


def _g_mm_relax(b, b50, q, bq):
    """Rise, then relax: [b/(b+b50)] * [1 - q*b/(b+bq)].

    Needed because the measured curve is **not monotone**. On Llama-3.2-1B at 128/128
    power rises 300 -> 345 W to a peak near b=64 and then falls back to 323 W by b=505,
    with the SM clock pinned at 1980 MHz the whole way (so this is not DVFS) and GPU
    utilization falling 75% -> 39%. Bigger batches make each step longer and leave the
    device idle a larger fraction of the time, so mean power drops even as throughput
    rises. A monotone family has to average through that, and mis-predicts both ends.

    g(0)=0, g(inf) = 1-q, with an interior maximum when bq > b50. q is the fraction of
    the peak given back at high concurrency; bq sets how fast it is given back.
    """
    b = np.asarray(b, dtype=float)
    return (b / (b + b50)) * (1.0 - q * b / (b + bq))


def _g_jump_mm(b, b50, jump):
    """mm plus a wake-up discontinuity: any b >= 1 immediately costs ``jump`` of the
    dynamic range, and concurrency scales the rest. g(0) is still exactly 0."""
    base = b / (b + b50)
    return np.where(np.asarray(b) <= 0, 0.0, jump + (1.0 - jump) * base)


def _g_jump_relax(b, b50, jump, q, bq):
    """All three features the measured curves actually show, in one form:

      * a **wake-up jump** -- on a 1B at 128/128, 81% of the whole dynamic range
        appears with a single sequence in flight, so the dominant power question is
        "is this replica busy at all", not "how busy";
      * a concave **rise** with concurrency over the remaining range;
      * a **relaxation** at high concurrency, where longer steps leave the device idle
        a larger share of the time and mean power falls back.

    g(0) = 0 exactly; for b >= 1 it is jump + (1-jump)*mm(b) scaled by the relaxation.
    """
    b = np.asarray(b, dtype=float)
    core = jump + (1.0 - jump) * (b / (b + b50))
    return np.where(b <= 0, 0.0, core * (1.0 - q * b / (b + bq)))


@dataclass(frozen=True)
class GForm:
    name: str
    fn: callable
    param_names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    init: tuple[float, ...]
    concave: bool
    doc: str
    monotone: bool = True


FORMS: dict[str, GForm] = {
    f.name: f for f in [
        GForm("mm", _g_mm, ("b50",), ((0.5, 2000.0),), (24.0,), True,
              "b/(b+b50); one parameter, concave everywhere"),
        GForm("exp", _g_exp, ("b0",), ((0.5, 2000.0),), (24.0,), True,
              "1-exp(-b/b0); concave, saturates faster than mm"),
        GForm("pow", _g_pow, ("bsat", "alpha"), ((1.0, 4000.0), (0.05, 1.5)), (128.0, 0.5),
              True, "min(1,(b/bsat)^alpha); concave iff alpha<1, saturates exactly"),
        GForm("hill", _g_hill, ("b50", "alpha"), ((0.5, 2000.0), (0.05, 4.0)), (24.0, 1.0),
              False, "b^a/(b^a+b50^a); a>1 would mean the curve is NOT concave"),
        GForm("jump_mm", _g_jump_mm, ("b50", "jump"), ((0.5, 2000.0), (0.0, 0.95)),
              (24.0, 0.3), True, "mm plus a wake-up jump at b=0->1"),
        GForm("jump_relax", _g_jump_relax, ("b50", "jump", "q", "bq"),
              ((0.5, 2000.0), (0.0, 0.95), (0.0, 0.9), (1.0, 20000.0)),
              (16.0, 0.5, 0.1, 200.0), False,
              "wake-up jump + concave rise + high-concurrency relaxation (NON-monotone)",
              monotone=False),
        GForm("mm_relax", _g_mm_relax, ("b50", "q", "bq"),
              ((0.5, 2000.0), (0.0, 0.9), (1.0, 20000.0)), (24.0, 0.15, 400.0),
              False, "mm times a saturating decay; peaks then relaxes (NON-monotone)",
              monotone=False),
    ]
}


# ======================================================================================
# Single-replica curve
# ======================================================================================

@dataclass
class PowerCurve:
    """P(b) for one GPU running one model."""

    form: str
    params: tuple[float, ...]
    p_idle_w: float
    p_busy_w: float
    model_slug: str = ""
    workload: str = ""
    fit_stats: dict = field(default_factory=dict)

    @property
    def dynamic_range_w(self) -> float:
        """``p_busy_w - p_idle_w``: the model's SCALE, not the achieved swing.

        For forms whose g does not reach 1 -- ``jump_relax`` peaks below 1 by
        construction, since the relaxation term multiplies it down -- the fitted
        ``p_busy_w`` inflates to compensate and this over-states the real range badly
        (1209 W for an 8B whose measured range is ~500 W). Use
        :meth:`achieved_range_w` for anything physical; this stays only because it is
        the parameterization's own scale.
        """
        return self.p_busy_w - self.p_idle_w

    def peak_power_w(self, b_max: float = 512) -> float:
        """Highest power the fitted curve reaches on ``[1, b_max]`` -- the physical peak."""
        b = np.arange(1, int(max(b_max, 1)) + 1)
        return float(np.max(self.power(b)))

    def achieved_range_w(self, b_max: float = 512) -> float:
        """Peak minus idle over the reachable range: the swing that actually exists."""
        return self.peak_power_w(b_max) - self.p_idle_w

    def g(self, b):
        return FORMS[self.form].fn(np.asarray(b, dtype=float), *self.params)

    def power(self, b):
        """Watts at concurrency b. b=0 returns exactly p_idle."""
        b = np.asarray(b, dtype=float)
        return self.p_idle_w + self.dynamic_range_w * np.where(b <= 0, 0.0, self.g(b))

    def marginal_power(self, b: float) -> float:
        """dP/db -- what the next admitted sequence costs. Falls with b when g is
        concave, which is the quantitative form of 'packing is cheap'."""
        h = max(0.5, 0.01 * max(b, 1.0))
        return float((self.power(b + h) - self.power(max(b - h, 0.0))) / (b + h - max(b - h, 0.0)))

    def to_dict(self) -> dict:
        return {"form": self.form, "params": list(self.params), "p_idle_w": self.p_idle_w,
                "p_busy_w": self.p_busy_w, "model_slug": self.model_slug,
                "workload": self.workload, "fit_stats": self.fit_stats}

    @classmethod
    def from_dict(cls, d: dict) -> "PowerCurve":
        return cls(d["form"], tuple(d["params"]), d["p_idle_w"], d["p_busy_w"],
                   d.get("model_slug", ""), d.get("workload", ""), d.get("fit_stats", {}))


def fit_curve(b: np.ndarray, p: np.ndarray, *, p_idle_w: float,
              form: str = "mm", fit_busy: bool = True,
              p_busy_init: float | None = None) -> PowerCurve:
    """Least-squares fit of one form to measured (b, P).

    ``p_idle_w`` is measured directly at b=0 and held fixed -- it is the one point of
    the curve that needs no model, and letting the optimizer move it lets a bad shape
    hide in a wrong intercept.
    """
    from scipy.optimize import curve_fit  # type: ignore

    b = np.asarray(b, float)
    p = np.asarray(p, float)
    spec = FORMS[form]
    busy0 = p_busy_init if p_busy_init is not None else float(np.max(p))

    if fit_busy:
        def model(bb, p_busy, *ps):
            return p_idle_w + (p_busy - p_idle_w) * np.where(bb <= 0, 0.0, spec.fn(bb, *ps))
        p0 = (busy0,) + spec.init
        lo = (p_idle_w,) + tuple(x[0] for x in spec.bounds)
        hi = (busy0 * 3.0,) + tuple(x[1] for x in spec.bounds)
    else:
        def model(bb, *ps):
            return p_idle_w + (busy0 - p_idle_w) * np.where(bb <= 0, 0.0, spec.fn(bb, *ps))
        p0 = spec.init
        lo = tuple(x[0] for x in spec.bounds)
        hi = tuple(x[1] for x in spec.bounds)

    # curve_fit is a local optimizer and these forms have long flat directions, so a
    # single start strands it on a parameter bound (on the measured 1B curve, mm_relax
    # from one start pins b50 and bq at their lower bounds and loses the interior peak
    # entirely). Restart from several data-derived guesses and keep the best.
    starts = _starting_points(b, p, p_idle_w, spec, busy0, fit_busy)
    best, best_sse = None, float("inf")
    for p0 in starts:
        try:
            popt, _ = curve_fit(model, b, p, p0=p0, bounds=(lo, hi), maxfev=60000)
        except Exception:
            continue
        sse = float(np.sum((p - model(b, *popt)) ** 2))
        if sse < best_sse:
            best, best_sse = popt, sse
    if best is None:
        raise RuntimeError(f"no start converged for form {form!r}")

    p_busy = float(best[0]) if fit_busy else busy0
    params = tuple(float(x) for x in (best[1:] if fit_busy else best))

    curve = PowerCurve(form, params, p_idle_w, p_busy)
    curve.fit_stats = goodness_of_fit(curve, b, p)
    curve.fit_stats["n_params"] = len(params) + (1 if fit_busy else 0)
    curve.fit_stats["n_starts"] = len(starts)
    return curve


def _starting_points(b, p, p_idle_w, spec: GForm, busy0: float, fit_busy: bool):
    """Data-derived initial guesses, plus a spread of fallbacks.

    The data-derived one matters most: b50 is seeded from where the curve actually
    reaches half its range, and (for the relaxing forms) q and bq from the size and
    location of the observed fall-off past the peak. Without that seed the optimizer
    has no reason to look for an interior maximum.
    """
    pos = b > 0
    bb, pp = b[pos], p[pos]
    rng_w = max(float(np.max(pp)) - p_idle_w, 1e-6)
    # half-range crossing -> b50
    half = p_idle_w + 0.5 * rng_w
    idx = np.nonzero(pp >= half)[0]
    b50_seed = float(bb[idx[0]]) if idx.size else float(np.median(bb))
    # wake-up jump: how much of the range is already present at the smallest b measured
    jump_seed = float(np.clip((pp[0] - p_idle_w) / rng_w, 0.0, 0.9))
    # relaxation: fall from the peak to the largest b measured
    k_peak = int(np.argmax(pp))
    q_seed = float(np.clip((pp[k_peak] - pp[-1]) / max(pp[k_peak] - p_idle_w, 1e-6), 0.0, 0.85))
    bq_seed = float(max(bb[k_peak], 2.0))

    seeds = {"b50": b50_seed, "b0": b50_seed, "bsat": float(bb[k_peak]),
             "alpha": 0.5, "jump": jump_seed, "q": max(q_seed, 0.02), "bq": bq_seed}
    derived = tuple(seeds.get(n, d) for n, d in zip(spec.param_names, spec.init))

    cands = [derived, spec.init]
    for scale in (0.25, 4.0):
        cands.append(tuple(v * scale if n in ("b50", "b0", "bsat", "bq") else v
                           for n, v in zip(spec.param_names, derived)))
    out = []
    for c in cands:
        c = tuple(float(np.clip(v, lo + 1e-9, hi - 1e-9))
                  for v, (lo, hi) in zip(c, spec.bounds))
        out.append((busy0,) + c if fit_busy else c)
    return out


def goodness_of_fit(curve: PowerCurve, b: np.ndarray, p: np.ndarray) -> dict:
    pred = curve.power(np.asarray(b, float))
    resid = np.asarray(p, float) - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((p - np.mean(p)) ** 2))
    return {
        "rmse_w": float(np.sqrt(np.mean(resid ** 2))),
        "mae_w": float(np.mean(np.abs(resid))),
        "max_abs_err_w": float(np.max(np.abs(resid))),
        "bias_w": float(np.mean(resid)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n_points": int(len(b)),
    }


def fit_all_forms(b: np.ndarray, p: np.ndarray, *, p_idle_w: float,
                  holdout_every: int = 3) -> dict:
    """Fit every form and rank by **held-out** error, not in-sample.

    In-sample RMSE always improves with parameters, so it cannot choose between a
    1-parameter and a 2-parameter shape. Every ``holdout_every``-th concurrency level
    (by sorted b) is withheld from the fit and scored -- so the ranking answers
    "which shape extrapolates to concurrencies it never saw", which is the only
    property that matters when the model is later asked about a fleet state that was
    never directly measured.
    """
    b = np.asarray(b, float)
    p = np.asarray(p, float)
    order = np.argsort(b)
    b, p = b[order], p[order]
    mask = np.ones(len(b), dtype=bool)
    mask[::holdout_every] = False
    # Always keep the extremes in the fit: extrapolating past the measured range is a
    # different (and unasked) question from interpolating within it.
    mask[0] = mask[-1] = True

    out = {}
    for name in FORMS:
        try:
            full = fit_curve(b, p, p_idle_w=p_idle_w, form=name)
            train = fit_curve(b[mask], p[mask], p_idle_w=p_idle_w, form=name)
            held = goodness_of_fit(train, b[~mask], p[~mask]) if (~mask).any() else {}
            out[name] = {
                "curve": full.to_dict(),
                "in_sample": full.fit_stats,
                "held_out": held,
                "concave": FORMS[name].concave,
                "doc": FORMS[name].doc,
            }
        except Exception as e:  # a form that cannot fit is a result, not a crash
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    ranked = sorted((n for n in out if "error" not in out[n]),
                    key=lambda n: out[n].get("held_out", {}).get("rmse_w", float("inf")))
    return {"forms": out, "ranking_by_heldout_rmse": ranked,
            "best": ranked[0] if ranked else None}


# ======================================================================================
# Pool: n replicas of one model, and the packing rule
# ======================================================================================

def occupancies(n_total: int, n_replicas: int, pack_cap: int,
                policy: str = "pack") -> list[int]:
    """How ``n_total`` in-flight sequences are distributed over ``n_replicas``.

    Both real policies light the **same number** of replicas, ``k = ceil(N/cap)``. They
    differ only in how the load is arranged across those k, which is what isolates the
    effect of the concave power curve from the effect of lighting more GPUs.

    ``pack``       first-fit: k-1 replicas at exactly ``pack_cap``, and the k-th absorbs
                   the remainder. e.g. N=70, cap=20 -> [20, 20, 20, 10].
    ``spread``     equal over the **active set**: choose k from the load, then split
                   evenly among those k. e.g. N=70, cap=20 -> [18, 18, 17, 17]. This is
                   the control arm: same replicas lit, different arrangement, so any
                   power difference is Jensen and nothing else.
    ``spread_all`` equal over **every** replica in the pool, lighting all n regardless of
                   load -- what a naive least-loaded balancer over a fixed fleet does.
                   Kept because it is the common default, but it is not the interesting
                   comparison: it confounds "more replicas lit" with "load arranged
                   differently".
    """
    if n_replicas <= 0:
        return []
    n_total = max(int(n_total), 0)
    cap = max(int(pack_cap), 1)

    if policy == "spread_all":
        base, rem = divmod(n_total, n_replicas)
        return [base + (1 if i < rem else 0) for i in range(n_replicas)]

    if policy not in ("pack", "spread"):
        raise ValueError(f"unknown packing policy {policy!r}")

    k = min(n_replicas, max(1, math.ceil(n_total / cap))) if n_total else 0
    occ = [0] * n_replicas
    if k == 0:
        return occ

    if policy == "spread":
        base, rem = divmod(n_total, k)
        for i in range(k):
            occ[i] = base + (1 if i < rem else 0)
        return occ

    left = n_total
    for i in range(n_replicas):
        take = min(left, cap)
        occ[i] = take
        left -= take
        if left <= 0:
            break
    if left > 0:                     # every replica at cap; spill the remainder evenly
        base, rem = divmod(left, n_replicas)
        occ = [o + base + (1 if i < rem else 0) for i, o in enumerate(occ)]
    return occ


def occupancy_vector(n_total: float, n_replicas: int, cap: int,
                     policy: str = "pack") -> list[float]:
    """``occupancies`` for a CONTINUOUS ``n_total``.

    The integer version is what ``power`` wants, because ``P(b)`` is indexed by a batch
    size. The continuous version is what a root-find wants: bisecting on N needs the
    pool's service rate to be a continuous function of N, and rounding inside the
    objective makes it a staircase whose resolution floors the solve at one sequence.

    Same arrangement in both cases -- ``pack`` fills by index, ``spread`` splits evenly
    over ``k = ceil(N/cap)``, ``spread_all`` over every replica.
    """
    n_total = max(float(n_total), 0.0)
    cap = max(int(cap), 1)
    if n_replicas <= 0 or n_total <= 0:
        return [0.0] * max(n_replicas, 0)
    if policy == "spread_all":
        return [n_total / n_replicas] * n_replicas
    k = min(n_replicas, max(1, int(math.ceil(n_total / cap))))
    if policy == "spread":
        return [n_total / k] * k + [0.0] * (n_replicas - k)
    occ, left = [], n_total
    for _ in range(n_replicas):
        take = min(left, float(cap))
        occ.append(take)
        left -= take
        if left <= 0:
            break
    occ += [0.0] * (n_replicas - len(occ))
    if left > 0:                     # past every cap: spill the remainder evenly
        occ = [o + left / n_replicas for o in occ]
    return occ


@dataclass
class PoolModel:
    """A homogeneous pool: ``n_replicas`` GPUs each running the same model."""

    curve: PowerCurve
    n_replicas: int
    pack_cap: int
    capacity_reqs_s: float = 0.0     # per replica, measured
    residence_a_s: float = 0.0       # T(b) = A + B*b, per-replica residence time
    residence_b_s: float = 0.0
    # The engine's own ceiling on sequences RESIDENT on a replica. vLLM admits at most
    # max_num_seqs; everything else waits in the scheduler's WAITING list, which is not
    # in flight and draws no power. Without this bound solve_inflight grows N until the
    # RATE diverges, so on loads where the cap binds it predicts an occupancy the engine
    # cannot physically hold -- the bound is worth 9.9% -> 5.4% MAPE on sharegpt and
    # 40.0% -> 4.6% on the qwen3-8b/4b pair. Default 0 means "no bound", for callers
    # that do not set it.
    max_running_per_replica: int = 0

    def power(self, n_total: int, policy: str = "pack") -> float:
        occ = occupancies(n_total, self.n_replicas, self.pack_cap, policy)
        return float(sum(self.curve.power(b) for b in occ))

    # ==================================================================================
    # The EXACT solve: occupancy vector instead of a mean field.
    #
    # `solve_inflight` below (the mean field) is the default everywhere; offline callers
    # opt in to this one (FleetModel.solver = "exact").
    #
    # Why it is not cosmetic. `b_bar = N/k` assumes the lit replicas are equally loaded,
    # which is the one thing `pack` exists to prevent: k-1 sit at the cap and one holds
    # the remainder, on purpose, so the rest can go dark. Since mu is concave, Jensen
    # gives k*mu(N/k) >= sum_i mu(b_i) -- the mean field claims throughput the pool does
    # not have and so under-states N. The two agree exactly at saturation, and differ
    # only at partial fill, i.e. exactly where packing acts.
    #
    # Scored like-for-like against the mean field: power 4.0% -> 3.2% MAPE, occupancy
    # 11.8% -> 8.0%, lit count 90% -> 95%, band MAE 114 -> 36 W, and on `pack` runs the
    # occupancy VECTOR itself 34.8% -> 8.2% in L1.
    # ==================================================================================

    @property
    def c_eff(self) -> int:
        """Effective per-replica residency limit: how full a replica can actually get.

        ``pack_cap`` is the router's preference. ``max_running_per_replica`` is what the
        engine will hold -- ``min(max_num_seqs, kv_pool_tokens / c_bar)``. Above the KV
        ceiling the router's spill condition is unreachable, so the packing rule has to be
        written in terms of this and not of ``pack_cap``.
        """
        lim = max(int(self.pack_cap), 1)
        if self.max_running_per_replica > 0:
            lim = min(lim, int(self.max_running_per_replica))
        return max(lim, 1)

    def _mu(self, b: float) -> float:
        """One replica's request completion rate at occupancy ``b``, from Little's Law
        inverted: ``b = lam*T(b)`` so ``lam = b / (A + B*b)``."""
        if b <= 0:
            return 0.0
        t = self.residence_a_s + self.residence_b_s * b
        return float(b / t) if t > 0 else 0.0

    def service_rate(self, n_total: float, policy: str = "pack") -> float:
        """Pool throughput at in-flight count ``n_total``: the SUM over the occupancy
        vector, not ``k*mu(N/k)``. Monotone increasing in ``n_total``."""
        return float(sum(self._mu(b) for b in
                         occupancy_vector(n_total, self.n_replicas, self.c_eff, policy)))

    def solve_inflight_exact(self, lam_reqs_s: float, policy: str = "pack",
                             tol: float = 1e-9) -> dict:
        """In-flight count as the unique root of ``sum_i mu(b_i(N)) = lam``.

        ``service_rate`` is continuous and increasing on ``[0, n*c_eff]``, so this is a
        bisection with a unique answer -- there is no ``cycle`` status, and there cannot
        be. The dither bands the mean-field cobweb reports are an artifact of iterating
        that map, not a property of the plant.
        """
        if self.residence_b_s <= 0 and self.residence_a_s <= 0:
            return {"status": "unmodelled", "n": float("nan"), "occupancy": []}
        if lam_reqs_s <= 0:
            return {"status": "converged", "n": 0.0,
                    "occupancy": [0.0] * self.n_replicas}
        hi = float(self.n_replicas * self.c_eff)
        if self.service_rate(hi, policy) < lam_reqs_s:
            return {"status": "diverged", "n": float("inf"), "occupancy": []}
        lo = 0.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.service_rate(mid, policy) < lam_reqs_s:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return {"status": "converged", "n": float(hi),
                "occupancy": occupancy_vector(hi, self.n_replicas, self.c_eff, policy)}

    def power_from_rate_exact(self, lam_reqs_s: float, policy: str = "pack") -> dict:
        """Predicted pool power at an offered rate, on the exact occupancy vector."""
        sol = self.solve_inflight_exact(lam_reqs_s, policy)
        if sol["status"] != "converged":
            return {**sol, "power_w": float("nan")}
        occ = sol["occupancy"]
        return {**sol,
                "power_w": float(sum(self.curve.power(int(round(b))) for b in occ)),
                "k": sum(1 for b in occ if b > 0.5)}

    def capacity_exact_reqs_s(self, policy: str = "pack") -> float:
        """Max sustainable rate: the service rate with every replica at ``c_eff``.

        Equals the mean field's ``n * c/(A + B*c)`` when ``c_eff == pack_cap`` -- the two
        forms agree exactly at saturation -- and is lower when the KV ceiling binds.
        """
        return self.service_rate(self.n_replicas * self.c_eff, policy)

    def active_replicas(self, n_total: int, policy: str = "pack") -> int:
        return sum(1 for b in occupancies(n_total, self.n_replicas, self.pack_cap, policy) if b > 0)

    def packing_saving_w(self, n_total: int) -> float:
        """Watts saved by packing instead of spreading at the same in-flight count.

        For a **monotone concave** g this is non-negative by Jensen, and it is the size
        of the routing/packing lever computed with no controller and no measurement.

        For a measured curve that peaks and relaxes it is not sign-guaranteed, and that
        is a real effect rather than a modelling wart: if per-replica power falls past
        its peak, concentrating load past the peak is doubly good -- fewer lit replicas
        *and* less power per lit replica -- while packing to exactly the peak is the
        worst place to stop. ``best_pack_cap_w`` finds where to stop.
        """
        return self.power(n_total, "spread") - self.power(n_total, "pack")

    def best_pack_cap_w(self, n_total: int, candidates=None) -> dict:
        """The pack_cap minimizing pool power at a given in-flight count.

        Power alone always favours a larger cap; the caller intersects this with the
        latency-derived cap from concurrency_sweep, which is what makes ``pack_cap`` an
        SLO-constrained choice rather than a power-only one.
        """
        cands = candidates if candidates is not None else range(1, max(n_total, 1) + 1)
        best = None
        for c in cands:
            occ = occupancies(n_total, self.n_replicas, int(c), "pack")
            pw = float(sum(self.curve.power(b) for b in occ))
            if best is None or pw < best["power_w"]:
                best = {"pack_cap": int(c), "power_w": pw, "occupancy": occ}
        return best or {}

    def solve_inflight(self, lam_reqs_s: float, policy: str = "pack",
                       max_iter: int = 400) -> dict:
        """Solve Little's Law and the packing rule together for the in-flight count.

        N = lam * T(b_bar) with b_bar = N / k, and k -- the number of *active* replicas
        -- itself depends on N through the packing rule. So this is a mutual fixed
        point, not a formula: packing onto fewer replicas raises each survivor's
        occupancy, which lengthens its residence time, which raises N, which can push k
        back up again.

        Three outcomes, and the third is the interesting one:

        ``converged``  a self-consistent (N, k) exists.
        ``diverged``   the offered rate exceeds what k replicas can retire; N -> inf.
        ``cycle``      **no** replica count is self-consistent. The iteration alternates
                       between two states, e.g. k=2 wants N=103 which needs k=3, and
                       k=3 wants N=90 which only needs k=2. This is not a numerical
                       failure to be smoothed away -- it is a load band in which the
                       packing rule has no fixed point and the fleet must oscillate.
                       The cycle's states are returned so a caller can time-average
                       power over them rather than throwing the prediction away.
        """
        if self.residence_b_s <= 0 and self.residence_a_s <= 0:
            return {"status": "unmodelled", "n": float("nan"), "cycle": []}
        n = max(lam_reqs_s * self.residence_a_s, 1.0)
        seen: list[float] = []
        for _ in range(max_iter):
            k = max(self.active_replicas(round(n), policy), 1)
            denom = 1.0 - lam_reqs_s * self.residence_b_s / k
            if denom <= 0:
                return {"status": "diverged", "n": float("inf"), "cycle": []}
            n_new = lam_reqs_s * self.residence_a_s / denom
            if self.max_running_per_replica > 0:
                n_new = min(n_new, self.n_replicas * self.max_running_per_replica)
            if abs(n_new - n) < 1e-6:
                return {"status": "converged", "n": float(n_new), "cycle": [float(n_new)]}
            r = round(n_new, 3)
            if r in seen:
                cycle = seen[seen.index(r):]
                return {"status": "cycle", "n": float(sum(cycle) / len(cycle)),
                        "cycle": [float(x) for x in cycle]}
            seen.append(r)
            n = n_new
        return {"status": "converged", "n": float(n), "cycle": [float(n)]}

    def inflight_from_rate(self, lam_reqs_s: float, policy: str = "pack") -> float:
        """Mean in-flight count; ``inf`` past capacity. In a dither band this is the
        average over the cycle's states, which is what a time-average power sees."""
        return self.solve_inflight(lam_reqs_s, policy)["n"]

    def power_from_rate(self, lam_reqs_s: float, policy: str = "pack") -> dict:
        """Predicted pool power at an offered rate, honest about dither bands.

        In a cycle the plant visits each state, so the time-average power is the mean
        of the per-state powers -- NOT the power at the mean occupancy. Those differ
        whenever g is curved (it is), and the gap between them is the Jensen term that
        this model can locate but cannot size.
        """
        sol = self.solve_inflight(lam_reqs_s, policy)
        if sol["status"] == "diverged":
            return {**sol, "power_w": float("nan"), "power_at_mean_w": float("nan"),
                    "jensen_gap_w": float("nan")}
        if not sol["cycle"]:
            return {**sol, "power_w": float("nan"), "power_at_mean_w": float("nan"),
                    "jensen_gap_w": float("nan")}
        powers = [self.power(round(x), policy) for x in sol["cycle"]]
        mean_p = sum(powers) / len(powers)
        at_mean = self.power(round(sol["n"]), policy)
        return {**sol, "power_w": mean_p, "power_at_mean_w": at_mean,
                "jensen_gap_w": at_mean - mean_p,
                "power_cycle_w": powers}


# ======================================================================================
# Fleet: a large pool and a small pool, with a routing fraction
# ======================================================================================

@dataclass
class FleetModel:
    """Two pools and one knob: ``f``, the share of *flexible* traffic sent to the small
    pool. Non-flexible traffic is pinned to the pool its class requires."""

    large: PoolModel
    small: PoolModel
    flexible_share: float = 1.0      # fraction of arrivals that MAY be rerouted
    # "meanfield" (default) or "exact" (the occupancy-vector solve). The default is
    # deliberate: switching it silently would change every number produced with the mean
    # field. Offline callers pass solver="exact" to opt in; see PoolModel.solve_inflight_exact.
    solver: str = "meanfield"

    def split_rates(self, lam_reqs_s: float, f: float) -> tuple[float, float]:
        flex = lam_reqs_s * self.flexible_share
        fixed = lam_reqs_s - flex
        return fixed + flex * (1.0 - f), flex * f

    def power(self, lam_reqs_s: float, f: float, policy: str = "pack") -> dict:
        lam_l, lam_s = self.split_rates(lam_reqs_s, f)
        if self.solver == "exact":
            sol_l = self.large.power_from_rate_exact(lam_l, policy)
            sol_s = self.small.power_from_rate_exact(lam_s, policy)
            n_l, n_s = sol_l["n"], sol_s["n"]
            return {
                "f": f, "lam_large": lam_l, "lam_small": lam_s,
                "n_large": n_l, "n_small": n_s,
                "k_large": sol_l.get("k", -1), "k_small": sol_s.get("k", -1),
                "power_large_w": sol_l["power_w"], "power_small_w": sol_s["power_w"],
                "power_total_w": sol_l["power_w"] + sol_s["power_w"],
                "status_large": sol_l["status"], "status_small": sol_s["status"],
                # The exact solve has a unique root, so there is no "cannot settle"
                # state to report. Anything it predicts, it predicts as settling.
                "self_consistent": (sol_l["status"] == "converged"
                                    and sol_s["status"] == "converged"),
            }
        sol_l = self.large.power_from_rate(lam_l, policy)
        sol_s = self.small.power_from_rate(lam_s, policy)
        n_l, n_s = sol_l["n"], sol_s["n"]
        return {
            "f": f, "lam_large": lam_l, "lam_small": lam_s,
            "n_large": n_l, "n_small": n_s,
            "k_large": self.large.active_replicas(round(n_l), policy) if np.isfinite(n_l) else -1,
            "k_small": self.small.active_replicas(round(n_s), policy) if np.isfinite(n_s) else -1,
            "power_large_w": sol_l["power_w"], "power_small_w": sol_s["power_w"],
            "power_total_w": sol_l["power_w"] + sol_s["power_w"],
            "status_large": sol_l["status"], "status_small": sol_s["status"],
            # A fraction is "self-consistent" only if BOTH pools settle. Where either
            # cycles, the model is predicting that the plant itself cannot hold still.
            "self_consistent": sol_l["status"] == "converged" and sol_s["status"] == "converged",
        }

    def feasible_fractions_exact(self, lam_reqs_s: float, grid: int = 101,
                                 policy: str = "pack") -> list[float]:
        """Fractions both pools can retire, using the capacity PACKING WILL REACH.

        ``feasible_fractions`` below asks ``lam_pool <= n_replicas * capacity_reqs_s``,
        where ``capacity_reqs_s`` is evaluated at ``pack_cap``. That over-states the pool
        twice over when the KV ceiling binds: the cap is unreachable, and the rate is read
        at a fill level the engine will not hold. This asks the same question at
        ``c_eff`` instead.
        """
        cap_l = self.large.capacity_exact_reqs_s(policy) if self.large.n_replicas else 0.0
        cap_s = self.small.capacity_exact_reqs_s(policy) if self.small.n_replicas else 0.0
        out = []
        for i in range(grid):
            f = i / (grid - 1)
            lam_l, lam_s = self.split_rates(lam_reqs_s, f)
            if (cap_l <= 0 or lam_l <= cap_l) and (cap_s <= 0 or lam_s <= cap_s):
                out.append(f)
        return out

    def feasible_fractions(self, lam_reqs_s: float, grid: int = 101) -> list[float]:
        """Fractions at which BOTH pools stay inside their own measured capacity.

        Outside this band a pool is offered more than it can retire, its queue grows
        without bound, and any power reading taken there describes a diverging system
        rather than an operating point.
        """
        out = []
        for i in range(grid):
            f = i / (grid - 1)
            lam_l, lam_s = self.split_rates(lam_reqs_s, f)
            cap_l = self.large.capacity_reqs_s * self.large.n_replicas
            cap_s = self.small.capacity_reqs_s * self.small.n_replicas
            if (cap_l <= 0 or lam_l <= cap_l) and (cap_s <= 0 or lam_s <= cap_s):
                out.append(f)
        return out

    def reachable_band(self, lam_reqs_s: float, policy: str = "pack") -> dict:
        """The knob's authority at this offered load: the span of total fleet power
        over all feasible routing fractions."""
        fs = self.feasible_fractions(lam_reqs_s)
        if not fs:
            return {"lam_reqs_s": lam_reqs_s, "feasible": False, "band_w": 0.0,
                    "p_min_w": float("nan"), "p_max_w": float("nan"),
                    "f_at_min": float("nan"), "f_at_max": float("nan"), "n_dither": 0}
        rows = [self.power(lam_reqs_s, f, policy) for f in fs]
        ok = [r for r in rows if np.isfinite(r["power_total_w"])]
        if not ok:
            return {"lam_reqs_s": lam_reqs_s, "feasible": False, "band_w": 0.0,
                    "p_min_w": float("nan"), "p_max_w": float("nan"),
                    "f_at_min": float("nan"), "f_at_max": float("nan"), "n_dither": 0}
        lo = min(ok, key=lambda r: r["power_total_w"])
        hi = max(ok, key=lambda r: r["power_total_w"])
        return {
            "lam_reqs_s": lam_reqs_s, "feasible": True,
            "f_lo": fs[0], "f_hi": fs[-1],
            "p_min_w": lo["power_total_w"], "f_at_min": lo["f"],
            "p_max_w": hi["power_total_w"], "f_at_max": hi["f"],
            "band_w": hi["power_total_w"] - lo["power_total_w"],
            # fractions with no self-consistent replica count: the model's prediction of
            # where the plant cannot settle and the power must oscillate.
            "n_dither": sum(1 for r in rows if not r["self_consistent"]),
            "dither_fractions": [r["f"] for r in rows if not r["self_consistent"]],
        }


# ======================================================================================
# persistence
# ======================================================================================

def save_curves(curves: dict[str, PowerCurve], path) -> None:
    with open(path, "w") as f:
        json.dump({k: c.to_dict() for k, c in curves.items()}, f, indent=2)


def load_curves(path) -> dict[str, PowerCurve]:
    with open(path) as f:
        return {k: PowerCurve.from_dict(v) for k, v in json.load(f).items()}


# ======================================================================================
def _selfcheck() -> None:
    rng = np.random.default_rng(0)

    # --- a synthetic replica we know the truth for
    truth = PowerCurve("mm", (30.0,), p_idle_w=110.0, p_busy_w=620.0)
    b = np.array([0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256], float)
    p = truth.power(b) + rng.normal(0, 2.0, size=b.size)

    fit = fit_curve(b, p, p_idle_w=110.0, form="mm")
    assert abs(fit.params[0] - 30.0) < 3.0, fit.params
    assert abs(fit.p_busy_w - 620.0) < 15.0, fit.p_busy_w
    assert fit.fit_stats["r2"] > 0.99, fit.fit_stats

    cmp = fit_all_forms(b, p, p_idle_w=110.0)
    assert cmp["best"] is not None
    # the true generating form must not be beaten badly by any other on held-out data
    assert cmp["forms"]["mm"]["held_out"]["rmse_w"] < 15.0, cmp["forms"]["mm"]

    # --- power is exactly idle at b=0, monotone, and concave
    assert abs(float(truth.power(0)) - 110.0) < 1e-9
    pv = truth.power(np.arange(0, 300))
    assert np.all(np.diff(pv) >= -1e-9), "P(b) must be non-decreasing"
    assert truth.marginal_power(4) > truth.marginal_power(64), "marginal cost must fall"

    # --- packing arithmetic
    assert occupancies(100, 4, 48, "pack") == [48, 48, 4, 0]
    # spread lights the SAME k as pack (ceil(100/48) = 3), then splits evenly over those
    assert occupancies(100, 4, 48, "spread") == [34, 33, 33, 0]
    assert occupancies(70, 4, 20, "pack") == [20, 20, 20, 10]
    assert occupancies(70, 4, 20, "spread") == [18, 18, 17, 17]
    # spread_all is the naive balancer: every replica lit regardless of load
    assert occupancies(100, 4, 48, "spread_all") == [25, 25, 25, 25]
    # same number lit under pack and spread -- that is the point of the comparison
    for n in range(1, 400, 13):
        a = occupancies(n, 5, 32, "pack")
        b = occupancies(n, 5, 32, "spread")
        assert sum(1 for x in a if x > 0) == sum(1 for x in b if x > 0), (n, a, b)
        assert sum(a) == sum(b) == n
    assert occupancies(300, 4, 48, "pack") == [75, 75, 75, 75]   # all capped, spill even
    assert sum(occupancies(137, 5, 32, "pack")) == 137

    pool = PoolModel(truth, n_replicas=4, pack_cap=48, capacity_reqs_s=40.0,
                     residence_a_s=1.2, residence_b_s=0.010)
    assert pool.active_replicas(100) == 3
    assert pool.active_replicas(10) == 1
    # Jensen at FIXED active count: pack and spread light the same k and hold the same
    # total, so for a monotone concave g the equal arrangement maximizes the sum and
    # packing can never cost more. This is a pure concavity result -- lighting every
    # replica (spread_all) would confound arrangement with replica count.
    assert FORMS[truth.form].monotone
    for n in range(0, 400, 7):
        assert pool.packing_saving_w(n) >= -1e-6, (n, pool.packing_saving_w(n))
    assert pool.packing_saving_w(100) > 0.0, pool.packing_saving_w(100)

    # --- the MEASURED 1B curve is not monotone (concurrency_sweep, Llama-3.2-1B, 128/128)
    b_m = np.array([0, 1, 4, 16, 64, 128, 256, 512], float)
    p_m = np.array([122.6, 300.5, 305.1, 321.6, 344.9, 333.2, 325.9, 322.9])
    relax = fit_curve(b_m, p_m, p_idle_w=122.6, form="jump_relax")
    mono = fit_curve(b_m, p_m, p_idle_w=122.6, form="mm")
    assert relax.fit_stats["rmse_w"] < 0.5 * mono.fit_stats["rmse_w"], (
        relax.fit_stats["rmse_w"], mono.fit_stats["rmse_w"])
    peak_b = int(relax.power(np.arange(1, 600)).argmax()) + 1
    assert 20 < peak_b < 200, f"measured curve peaks near b=64, fit says {peak_b}"
    # the wake-up jump dominates: most of the dynamic range is present at b=1
    jump = relax.params[1]
    assert jump > 0.25, f"expected a large wake-up jump, got {jump:.2f}"
    # packing past the peak still wins, because idling whole replicas dominates
    # Pack and spread light the same k, so they COINCIDE whenever the load fills every
    # active replica exactly (N = k*cap); the difference lives in the partial fill.
    pool_m = PoolModel(relax, n_replicas=4, pack_cap=128, capacity_reqs_s=200.0)
    assert pool_m.power(512, "pack") == pool_m.power(512, "spread")     # 4 x 128, exact
    assert pool_m.power(400, "pack") < pool_m.power(400, "spread")      # partial fill
    # and against the naive balancer that lights every replica regardless of load
    assert pool_m.power(400, "pack") < pool_m.power(400, "spread_all")

    # --- Little's Law fixed point: three outcomes, all reachable
    lo = pool.solve_inflight(20.0)
    assert lo["status"] == "converged" and lo["n"] > 0, lo
    assert pool.solve_inflight(4 * 100.0 + 50)["status"] == "diverged"
    # Sweep for a load band where no replica count is self-consistent. The packing rule
    # makes these generic, not pathological: k=2 wants an N that needs k=3, and k=3
    # wants an N that only needs k=2.
    cycles = [r for r in (pool.solve_inflight(x) for x in np.arange(5, 120, 0.5))
              if r["status"] == "cycle"]
    assert cycles, "expected at least one dither band in the load sweep"
    # in a dither band the time-average power differs from power-at-the-mean (Jensen)
    band = pool.power_from_rate(60.0)
    assert band["status"] == "cycle", band["status"]
    assert len(band["cycle"]) >= 2 and np.isfinite(band["power_w"]), band
    # monotone in load, once restricted to loads that actually settle
    conv = [(x, pool.solve_inflight(x)["n"]) for x in (10.0, 20.0, 30.0)
            if pool.solve_inflight(x)["status"] == "converged"]
    assert all(b[1] >= a[1] for a, b in zip(conv, conv[1:])), conv

    # --- fleet band
    small_curve = PowerCurve("mm", (40.0,), p_idle_w=105.0, p_busy_w=330.0)
    small = PoolModel(small_curve, n_replicas=1, pack_cap=64, capacity_reqs_s=90.0,
                      residence_a_s=0.6, residence_b_s=0.004)
    fleet = FleetModel(pool, small, flexible_share=1.0)
    band = fleet.reachable_band(40.0)
    assert band["feasible"] and band["band_w"] > 0, band
    assert band["p_min_w"] < band["p_max_w"]
    # routing to the cheap pool must be the low-power end
    assert band["f_at_min"] > band["f_at_max"], band

    print("power_model selfcheck OK")
    print(f"  recovered mm b50={fit.params[0]:.1f} (truth 30), "
          f"P_busy={fit.p_busy_w:.0f} W (truth 620), R2={fit.fit_stats['r2']:.4f}")
    print(f"  form ranking by held-out RMSE: {', '.join(cmp['ranking_by_heldout_rmse'])}")
    print(f"  N=100 on 4 replicas, cap 48 (both light k=3): "
          f"pack {pool.power(100,'pack'):.0f} W {occupancies(100,4,48,'pack')} vs "
          f"spread {pool.power(100,'spread'):.0f} W {occupancies(100,4,48,'spread')} "
          f"-> {pool.packing_saving_w(100):.0f} W from arrangement alone")
    print(f"  (naive spread_all lights all 4: "
          f"{pool.power(100,'spread_all'):.0f} W {occupancies(100,4,48,'spread_all')})")
    print(f"  measured 1B curve: mm RMSE {mono.fit_stats['rmse_w']:.1f} W vs "
          f"jump_relax {relax.fit_stats['rmse_w']:.1f} W, peak at b={peak_b}, "
          f"wake-up jump {jump:.0%} of the dynamic range")
    print(f"  measured curve, N=400 over 4 replicas at cap 128: pack "
          f"{pool_m.power(400,'pack'):.0f} W vs spread {pool_m.power(400,'spread'):.0f} W "
          f"vs spread_all {pool_m.power(400,'spread_all'):.0f} W")
    print(f"  fleet band at 40 req/s: {band['p_min_w']:.0f}..{band['p_max_w']:.0f} W "
          f"= {band['band_w']:.0f} W  (f_min={band['f_at_min']:.2f}, f_max={band['f_at_max']:.2f})")
    print(f"  dither bands found in a 5..120 req/s sweep: {len(cycles)} loads with no "
          f"self-consistent replica count")
    b60 = pool.power_from_rate(60.0)
    print(f"    e.g. 60 req/s cycles over N={[round(x) for x in b60['cycle']]}, "
          f"time-avg {b60['power_w']:.0f} W vs power-at-mean {b60['power_at_mean_w']:.0f} W "
          f"(Jensen gap {b60['jensen_gap_w']:+.0f} W)")

    # ==================================================================================
    # The EXACT solve: the seven properties, as assertions rather than as a claim.
    # This is where they are enforced, so a future edit that breaks one fails here
    # instead of silently changing every prediction.
    # ==================================================================================
    ex = PoolModel(fit, n_replicas=4, pack_cap=48,
                   residence_a_s=pool.residence_a_s, residence_b_s=pool.residence_b_s,
                   max_running_per_replica=0)

    # (0) the continuous vector conserves N, lights the same replicas as the integer
    # version, and agrees with it up to the rounding the integer version must do
    # (occupancies(49, 4, 48, "spread") is [25, 24, 0, 0]; the continuous form is
    # [24.5, 24.5, 0, 0] -- the same arrangement, not the same list).
    for pol in ("pack", "spread", "spread_all"):
        for nt in (0, 1, 7, 48, 49, 100, 192, 300):
            v = occupancy_vector(nt, 4, 48, pol)
            iv = [float(x) for x in occupancies(nt, 4, 48, pol)]
            assert abs(sum(v) - nt) < 1e-9, (pol, nt, v)
            # Same lit set, once there is at least one request per lit replica to give.
            # Below that the continuous form is a fiction: spread_all puts 0.25 on each of
            # four replicas for N=1, where the integer form must pick one.
            if nt >= 4:
                assert sum(1 for x in v if x > 0) == sum(1 for x in iv if x > 0), \
                    (pol, nt, v, iv)
            assert all(abs(a - b) <= 1.0 for a, b in zip(v, iv)), (pol, nt, v, iv)

    # (1) the mean field IS k*mu(N/k) = lam -- the two forms are the same question
    for k in (1, 2, 3, 4):
        lam = 0.6 * k / ex.residence_b_s
        n_mf = lam * ex.residence_a_s / (1 - lam * ex.residence_b_s / k)
        assert abs(k * ex._mu(n_mf / k) - lam) < 1e-9, (k, lam)

    # (2)+(3) mu strictly increasing and strictly concave
    bs = np.linspace(1e-6, 400, 4000)
    mus = np.array([ex._mu(b) for b in bs])
    assert np.all(np.diff(mus) > 0), "mu must be strictly increasing"
    assert np.all(np.diff(mus, 2) < 1e-18), "mu must be strictly concave"

    # (4) service_rate monotone in N  =>  the root is unique
    ns = np.linspace(0, 4 * ex.c_eff, 2000)
    sr = np.array([ex.service_rate(x) for x in ns])
    assert np.all(np.diff(sr) >= -1e-12), "service_rate must be monotone in N"
    lam_mid = 0.5 * sr[-1]
    assert int(np.sum(np.diff(np.sign(sr - lam_mid)) != 0)) == 1, "root must be unique"

    # (5) JENSEN: the mean field over-states pool throughput at equal (N, k)
    for nt in (10.0, 45.0, 60.0, 100.0, 150.0):
        v = occupancy_vector(nt, 4, ex.c_eff, "pack")
        k = sum(1 for b in v if b > 0)
        assert k * ex._mu(nt / k) >= sum(ex._mu(b) for b in v) - 1e-12, nt

    # (6) the two forms agree EXACTLY at saturation, so no capacity claim moves
    sat = 4 * ex.c_eff
    assert abs(4 * ex._mu(sat / 4) - ex.service_rate(sat)) < 1e-9

    # (7) bisection residual, and NO cycle status anywhere the cobweb found one
    worst = 0.0
    for lam in np.arange(5.0, 120.0, 0.5):
        sol = ex.solve_inflight_exact(float(lam))
        assert sol["status"] in ("converged", "diverged"), sol["status"]
        if sol["status"] == "converged":
            worst = max(worst, abs(ex.service_rate(sol["n"]) - lam))
    assert worst < 1e-6, f"bisection residual {worst}"

    # c_eff honours the KV ceiling, and capacity falls when it binds
    kv_bound = PoolModel(fit, n_replicas=4, pack_cap=256,
                         residence_a_s=ex.residence_a_s, residence_b_s=ex.residence_b_s,
                         max_running_per_replica=68)
    assert kv_bound.c_eff == 68, kv_bound.c_eff
    assert PoolModel(fit, 4, 48, residence_a_s=1.0, residence_b_s=0.01).c_eff == 48
    # The KV ceiling must REDUCE capacity relative to the SAME pool without it -- not
    # relative to a pool with a different pack_cap, which is a different machine.
    kv_free = PoolModel(fit, n_replicas=4, pack_cap=256,
                        residence_a_s=ex.residence_a_s, residence_b_s=ex.residence_b_s,
                        max_running_per_replica=0)
    assert kv_free.c_eff == 256, kv_free.c_eff
    assert kv_bound.capacity_exact_reqs_s() < kv_free.capacity_exact_reqs_s(), (
        kv_bound.capacity_exact_reqs_s(), kv_free.capacity_exact_reqs_s())

    n_cyc_exact = sum(1 for lam in np.arange(5, 120, 0.5)
                      if ex.solve_inflight_exact(float(lam))["status"] == "cycle")
    assert n_cyc_exact == 0
    print(f"  exact solve: 7/7 properties hold; bisection residual {worst:.1e}; "
          f"{len(cycles)} mean-field dither bands -> {n_cyc_exact} under the exact solve")
    # pool_m carries no residence line (it exists for the packing arithmetic above), so
    # it correctly reports `unmodelled`; the fleet demo needs pools that have one.
    fl_ex = FleetModel(large=ex, small=ex, flexible_share=1.0, solver="exact")
    p_ex = fl_ex.power(40.0, 0.5, "pack")
    assert np.isfinite(p_ex["power_total_w"]) and p_ex["self_consistent"], p_ex
    print(f"  FleetModel(solver='exact') at 40 req/s, f=0.5: "
          f"{p_ex['power_total_w']:.0f} W, N={p_ex['n_large']:.0f}+{p_ex['n_small']:.0f}, "
          f"k_large={p_ex['k_large']}  (default solver unchanged: 'meanfield')")


if __name__ == "__main__":  # python -m lib.<module>
    _selfcheck()
