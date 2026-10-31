#!/usr/bin/env python
"""Measured per-replica capacity for a fleet run's cells -- for figures and markers.

    python analyze/capacity_lookup.py --selfcheck

Two rules, each guarding against a way this number can go wrong:

- **Hardware comes from where the run lives**, never from the machine plotting it.
  `results/` is the H100 baseline and `results/<slug>/` that card. `feasible.load_measured`
  keys on the LIVE NVML slug, which is right inside a run and wrong here: on a GPU-less
  analysis node it is the baseline, and would refuse every RTX capacity for an RTX figure.
- **A capacity.json without `stability_rule` is re-judged from its own probes.** Such a
  file was judged with an absolute 0.5/s stability floor, which can pass probes whose
  queue is growing. The rule applied instead is capacity_bisection's `is_stable` / `monotone_capacity`,
  re-implemented here because importing capacity_bisection pulls in the engine: stable iff
  slope < max(1 % of rate, 2 SE), and the capacity is the highest stable probe below the
  lowest diverging one.

Later bisections of the same cell win, as in `load_measured`. Never an analytic capacity.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

BASELINE_SLUG = "h100-80gb"
REJUDGED = "capacity_bisection, re-judged without the 0.5/s floor"
AS_RECORDED = "capacity_bisection"


def rejudge(p: dict) -> bool:
    """capacity_bisection's stability rule (`is_stable`), applied to one recorded probe."""
    s, se, r = p.get("waiting_slope_per_s"), p.get("waiting_slope_se"), p["rate_reqs_s"]
    if s is None or not math.isfinite(s):
        return False
    sig = 2.0 * se if (se is not None and math.isfinite(se)) else 0.0
    return s < max(0.01 * r, sig)


def capacity_of(j: dict) -> tuple[float | None, str]:
    """(capacity req/s, provenance) for one capacity.json."""
    if j.get("stability_rule"):
        return j.get("capacity_reqs_s"), AS_RECORDED
    probes = j.get("probes") or []
    if not probes:
        return j.get("capacity_reqs_s"), AS_RECORDED
    bad = [p["rate_reqs_s"] for p in probes if not rejudge(p)]
    first_bad = min(bad, default=math.inf)
    ok = [p["rate_reqs_s"] for p in probes if rejudge(p) and p["rate_reqs_s"] < first_bad]
    return (max(ok) if ok else None), REJUDGED


def hardware_of(results_root: Path) -> str:
    """`results/` -> the baseline; `results/<slug>/` -> that slug."""
    return BASELINE_SLUG if results_root.name == "results" else results_root.name


def lookup(results_root: Path, model: str, workload: str, cap: int) -> dict | None:
    """Latest bisection of (model, workload, max_num_seqs) on this root's hardware."""
    root = Path(results_root) / "capacity_bisection"
    if not root.is_dir():
        return None
    hw, best = hardware_of(Path(results_root)), None
    for d in sorted(root.iterdir()):
        f = d / "capacity.json"
        if not f.exists() or "smoke" in d.name.lower():   # shakeout runs are never a measurement
            continue
        try:
            j = json.loads(f.read_text())
        except ValueError:
            continue
        if ((j.get("hardware") or {}).get("slug") or BASELINE_SLUG) != hw:
            continue
        if (j.get("model_slug") == model
                and (j.get("workload") or {}).get("name") == workload
                and int(j.get("max_num_seqs", -1)) == int(cap)):
            c, rule = capacity_of(j)
            if c is not None:
                best = {"reqs_s": float(c), "rule": rule, "source": d.name}   # later wins
    return best


def fleet_capacity(run: Path) -> dict | None:
    """Measured fleet capacity for a fleet run (fleet_routing_sweep), or None if a cell was never bisected."""
    # RESOLVE FIRST: the hardware is read off the results root's name, and a relative run
    # path makes run.parent.parent '.', whose name is '' -- silently matching nothing.
    run = Path(run).resolve()
    a = json.loads((run / "summary.json").read_text()).get("args") or {}
    try:
        nL, nS = int(a["n_large"]), int(a.get("n_small") or 0)
        wl = a["workload"]
        L = lookup(run.parent.parent, a["large"], wl, a["concurrency_large"])
        S = (lookup(run.parent.parent, a["small"], wl, a["concurrency_small"])
             if nS else {"reqs_s": 0.0, "rule": AS_RECORDED, "source": None})
    except (KeyError, TypeError, ValueError):
        return None
    if L is None or S is None:
        return None
    split = [float(x) for x in str(a.get("class_split", "0,0,1")).split(",")]
    rules = {L["rule"], S["rule"]} if nS else {L["rule"]}
    return {"n_large": nL, "n_small": nS, "large": a["large"], "small": a.get("small"),
            "workload": wl, "c_large": L["reqs_s"], "c_small": S["reqs_s"],
            "fleet_reqs_s": nL * L["reqs_s"] + nS * S["reqs_s"],
            "rule": REJUDGED if REJUDGED in rules else AS_RECORDED,
            "sources": [L["source"], S["source"]], "class_split": split}


def _selfcheck() -> None:
    res = Path(__file__).resolve().parent.parent / "results"
    assert hardware_of(res) == BASELINE_SLUG
    assert hardware_of(res / "rtx-8000-48gb") == "rtx-8000-48gb"
    # A floor-judged RTX file re-judges to its highest stable probe below the first
    # real divergence: 3.375 was called stable while its queue grew at +0.40 /s.
    f = res / "rtx-8000-48gb/capacity_bisection/20260922_102259__rtx_qwen3-8b__sharegpt__cap64/capacity.json"
    if f.exists():
        c, rule = capacity_of(json.loads(f.read_text()))
        assert (c, rule) == (3.0, REJUDGED), (c, rule)
    # A file that records its `stability_rule` is taken as recorded.
    j = {"stability_rule": "x", "capacity_reqs_s": 4.25,
         "probes": [{"rate_reqs_s": 4.25, "stable": True, "waiting_slope_per_s": 9.0,
                     "waiting_slope_se": 0.1}]}
    assert capacity_of(j) == (4.25, AS_RECORDED)
    # The H100 lookup must not see RTX files, nor the reverse.
    assert lookup(res, "qwen3-8b", "sharegpt", 64)["reqs_s"] != 3.0
    runs = sorted((res / "rtx-8000-48gb/fleet_routing_sweep").glob("*5L3S*_fine"))
    if runs:
        fc = fleet_capacity(runs[0])
        # The RTX run must be priced with RTX bisections -- the latest of each cell, as the
        # lookup selects them -- so the check compares against the lookup and only asserts
        # the values are RTX-sized (the H100's are ~20 and ~40 req/s).
        rtx = res / "rtx-8000-48gb"
        cl = (lookup(rtx, "qwen3-8b", "sharegpt", 64) or {}).get("reqs_s")
        cs = (lookup(rtx, "qwen3-4b", "sharegpt", 128) or {}).get("reqs_s")
        assert cl and cs and cl < 5.0 and cs < 10.0, (cl, cs)
        assert fc and abs(fc["fleet_reqs_s"] - (5 * cl + 3 * cs)) < 1e-9, (fc, cl, cs)
    # A RELATIVE run path must find the same capacity ('.'.name == '' would match no hardware).
    h100 = sorted((res / "fleet_routing_sweep").glob("*3L1S__sharegpt__poisson__pack*"))
    if h100:
        import os
        here = os.getcwd()
        try:
            os.chdir(h100[0].parent.parent)
            rel = fleet_capacity(Path(h100[0].parent.name) / h100[0].name)
        finally:
            os.chdir(here)
        assert rel and rel == fleet_capacity(h100[0]), rel
    print("capacity_lookup selfcheck OK: hardware from the run's path, floor-judged files "
          "re-judged, files with a stability_rule as recorded, no cross-hardware match")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
