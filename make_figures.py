#!/usr/bin/env python
"""Regenerate every figure of the paper from the result tables in results/ (no GPU, ~1 minute).

    python make_figures.py                       # all figures -> figures/main/ and figures/appendix/
    python make_figures.py --only single_gpu,fleet,datacenter
    python make_figures.py --out /some/dir       # write elsewhere
    python make_figures.py --list                # the figure map, without drawing

Each figure is written as PDF and PNG. The figures read only the CSV tables under results/
(see README.md for which script produces which table from which experiment); nothing here
touches the raw runs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_figures import common  # noqa: E402

GROUPS = ("single_gpu", "fleet", "datacenter")

# (group, placement, file name, what it shows)
FIGURE_MAP = [
    ("single_gpu", "main", "single_gpu_power_vs_resident_requests_h100-80gb",
     "per-replica power P(b) against resident requests, measured levels and the fitted profile, four core models x six tasks"),
    ("single_gpu", "main", "single_gpu_step_time_vs_resident_requests_h100-80gb",
     "step time tau(b) = t0 + t1 b against measured mean TBT, and t1 by task"),
    ("single_gpu", "main", "single_gpu_generation_time_vs_resident_requests_h100-80gb",
     "request residence T(b) = s tau(b) against the measured mean request residence time"),
    ("single_gpu", "main", "single_gpu_resident_requests_vs_arrival_rate_h100-80gb",
     "occupancy fixed point b(lambda) against the arrival rate, with the Poisson phases and the measured capacity"),
    ("single_gpu", "main", "single_gpu_completion_rate_vs_resident_requests_h100-80gb",
     "request throughput mu(b) = b / T(b) against the measured closed-loop levels"),
    ("single_gpu", "appendix", "single_gpu_parity_all_gpus",
     "five predicted quantities, measured against predicted, on H100, RTX 5090, RTX 8000 and V100"),
    ("fleet", "main", "fleet_feasible_window_and_power_band_4gpu_pairs",
     "feasible routing window and fleet-power band against load, model vs measured, two 4-GPU pairs on ShareGPT"),
    ("fleet", "main", "fleet_power_vs_routing_fraction_llama_sharegpt",
     "fleet power against the routing fraction at every load, and the holdable range against load (Llama pair, ShareGPT)"),
    ("fleet", "appendix", "fleet_feasible_window_and_power_band_grid",
     "the feasible window and power band for the five other fleets with a full load sweep"),
    ("fleet", "appendix", "fleet_power_vs_routing_fraction_grid",
     "fleet power against the routing fraction for the same five fleets"),
    ("datacenter", "main", "datacenter_week_flexibility_8b_1b_sharegpt",
     "a week of real traffic at the reference site: facility power, the floor routing reaches, duration curve, by hour"),
    ("datacenter", "main", "datacenter_week_flexibility_70b_24b_azure_chat", "the same for the 70B -> 24B site at Azure chat request lengths"),
    ("datacenter", "main", "datacenter_week_flexibility_70b_24b_azure_code", "the same for the 70B -> 24B site at Azure code request lengths"),
    ("datacenter", "main", "datacenter_event_day_8b_1b_sharegpt",
     "a 4-hour demand-response event on the busiest day, with its first and last minute at 1 s resolution"),
    ("datacenter", "main", "datacenter_event_day_70b_24b_azure_chat", "the same for the 70B -> 24B site at Azure chat request lengths"),
    ("datacenter", "main", "datacenter_event_day_70b_24b_azure_code", "the same for the 70B -> 24B site at Azure code request lengths"),
    ("datacenter", "main", "datacenter_gpus_leaving_8b_1b_sharegpt",
     "power reduction as inference GPUs leave, and a day with 40 % of the GPUs training at night"),
    ("datacenter", "main", "datacenter_gpus_leaving_70b_24b_azure_chat", "the same for the 70B -> 24B site at Azure chat request lengths"),
    ("datacenter", "main", "datacenter_gpus_leaving_70b_24b_azure_code", "the same for the 70B -> 24B site at Azure code request lengths"),
    ("datacenter", "appendix", "grid_programs_response_timescale",
     "what grid programs require (response time, duration) against the measured routing drop"),
]


def print_map():
    w = max(len(f) for _, _, f, _ in FIGURE_MAP)
    for group, placement, name, what in FIGURE_MAP:
        print(f"{placement:9s} {group:11s} {name:{w}s}  {what}")


def run(groups):
    common.apply_style()
    t = time.time()
    if "single_gpu" in groups:
        from paper_figures import single_gpu
        print("== single-GPU model figures")
        T = common.load_single_gpu()
        for f in single_gpu.MAIN + single_gpu.APPENDIX:
            f(T)
    if "fleet" in groups:
        from paper_figures import fleet_routing
        print("== fleet routing figures")
        F = common.load_fleet()
        for f in fleet_routing.MAIN + fleet_routing.APPENDIX:
            f(F)
    if "datacenter" in groups:
        from paper_figures import datacenter
        print("== datacenter-scale figures")
        for f in datacenter.MAIN + datacenter.APPENDIX:
            f()
    names = {w["name"] for w in common.WRITTEN}
    expected = {f for g, _, f, _ in FIGURE_MAP if g in groups}
    missing = expected - names
    print(f"\n{len(common.WRITTEN)} figures written to {common.FIG_ROOT} in {time.time() - t:.0f} s")
    if missing:
        raise SystemExit(f"expected figures not written: {sorted(missing)}")
    extra = names - expected
    if extra:
        raise SystemExit(f"figures written but not in FIGURE_MAP: {sorted(extra)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=",".join(GROUPS), help=f"comma-separated subset of {GROUPS}")
    ap.add_argument("--out", type=Path, default=None, help="output root (default: figures/ beside this script)")
    ap.add_argument("--list", action="store_true", help="print the figure map and exit")
    args = ap.parse_args()
    if args.list:
        print_map()
        return
    groups = [g.strip() for g in args.only.split(",") if g.strip()]
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        raise SystemExit(f"unknown group(s) {bad}; choose from {GROUPS}")
    if args.out:
        common.set_output(args.out)
    run(groups)


if __name__ == "__main__":
    main()
