#!/usr/bin/env python
"""Re-fit results/power_curves.json from the exact closed-loop sweeps it was fitted from, and compare.

    python refit_power_curves.py [--out <dir>]

``results/power_curves_report.json`` records, per curve, the sweep directory it was fitted from.
This script passes those directories, in the report's order, to ``fit_power_model.py``, writes
``power_curves.json`` and ``power_curves_report.json`` into ``--out`` (default: a temporary
directory) and reports whether every curve and report entry is identical to the shipped one.

Why not simply refit on every sweep under results/concurrency_sweep? That gives 66 curves (a
later sweep of the same (model, workload) cell replaces the earlier one), and the 36 cells the
fleet model and the datacenter simulation read differ wherever a cell was re-swept later with
longer windows or a KV guard. The fleet model keeps the calibration it was validated with, so
the fitted file ships beside the raw sweeps and this script shows it is reproducible.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Curves the fitter does not reproduce from their recorded sweep, and why. None of them is
# used by a figure in the paper.
KNOWN_EXCEPTIONS = {
    "llama31-8b__fixed_6144_1024":
        "the shipped curve includes four KV-pinned levels of this 6144-token-prompt sweep "
        "(c = 96..256) that retired no request; fit_power_model.py drops such levels, which leaves "
        "7 usable levels, below --min-levels 8, so the refit skips the cell. The workload is an "
        "ablation; no figure in the paper uses it.",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="where to write the refitted files")
    args = ap.parse_args()
    out = args.out or Path(tempfile.mkdtemp(prefix="power_curves_"))
    out.mkdir(parents=True, exist_ok=True)

    report = json.loads((ROOT / "results" / "power_curves_report.json").read_text())
    runs = [v["run"] for v in report.values() if isinstance(v, dict) and "run" in v]
    missing = [r for r in runs if not (ROOT / r / "levels.csv").exists()]
    if missing:
        raise SystemExit(f"sweeps recorded in the report but not present: {missing}")
    print(f"refitting {len(runs)} curves from the sweeps the report records ...")
    cmd = [sys.executable, str(ROOT / "fit_power_model.py"), *runs, "--out", str(out / "power_curves.json")]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit("fit_power_model.py failed")

    a = json.loads((ROOT / "results" / "power_curves.json").read_text())
    b = json.loads((out / "power_curves.json").read_text())
    ra = report
    rb = json.loads((out / "power_curves_report.json").read_text())
    same_curves = [k for k in a if a[k] == b.get(k)]
    same_report = [k for k in ra if ra[k] == rb.get(k)]
    print(f"curves identical: {len(same_curves)} / {len(a)} (refit wrote {len(b)})")
    print(f"report entries identical: {len(same_report)} / {len(ra)}")
    unexplained = []
    for k in a:
        if a[k] != b.get(k) or ra[k] != rb.get(k):
            why = KNOWN_EXCEPTIONS.get(k)
            print(f"  differs: {k}" + (f" -- known: {why}" if why else ""))
            if not why:
                unexplained.append(k)
    extra = sorted(set(b) - set(a))
    if extra:
        print(f"  curves the refit wrote that the shipped file lacks: {extra}")
        unexplained += extra
    print(f"refitted files: {out}")
    if unexplained:
        raise SystemExit(f"the refit does not reproduce results/power_curves.json: {unexplained}")
    n_exc = sum(1 for k in a if a[k] != b.get(k))
    print(f"results/power_curves.json reproduced{'' if not n_exc else f' except the {n_exc} documented cell(s)'}")


if __name__ == "__main__":
    main()
