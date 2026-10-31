#!/usr/bin/env python
"""kv_ceiling_sweep's KV-ceiling grid, placed from a KV pool this model has MEASURED on this hardware.

    python analyze/kv_ceiling_levels.py \
        --model llama31-70b --workload fixed_1024_1024        # -> "levels:..." max_num_seqs
    python analyze/kv_ceiling_levels.py --selfcheck

The rule is the one every hand-written KV-ceiling grid follows:
the ceiling C = round(K / l_bar), with K the engine's KV pool in tokens and l_bar the
workload's decode-weighted mean context; the levels are C x (0.4 0.6 0.8 0.9 1.0 1.1 1.25
1.5 2.0) and max_num_seqs is 2.2 C, all rounded half up. K is read from this model's own
runs -- concurrency_sweep summaries and capacity_bisection capacity files under this hardware's results root
(results/h100-80gb-tp4/ for a four-GPU instance) -- never computed from memory arithmetic.
Prints one line, "<levels colon-separated> <max_num_seqs>", for the launcher to read;
exits 1 when no measured pool exists.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FACTORS = (0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)
MAX_FACTOR = 2.2


def _half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def levels(kv_pool_tokens: float, l_bar: float) -> tuple[list[int], int]:
    c = _half_up(kv_pool_tokens / l_bar)
    return [_half_up(f * c) for f in FACTORS], _half_up(MAX_FACTOR * c)


def measured_pool(root: Path, model: str) -> tuple[int | None, list[str]]:
    """Newest measured KV pool (tokens) for ``model`` under ``root``, and where it came from."""
    found: list[tuple[str, int, str]] = []
    for exp, fname, key in (("concurrency_sweep", "summary.json", "meta"),
                            ("capacity_bisection", "capacity.json", None)):
        base = root / exp
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            f = d / fname
            if not f.exists():
                continue
            try:
                j = json.loads(f.read_text())
            except ValueError:
                continue
            rec = j.get(key, {}) if key else j
            if rec.get("model_slug") != model:
                continue
            tok = (rec.get("kv") or {}).get("kv_pool_tokens")
            if tok:
                found.append((d.name, int(tok), str(d)))
    if not found:
        return None, []
    found.sort()                      # run dirs start with a timestamp: newest last
    return found[-1][1], [f"{n}: {t:,}" for n, t, _ in found]


def _selfcheck() -> None:
    from lib import workloads
    # The grids written by hand for Qwen3-14B and Mistral-24B, from their measured pools.
    hand_grids = {
        (282848, "arxiv_summarization"): ("48:72:96:108:120:132:150:180:240", 264),
        (282848, "fixed_1024_1024"): ("74:110:147:166:184:202:230:276:368", 405),
        (282848, "fixed_2048_2048"): ("37:55:74:83:92:101:115:138:184", 202),
        (282848, "fixed_6144_1024"): ("17:25:34:38:42:46:53:63:84", 92),
        (174816, "arxiv_summarization"): ("30:44:59:67:74:81:93:111:148", 163),
        (174816, "fixed_1024_1024"): ("46:68:91:103:114:125:143:171:228", 251),
        (174816, "fixed_2048_2048"): ("23:34:46:51:57:63:71:86:114", 125),
        (174816, "fixed_6144_1024"): ("10:16:21:23:26:29:33:39:52", 57),
    }
    for (k, wl), (want, want_max) in hand_grids.items():
        lv, mx = levels(k, workloads.get(wl).decode_weighted_context)
        got = ":".join(map(str, lv))
        assert (got, mx) == (want, want_max), (k, wl, got, mx, want, want_max)
    print(f"kv_ceiling_levels selfcheck OK: reproduces all {len(hand_grids)} hand-written grids")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--workload")
    ap.add_argument("--kv-pool", type=int, default=None,
                    help="use this pool instead of looking one up (tokens)")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
        return 0
    if not args.model or not args.workload:
        ap.error("--model and --workload are required")
    from lib import env, hardware, models, workloads
    spec = models.get(args.model)
    hardware.set_instance(spec.tensor_parallel_size)
    root = env.results_root()
    k, seen = (args.kv_pool, ["--kv-pool"]) if args.kv_pool else measured_pool(root, spec.slug)
    if not k:
        print(f"no measured KV pool for {spec.slug} under {root}", file=sys.stderr)
        return 1
    wl = workloads.get(args.workload)
    lv, mx = levels(k, wl.decode_weighted_context)
    print(f"  K = {k:,} tokens (from {seen[-1]}; {len(seen)} run(s)), l_bar "
          f"{wl.decode_weighted_context:.1f} -> C = {lv[FACTORS.index(1.0)]}", file=sys.stderr)
    print(f"{':'.join(map(str, lv))} {mx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
