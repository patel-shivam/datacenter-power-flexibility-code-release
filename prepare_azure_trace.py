#!/usr/bin/env python
"""Build the Azure LLM inference trace 2024 length traces. Login-node safe (no GPU, no tokenizer).

The Azure trace (Microsoft, AzurePublicDataset, CC-BY-4.0; released with DynamoLLM, HPCA 2025)
is two week-long CSVs, one row per request: ``TIMESTAMP`` (UTC), ``ContextTokens`` (prompt) and
``GeneratedTokens`` (output) -- nothing about the model, the GPUs or the user. The datacenter
simulation already uses its *timing* (``analyze/fetch_datacenter_inputs.py``); this script takes
its *lengths*, so a GPU sweep can measure a replica on the requests that timing belongs to:

    conversation   12-18 May 2024, 27.3 M requests, prompt / output mean 1,632 / 106 tokens
    code           10-16 May 2024, 16.8 M requests, prompt / output mean 2,511 / 23 tokens

The counts are the production tokenizer's, which is not published. There is no text to
re-tokenize, so they are replayed as token counts on every model, as the splitwise traces
are -- a ~10-20 % tokenizer difference in the prompt lengths is not modelled.

Each output is a uniform random sample of ``--rows`` requests over the whole week (seed 0),
written as ``num_prefill_tokens,num_decode_tokens,num_total_tokens`` like
``sharegpt_llama31.csv``. Clipping is the workload's job (``lib/workloads.py``), not this
file's, so the CSV keeps the trace's own lengths.

    python prepare_azure_trace.py

The raw CSVs come from ``analyze/fetch_datacenter_inputs.py --only azure``, which caches them
under ``.cache/dc_public_inputs/`` and records their URL, size and SHA-256 in ``data/datacenter/SOURCES.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.env import DERIVED_TRACES_DIR  # noqa: E402

CACHE = Path(__file__).resolve().parent / ".cache" / "dc_public_inputs"
KINDS = ("conv", "code")


def build(kind: str, rows: int, seed: int) -> Path:
    raw = CACHE / f"azure_{kind}_1week" / f"AzureLLMInferenceTrace_{kind}_1week.csv"
    if not raw.exists():
        raise SystemExit(f"{raw} missing: run analyze/fetch_datacenter_inputs.py --only azure first")
    d = pd.read_csv(raw, usecols=["ContextTokens", "GeneratedTokens"],
                    dtype={"ContextTokens": np.int32, "GeneratedTokens": np.int32})
    idx = np.sort(np.random.default_rng(seed).choice(len(d), size=rows, replace=False))
    s = d.iloc[idx]
    out = pd.DataFrame({"num_prefill_tokens": s.ContextTokens.to_numpy(),
                        "num_decode_tokens": s.GeneratedTokens.to_numpy()})
    out["num_total_tokens"] = out.num_prefill_tokens + out.num_decode_tokens
    path = DERIVED_TRACES_DIR / f"azure_{kind}_2024.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    full_p, full_d = d.ContextTokens.to_numpy(), d.GeneratedTokens.to_numpy()
    print(f"{kind}: {len(d):,} requests -> {rows:,} rows in {path.name}; "
          f"prompt mean {out.num_prefill_tokens.mean():.0f} (trace {full_p.mean():.0f}), "
          f"output mean {out.num_decode_tokens.mean():.1f} (trace {full_d.mean():.1f})")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kinds", default=",".join(KINDS))
    a = ap.parse_args()
    for kind in a.kinds.split(","):
        build(kind.strip(), a.rows, a.seed)


if __name__ == "__main__":
    main()
