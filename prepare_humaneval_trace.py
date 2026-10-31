#!/usr/bin/env python
"""Build the HumanEval length trace. Login-node safe (tokenizer only, no GPU).

HumanEval (Chen et al., 2021; ``openai/openai_humaneval``) is 164 hand-written Python
problems. Each row's *prompt* is the function signature plus docstring (with any imports
and helper code), and its *canonical_solution* is the reference function body. Like
ShareGPT, the dataset carries no token counts -- lengths are a property of a tokenizer --
so this script derives them exactly as ``prepare_traces.py`` does for ShareGPT: one
reference tokenizer (Llama-3.1's 128k vocabulary), ``add_special_tokens=False``.

The output length is the canonical solution's token count: the length of a correct
completion that stops at the end of the function. Served models that append tests or
explanations produce more; the suite replays lengths, not text, so what matters is that
the (prompt, output) distribution is the benchmark's own. It is a short-prompt,
short-output code workload (prompts ~130 tokens, outputs ~60 tokens), the opposite corner
from arxiv_summarization and far shorter than splitwise_code's prompts.

    HF_HUB_OFFLINE=0 python prepare_humaneval_trace.py

The CSV (``data/traces/humaneval_llama31.csv``) is small enough to commit;
``lib/workloads.py`` registers it as the ``humaneval`` workload.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import env  # noqa: F401  (bootstraps HF cache env vars at import)
from lib.env import DERIVED_TRACES_DIR

HUMANEVAL_REPO = "openai/openai_humaneval"
HUMANEVAL_FILE = "openai_humaneval/test-00000-of-00001.parquet"
REFERENCE_TOKENIZER = "meta-llama/Llama-3.1-8B"


def fetch_humaneval() -> Path:
    from huggingface_hub import hf_hub_download

    print(f"fetching {HUMANEVAL_REPO}/{HUMANEVAL_FILE} (~80 KB, cached after first run)...")
    return Path(hf_hub_download(HUMANEVAL_REPO, HUMANEVAL_FILE, repo_type="dataset"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default=REFERENCE_TOKENIZER)
    ap.add_argument("--out-dir", type=Path, default=DERIVED_TRACES_DIR)
    ap.add_argument("--force", action="store_true", help="rebuild even if the CSV exists")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "humaneval_llama31.csv"
    if out_csv.exists() and not args.force:
        print(f"{out_csv} exists; pass --force to rebuild")
        return

    import pandas as pd
    from transformers import AutoTokenizer

    df = pd.read_parquet(fetch_humaneval())
    need = {"task_id", "prompt", "canonical_solution"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"unexpected HumanEval schema, missing {sorted(missing)}; have {list(df.columns)}")
    print(f"  {len(df):,} problems")

    print(f"loading tokenizer {args.tokenizer}...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def lengths(texts):
        enc = tok(list(texts), add_special_tokens=False)["input_ids"]
        return [len(x) for x in enc]

    out = pd.DataFrame({
        "task_id": df["task_id"],
        "num_prefill_tokens": lengths(df["prompt"]),
        "num_decode_tokens": lengths(df["canonical_solution"]),
    })
    out["num_total_tokens"] = out.num_prefill_tokens + out.num_decode_tokens
    bad = (out.num_prefill_tokens <= 0) | (out.num_decode_tokens <= 0)
    if bad.any():
        print(f"  dropping {int(bad.sum())} rows with an empty prompt or solution")
        out = out[~bad]
    out.to_csv(out_csv, index=False)

    print(f"\nwrote {out_csv}")
    for col in ("num_prefill_tokens", "num_decode_tokens"):
        d = out[col]
        print(f"  {col:19s} n={len(d):4d}  mean {d.mean():7.1f}  std {d.std(ddof=0):6.1f}  "
              f"min {d.min():4d}  p50 {d.median():5.0f}  p99 {d.quantile(0.99):6.0f}  max {d.max():4d}")


if __name__ == "__main__":
    main()
