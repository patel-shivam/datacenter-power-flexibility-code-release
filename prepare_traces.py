#!/usr/bin/env python
"""Build the ShareGPT length trace. Login-node safe (tokenizer only, no GPU).

ShareGPT ships as raw conversation text -- it has **no** input/output length columns.
Lengths are a property of a *tokenizer*, so they have to be derived, and every paper
that reports "ShareGPT" has silently made the same three choices this script makes
explicit:

  1. **Which turns.** The first human turn is the prompt, the first assistant turn is
     the output. Multi-turn context is discarded. (This is what vLLM's
     ``benchmark_serving.py`` does, so it is what the numbers in the literature mean.)
  2. **Which tokenizer.** One reference tokenizer for all nine models -- see
     ``workloads.py`` for why replaying one length distribution everywhere is what
     makes the cross-model comparison valid. Llama-3.1's 128k vocab is the reference.
  3. **Which filter.** The standard filter drops degenerate and over-long rows:
     ``prompt_len < 4 or output_len < 4``, ``prompt_len > 1024``, or
     ``prompt_len + output_len > 2048``.

Both the filtered and unfiltered distributions are written. The filtered one is the
default ``sharegpt`` workload because it is what "ShareGPT" means elsewhere; the
unfiltered one keeps the real right tail, which the filter cuts hard, and is the
honest input if the question is about long requests.

    python prepare_traces.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import env  # noqa: F401  (bootstraps HF cache env vars at import)
from lib.env import DERIVED_TRACES_DIR

SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
REFERENCE_TOKENIZER = "meta-llama/Llama-3.1-8B"

# The standard vLLM benchmark filter.
MIN_LEN = 4
MAX_PROMPT = 1024
MAX_TOTAL = 2048


def fetch_sharegpt() -> Path:
    from huggingface_hub import hf_hub_download

    print(f"fetching {SHAREGPT_REPO}/{SHAREGPT_FILE} (~640 MB, cached after first run)...")
    return Path(hf_hub_download(SHAREGPT_REPO, SHAREGPT_FILE, repo_type="dataset"))


def first_turn_pairs(path: Path) -> list[tuple[str, str]]:
    """(human, assistant) text for the first exchange of each conversation."""
    with open(path) as f:
        data = json.load(f)
    pairs = []
    for conv in data:
        turns = conv.get("conversations") or []
        if len(turns) < 2:
            continue
        # Conversations do not all start with the human turn; find the first
        # human->assistant adjacency rather than assuming turns[0]/turns[1].
        for a, b in zip(turns, turns[1:]):
            if a.get("from") in ("human", "user") and b.get("from") in ("gpt", "assistant"):
                pairs.append((a.get("value") or "", b.get("value") or ""))
                break
    return pairs


def tokenize_lengths(pairs, tokenizer, batch: int = 2000):
    import numpy as np

    prompts = [p for p, _ in pairs]
    outputs = [o for _, o in pairs]

    def lengths(texts):
        out = np.empty(len(texts), dtype=np.int32)
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tokenizer(chunk, add_special_tokens=False)["input_ids"]
            out[i:i + len(chunk)] = [len(x) for x in enc]
            if (i // batch) % 10 == 0:
                print(f"    tokenized {i + len(chunk):,}/{len(texts):,}", flush=True)
        return out

    print("  tokenizing prompts...")
    p_len = lengths(prompts)
    print("  tokenizing outputs...")
    d_len = lengths(outputs)
    return p_len, d_len


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default=REFERENCE_TOKENIZER)
    ap.add_argument("--out-dir", type=Path, default=DERIVED_TRACES_DIR)
    ap.add_argument("--force", action="store_true", help="rebuild even if the CSV exists")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    filtered_csv = args.out_dir / "sharegpt_llama31.csv"
    full_csv = args.out_dir / "sharegpt_llama31_full.csv"
    if filtered_csv.exists() and not args.force:
        print(f"{filtered_csv} exists; pass --force to rebuild")
        return

    import pandas as pd
    from transformers import AutoTokenizer

    path = fetch_sharegpt()
    print("parsing conversations...")
    pairs = first_turn_pairs(path)
    print(f"  {len(pairs):,} conversations with a human->assistant first exchange")

    print(f"loading tokenizer {args.tokenizer}...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    p_len, d_len = tokenize_lengths(pairs, tok)

    df = pd.DataFrame({"num_prefill_tokens": p_len, "num_decode_tokens": d_len})
    df["num_total_tokens"] = df.num_prefill_tokens + df.num_decode_tokens
    df.to_csv(full_csv, index=False)

    keep = ((df.num_prefill_tokens >= MIN_LEN) & (df.num_decode_tokens >= MIN_LEN)
            & (df.num_prefill_tokens <= MAX_PROMPT) & (df.num_total_tokens <= MAX_TOTAL))
    df[keep].to_csv(filtered_csv, index=False)

    def describe(d, label):
        print(f"  {label:9s} n={len(d):7,}  "
              f"prefill mean {d.num_prefill_tokens.mean():7.1f} "
              f"p50 {d.num_prefill_tokens.median():6.0f} "
              f"p99 {d.num_prefill_tokens.quantile(0.99):7.0f}  |  "
              f"decode mean {d.num_decode_tokens.mean():7.1f} "
              f"p50 {d.num_decode_tokens.median():6.0f} "
              f"p99 {d.num_decode_tokens.quantile(0.99):7.0f}")

    print(f"\nwrote {full_csv}")
    describe(df, "full")
    print(f"wrote {filtered_csv}")
    describe(df[keep], "filtered")
    print(f"\nthe standard filter keeps {keep.mean():.1%} of conversations")


if __name__ == "__main__":
    main()
