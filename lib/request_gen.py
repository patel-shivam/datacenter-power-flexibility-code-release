"""Turning a (prefill, decode) shape into a concrete vLLM request.

A request is a random token-id prompt of *exactly* ``num_prefill_tokens`` and a
sampling config that forces *exactly* ``num_decode_tokens`` of output
(``ignore_eos=True``, ``min_tokens == max_tokens``). Both counts are therefore ground
truth, not an estimate -- which is what lets a power model be fit against batch
geometry rather than against a guess at it.

The one thing to be careful about here is cost. Building the candidate token list is
O(vocab) = 128k entries for Llama-3; rebuilding it inside every call to draw
O(prefill) samples from it costs **2.9 ms per request**, all of it list construction.
Inside a timed serving loop that is 34.5% of a capacity-measurement loop and ~37% of a
fleet worker's serving thread -- enough to profile at 15,049 tok/s a model that
sustains 25,331 tok/s when its prompts are generated *before* the timed window. The
list is therefore cached per (vocab, special-ids) pair; caching does not change what
is sampled.

Prefer ``pregenerate`` and hand the timed loop a finished list.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass


@dataclass
class RequestSpec:
    prompt_token_ids: list[int]
    num_prefill_tokens: int
    num_decode_tokens: int


@functools.lru_cache(maxsize=8)
def _candidates(vocab_size: int, special: frozenset) -> tuple[int, ...]:
    return tuple(t for t in range(vocab_size) if t not in special)


def make_specs(shapes: list[tuple[int, int]], vocab_size: int, rng: random.Random,
               special_token_ids: set[int] | None = None) -> list[RequestSpec]:
    cand = _candidates(vocab_size, frozenset(special_token_ids or ()))
    n = len(cand)
    out = []
    for p, d in shapes:
        out.append(RequestSpec([cand[rng.randrange(n)] for _ in range(p)], p, d))
    return out


def pregenerate(workload, count: int, vocab_size: int, rng: random.Random,
                special_token_ids: set[int] | None = None) -> list[RequestSpec]:
    """Sample ``count`` shapes from ``workload`` and build all prompts up front."""
    return make_specs(workload.sample(count, rng), vocab_size, rng, special_token_ids)


def sampling_params(spec: RequestSpec):
    """SamplingParams forcing exactly ``spec.num_decode_tokens`` output tokens.

    ``temperature=1.0`` rather than 0: greedy decoding takes a different sampler path
    and the point here is to measure the ordinary serving path. Content is irrelevant
    -- nothing ever reads the output tokens.
    """
    from vllm import SamplingParams  # type: ignore

    return SamplingParams(max_tokens=spec.num_decode_tokens,
                          min_tokens=spec.num_decode_tokens,
                          ignore_eos=True, temperature=1.0)


def _selfcheck() -> None:
    rng = random.Random(0)
    specs = make_specs([(16, 8), (4, 2)], vocab_size=1000, rng=rng, special_token_ids={0, 1})
    assert [len(s.prompt_token_ids) for s in specs] == [16, 4]
    assert all(t not in (0, 1) for s in specs for t in s.prompt_token_ids)
    assert specs[0].num_decode_tokens == 8
    print("requests selfcheck OK")


if __name__ == "__main__":  # python -m lib.<module>
    _selfcheck()
