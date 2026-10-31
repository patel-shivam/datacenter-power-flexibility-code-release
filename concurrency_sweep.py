#!/usr/bin/env python
"""concurrency_sweep -- TBT, throughput, power, memory vs the max-parallelism cap, per model/workload.

This is the foundational measurement. Everything downstream reads it:

  * ``pack_cap`` comes from the TBT curve: the largest concurrency at which this model
    still meets its inter-token-latency target.
  * The analytical power model P(b) is fitted to the (running batch, power) pairs here.
  * Per-replica capacity and the residence-time coefficients A, B come from the
    throughput and latency curves.

Method
------
The engine is built **once** at the top of the concurrency grid and concurrency is held
at each level from the client side: keep exactly ``c`` requests in flight, submitting a
replacement the moment one finishes. Two reasons this is the right primitive:

  1. It pins the running batch at exactly ``c``, so the power model is fitted against a
     known b rather than against a noisy average of a fluctuating one.
  2. Rebuilding the engine per level costs 1-4 minutes per level per model (a 24B takes
     minutes to load), which for 9 models x 13 levels x several workloads is most of
     the GPU budget spent on model loading.

``--cap-mode engine`` rebuilds the engine with ``max_num_seqs=c`` at each level instead.
It is the ground truth for "what does the vLLM cap itself do", and the two differ where
vLLM's CUDA-graph capture sizes and scheduler budget depend on the declared cap. Run it
at a few levels to check the client-side proxy, not for the whole grid.

Every reported number is taken over a steady-state window *after* a warmup, and the
**achieved** mean running batch is recorded next to the requested one. They diverge at
the top of the grid when KV, not the cap, is what binds -- and that divergence is a
result, not an error: it is where the model runs out of memory rather than out of
scheduler slots.

    python concurrency_sweep.py \
        --model llama31-8b --workload sharegpt
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import env  # noqa: F401  (bootstraps env before vllm is imported)
from lib import engine as eng
from lib import hardware, metrics, models
from lib import power_sampler as ps
from lib import request_gen as reqlib
from lib import workloads
from lib.env import run_dir


def measure_idle(sampler: ps.PowerSampler, seconds: float) -> dict:
    """Power with the model resident and nothing running.

    This is P_idle -- the floor the power model is anchored on -- and it is measured,
    never fitted. It is also not the same as the power of a GPU with no model loaded:
    weights sitting in HBM cost real watts, and 'idle' here means warm-idle, the state
    a drained replica actually sits in.
    """
    t0 = time.monotonic()
    time.sleep(seconds)
    t1 = time.monotonic()
    avg = sampler.average_over(t0, t1) or {}
    return {"idle_power_w": avg.get("avg_power_w", float("nan")),
            "idle_mem_used_mib": avg.get("avg_mem_used_mib", float("nan")),
            "idle_sm_clock_mhz": avg.get("avg_sm_clock_mhz", float("nan")),
            "idle_window_s": seconds}


def run_level(engine, specs, concurrency: int, sampler: ps.PowerSampler,
              warmup_s: float, measure_s: float, slo: metrics.SLO,
              rng: random.Random) -> dict:
    """Hold exactly ``concurrency`` requests in flight and measure the steady state."""
    records: dict[str, metrics.RequestRecord] = {}
    order = list(range(len(specs)))
    rng.shuffle(order)
    cursor = 0
    next_id = 0

    def submit() -> None:
        nonlocal cursor, next_id
        spec = specs[order[cursor % len(order)]]
        cursor += 1
        # ids carry the level: if the previous level's drain timed out, its requests are
        # still resident, and a bare "r0" collides ("Request id r0 already running", the
        # V100 shakeout 47440867, where 1024-token decodes outlasted a 180 s drain)
        rid = f"c{concurrency}-r{next_id}"
        next_id += 1
        now = time.monotonic()
        engine.add_request(rid, {"prompt_token_ids": spec.prompt_token_ids},
                           reqlib.sampling_params(spec))
        records[rid] = metrics.RequestRecord(
            rid, 0, "solo", "flexible", spec.num_prefill_tokens,
            spec.num_decode_tokens, now)

    for _ in range(concurrency):
        submit()

    watch = metrics.Stopwatch(warmup_s, measure_s)
    running_samples: list[int] = []
    waiting_samples: list[int] = []
    kv_samples: list[float] = []
    step_times: list[float] = []
    window_records: list[metrics.RequestRecord] = []
    last_probe = 0.0

    while not watch.done:
        t_step = time.monotonic()
        outs = engine.step()
        step_times.append(time.monotonic() - t_step)
        now = time.monotonic()
        for o in outs:
            rec = records.get(o.request_id)
            if rec is None:
                continue
            n_tok = len(o.outputs[0].token_ids) if o.outputs else 0
            if n_tok > 0 and rec.tokens_emitted == 0:
                rec.t_first_token = now
            if n_tok > rec.tokens_emitted:
                rec.tokens_emitted = n_tok
                rec.t_last_token = now
            if o.finished:
                # Only requests that both started and finished inside the measurement
                # window describe the steady state; one that began during warmup carries
                # the ramp's queueing in its TTFT.
                if watch.window_start <= rec.t_submit and now <= watch.window_end:
                    window_records.append(rec)
                records.pop(o.request_id, None)
                submit()
        if now - last_probe > 0.05:
            st = eng.read_state(engine)
            if watch.in_window:
                running_samples.append(st.num_running)
                waiting_samples.append(st.num_waiting)
                kv_samples.append(st.kv_usage)
            last_probe = now

    pw = sampler.average_over(watch.window_start, watch.window_end) or {}
    pct = sampler.percentiles_over(watch.window_start, watch.window_end) or {}
    energy_j = sampler.energy_over(watch.window_start, watch.window_end)
    summary = metrics.summarize(window_records, window_s=measure_s, slo=slo)

    mean_running = (sum(running_samples) / len(running_samples)) if running_samples else 0.0
    decode_tok = summary.get("decode_tokens", 0)
    return {
        "concurrency_requested": concurrency,
        "concurrency_achieved": mean_running,
        "concurrency_achieved_max": max(running_samples) if running_samples else 0,
        "num_waiting_mean": (sum(waiting_samples) / len(waiting_samples)) if waiting_samples else 0.0,
        "kv_usage_mean": (sum(kv_samples) / len(kv_samples)) if kv_samples else 0.0,
        "kv_usage_max": max(kv_samples) if kv_samples else 0.0,
        "step_time_ms_mean": 1e3 * sum(step_times) / len(step_times) if step_times else float("nan"),
        "n_steps": len(step_times),
        **{k: v for k, v in pw.items()},
        **{k: v for k, v in pct.items()},
        "energy_j": energy_j,
        "energy_per_decode_token_j": (energy_j / decode_tok) if (energy_j and decode_tok) else None,
        "n_completed": summary["n_completed"],
        "throughput_reqs_s": summary.get("throughput_reqs_s", 0.0),
        "throughput_decode_tok_s": summary.get("throughput_decode_tok_s", 0.0),
        "throughput_total_tok_s": summary.get("throughput_total_tok_s", 0.0),
        "goodput_reqs_s": summary.get("goodput_reqs_s"),
        "slo_attainment": summary.get("slo_attainment"),
        "ttft_ms_mean": summary["ttft_ms"]["mean"], "ttft_ms_p50": summary["ttft_ms"]["p50"],
        "ttft_ms_p95": summary["ttft_ms"]["p95"], "ttft_ms_p99": summary["ttft_ms"]["p99"],
        "tbt_ms_mean": summary["tbt_ms"]["mean"], "tbt_ms_p50": summary["tbt_ms"]["p50"],
        "tbt_ms_p95": summary["tbt_ms"]["p95"], "tbt_ms_p99": summary["tbt_ms"]["p99"],
        "e2e_ms_mean": summary["e2e_ms"]["mean"], "e2e_ms_p50": summary["e2e_ms"]["p50"],
    }


def kv_guarded_grid(grid: list[int], kv_pool_tokens: float | None, wl,
                    keep_frac: float = 0.9) -> tuple[list[int], dict]:
    """The levels of ``grid`` the KV pool can hold for ``wl``, and a record of the cut.

    The bound is PEAK-aware: a sequence occupies its full context (prompt plus every
    token it will generate) just before it finishes, so ``c`` resident sequences need
    ``c * (E[P] + E[D])`` tokens at the moment the batch is oldest. The decode-weighted
    mean context predicts the *mean* occupancy well, but the peak runs ~1.2x above it, and
    the first grid level above pool / (P + D) saturates on every 1:1-ladder cell.
    ``keep_frac`` leaves headroom for the prompt-length spread. The first level is always
    kept.
    """
    if not kv_pool_tokens or kv_pool_tokens <= 0:
        return list(grid), {"applied": False}
    full_context = float(wl.mean_prefill) + float(wl.mean_decode)
    ceiling = kv_pool_tokens / full_context
    kept = [c for c in grid if c <= keep_frac * ceiling] or [grid[0]]
    return kept, {"applied": True, "kv_pool_tokens": kv_pool_tokens,
                  "full_context_tokens": full_context, "ceiling": ceiling,
                  "keep_frac": keep_frac, "dropped": [c for c in grid if c not in kept]}


def drain(engine, timeout_s: float = 600.0) -> float:
    """Step until nothing is in flight. Requests still running belong to the level that
    submitted them, so the next level must not start with them resident -- their decode
    would land in the next window's power and throughput. A slow card decoding long
    outputs can need minutes (a V100 on 1024-token decodes can take longer than 180 s); a
    drain that still times out is reported, because the next level is then contaminated."""
    t0 = time.monotonic()
    while engine.has_unfinished_requests() and time.monotonic() - t0 < timeout_s:
        engine.step()
    if engine.has_unfinished_requests():
        print(f"  WARNING: drain timed out after {timeout_s:.0f} s with requests still "
              f"running; the next level starts contaminated", flush=True)
    return time.monotonic() - t0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help=f"one of: {', '.join(models.ALL_SLUGS)}")
    p.add_argument("--workload", default="sharegpt",
                   help=("comma-separated; all are swept on ONE engine load, which "
                         "matters because a cold model read from Lustre costs minutes "
                         "(measured: 8m52s for a single 4 GB shard of Llama-3.1-8B). "
                         f"choices: {', '.join(workloads.ALL_WORKLOADS)}"))
    p.add_argument("--concurrency", type=str, default=None,
                   help="comma-separated override of the model's default grid")
    p.add_argument("--cap-mode", choices=("client", "engine"), default="client",
                   help="client: hold c in flight on one engine (default). "
                        "engine: rebuild with max_num_seqs=c per level (ground truth)")
    p.add_argument("--warmup-s", type=float, default=20.0)
    p.add_argument("--measure-s", type=float, default=60.0)
    p.add_argument("--idle-s", type=float, default=15.0)
    p.add_argument("--pool-size", type=int, default=4096,
                   help="requests pregenerated up front; the timed loop never builds prompts")
    p.add_argument("--slo-ttft-ms", type=float, default=None)
    p.add_argument("--slo-tbt-ms", type=float, default=50.0)
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=None,
                   help="client mode: the engine's max_num_seqs (default: the top of the "
                        "grid). Raise it above the grid to sweep PAST the KV ceiling without "
                        "the request cap binding first (the KV-ceiling figure).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kv-guard", action="store_true",
                   help="drop the grid levels the KV pool cannot hold for a workload "
                        "(c > 0.9 * pool_tokens / (mean prefill + mean decode), the "
                        "PEAK-aware bound: every resident sequence reaches its full "
                        "context before it leaves); client cap mode only, since the pool "
                        "is known once the engine is built. Levels above the bound measure "
                        "a memory limit, not P(b).")
    p.add_argument("--smoke", action="store_true",
                   help="tiny windows and 3 concurrency levels, for a hardware shakeout")
    p.add_argument("--out-tag", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = models.get(args.model)
    # Before run_dir: a multi-GPU instance writes under its own namespace (h100-80gb-tp4).
    hardware.set_instance(spec.tensor_parallel_size)
    wls = [workloads.get(w.strip()) for w in args.workload.split(",")]
    slo = metrics.SLO(ttft_ms=args.slo_ttft_ms, tbt_ms=args.slo_tbt_ms)

    grid = ([int(x) for x in args.concurrency.split(",")] if args.concurrency
            else list(spec.concurrency_grid))
    if args.smoke:
        grid = grid[:3]
        args.warmup_s, args.measure_s, args.idle_s = 3.0, 8.0, 4.0
        args.pool_size = 256

    if args.cap_mode == "engine" and len(wls) > 1:
        raise SystemExit("--cap-mode engine rebuilds per level; use one workload at a time")

    print(f"concurrency_sweep: {spec.model_id}")
    print(f"  workloads: {', '.join(w.label() for w in wls)}")
    print(f"  grid: {grid}")

    tag = args.out_tag or (f"{spec.slug}__{'-'.join(w.name for w in wls)}"
                           f"__{args.cap_mode}")
    out = run_dir("concurrency_sweep", tag)
    print(f"  out:  {out}")

    sampler = ps.for_model(spec)  # its own card, or every card of a multi-GPU instance
    sampler.start()
    dev = sampler.device_info()
    print(f"  gpu:  {dev['name']} limit {dev['power_limit_w']:.0f} W "
          f"energy_counter={dev['energy_counter_supported']}")
    if sampler.num_devices > 1:
        print(f"  instance: TP={spec.tensor_parallel_size} over NVML {dev['device_indices']} "
              f"({', '.join(dev['pci_bus_ids'])}); power is the sum", flush=True)

    rows: list[dict] = []
    meta = {
        "model": spec.model_id, "model_slug": spec.slug, "params_b": spec.params_b,
        "workloads": [w.descriptor() for w in wls],
        "workload_stats": {w.name: {"mean_prefill": w.mean_prefill,
                                    "mean_decode": w.mean_decode,
                                    "cv_decode": w.cv_decode} for w in wls},
        "cap_mode": args.cap_mode, "grid": grid, "device": dev,
        # THE CHUNK BUDGET, RECORDED. None means "whatever this vLLM defaults to", which
        # is not knowable after the fact: 0.11.0 exposes
        # vllm.utils.DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048, and 0.29.0 does not export it
        # at all. Every residence coefficient fitted from a long-prefill workload depends on
        # this number (a ShareGPT prompt, mean prefill 87, is a single chunk under any
        # budget). Pass --max-num-batched-tokens on long-prefill work.
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        # Provenance: which accelerator, and what dtype it forced. Plotting reads
        # these back and refuses to put two different cards on one axis.
        "hardware": hardware.current(),
        "dtype": hardware.resolve_dtype(spec.dtype),
        "warmup_s": args.warmup_s, "measure_s": args.measure_s,
        "slo": {"ttft_ms": args.slo_ttft_ms, "tbt_ms": args.slo_tbt_ms},
        "seed": args.seed, "vllm_multiprocessing": "disabled (in-process engine core)",
    }
    if spec.tensor_parallel_size > 1:
        meta["tensor_parallel_size"] = spec.tensor_parallel_size
        meta["power_scope"] = (f"sum over the instance's {sampler.num_devices} GPUs "
                               f"(per-card: avg_power_w_gpu<k>)")

    engine = None
    try:
        if args.cap_mode == "client":
            t_load = time.monotonic()
            engine = eng.build_engine(
                spec, max_num_seqs=args.max_num_seqs or max(grid),
                max_num_batched_tokens=args.max_num_batched_tokens,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len)
            meta["load_s"] = time.monotonic() - t_load
            if spec.tensor_parallel_size > 1:
                meta["mem_used_mib_after_load"] = ps.assert_instance_resident(sampler, spec)
            meta["kv"] = eng.kv_sizing(engine)
            print(f"  loaded in {meta['load_s']:.0f}s; kv "
                  f"{meta['kv']['kv_pool_tokens']:,} tokens "
                  f"({meta['kv']['kv_pool_gib']:.1f} GiB)", flush=True)
            meta["idle"] = measure_idle(sampler, args.idle_s)
            print(f"  idle: {meta['idle']['idle_power_w']:.1f} W", flush=True)

        for wl in wls:
            print(f"\n  == workload {wl.label()} ==", flush=True)
            level_grid = grid
            if args.cap_mode == "client":
                vocab, special = eng.tokenizer_info(engine)
                specs = reqlib.pregenerate(wl, args.pool_size, vocab,
                                           random.Random(args.seed), special)
                if args.kv_guard:
                    level_grid, cut = kv_guarded_grid(
                        grid, (meta.get("kv") or {}).get("kv_pool_tokens"), wl)
                    meta.setdefault("kv_guard", {})[wl.name] = cut
                    if cut.get("dropped"):
                        print(f"  kv guard: pool {cut['kv_pool_tokens']:,} tokens / "
                              f"{cut['full_context_tokens']:.0f} per sequence = "
                              f"{cut['ceiling']:.0f}; dropping c = {cut['dropped']}",
                              flush=True)
            for c in level_grid:
                if args.cap_mode == "engine":
                    if engine is not None:
                        eng.teardown(engine)
                    engine = eng.build_engine(
                        spec, max_num_seqs=c,
                        max_num_batched_tokens=args.max_num_batched_tokens,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        max_model_len=args.max_model_len)
                    vocab, special = eng.tokenizer_info(engine)
                    specs = reqlib.pregenerate(wl, min(args.pool_size, 8 * c + 64), vocab,
                                               random.Random(args.seed), special)
                    if "kv" not in meta:
                        meta["kv"] = eng.kv_sizing(engine)
                        meta["idle"] = measure_idle(sampler, args.idle_s)

                t0 = time.monotonic()
                row = run_level(engine, specs, c, sampler, args.warmup_s, args.measure_s,
                                slo, random.Random(args.seed + c))
                row["wall_s"] = time.monotonic() - t0
                row["workload"] = wl.name
                row["drain_s"] = drain(engine)
                rows.append(row)
                print(f"  c={c:<4d} b={row['concurrency_achieved']:6.1f}  "
                      f"P={row.get('avg_power_w', float('nan')):6.1f}W  "
                      f"tok/s={row['throughput_decode_tok_s']:8.1f}  "
                      f"TBT p50={row['tbt_ms_p50']:6.1f}ms p99={row['tbt_ms_p99']:7.1f}ms  "
                      f"TTFT p50={row['ttft_ms_p50']:7.1f}ms  "
                      f"KV={row['kv_usage_mean']:.1%}  "
                      f"mem={row.get('avg_mem_used_mib', 0)/1024:.1f}GiB", flush=True)
    finally:
        samples = sampler.stop()
        ps.write_trace_csv(samples, out / "power_trace.csv.gz")
        sampler.shutdown()
        if engine is not None:
            try:
                eng.teardown(engine)
            except Exception:
                pass

    if rows:
        with open(out / "levels.csv", "w", newline="") as f:
            keys = sorted({k for r in rows for k in r})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    meta["n_levels"] = len(rows)
    meta["power_sample_rate_hz"] = round(len(samples) / max(samples[-1].timestamp
                                                            - samples[0].timestamp, 1e-9), 1) \
        if len(samples) > 1 else 0.0
    with open(out / "summary.json", "w") as f:
        json.dump({"meta": meta, "levels": rows}, f, indent=2, default=str)
    print(f"\nwrote {out}/levels.csv and summary.json")


if __name__ == "__main__":
    main()
