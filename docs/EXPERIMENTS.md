# The experiments: what each one measures, and how

Every number in the paper comes from one of the measurements below, taken with the scripts in this
repository, or from the analytical model (`lib/power_model.py`) built from those measurements. The
launchers under `slurm/` carry the exact windows, caps and grids used.

## Models

| slug | model | parameters | per-replica cap in the fleets | note |
|---|---|---|---|---|
| `llama32-1b` | meta-llama/Llama-3.2-1B | 1.24 B | 128 (small pool) | core |
| `qwen3-1.7b` | Qwen/Qwen3-1.7B | 1.72 B | 128 | |
| `llama32-3b` | meta-llama/Llama-3.2-3B | 3.21 B | 128 | |
| `qwen3-4b` | Qwen/Qwen3-4B | 4.02 B | 128 (small pool) | core |
| `mistral-7b` | mistralai/Mistral-7B-Instruct-v0.2 | 7.24 B | 64 | |
| `llama31-8b` | meta-llama/Llama-3.1-8B | 8.03 B | 64 (large pool) | core |
| `qwen3-8b` | Qwen/Qwen3-8B | 8.19 B | 64 (large pool) | core |
| `qwen3-14b` | Qwen/Qwen3-14B | 14.8 B | 64 | |
| `mistral-24b` | mistralai/Mistral-Small-24B-Instruct-2501 | 23.6 B | 64 | second datacenter site (small model) |
| `llama31-70b` | meta-llama/Llama-3.1-70B-Instruct | 70.6 B | 64, one instance over four GPUs (TP = 4) | second datacenter site (large model) |

Every model is served by vLLM 0.11.0 with `gpu_memory_utilization` 0.90, `max_model_len` 8192,
chunked prefill with a 2048-token per-step budget (vLLM's default under this engine context,
recorded in every run), bf16 on the H100 and RTX 5090, fp16 on the V100 and RTX 8000
(`lib/hardware.resolve_dtype`). The engine core runs in-process
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`) so the scheduler's waiting list can be read every step; this
costs ~1.24x of single-replica throughput against vLLM's default and applies uniformly.

## Workloads

Only the **shape** of a workload is replayed: each request is a random token-id prompt of an exact
length with the output length forced exactly (`ignore_eos`, `min_tokens = max_tokens`), so the
ground truth of every request is known and the same token counts are served to every model
regardless of its tokenizer.

| name | source | mean prompt / output (tokens) |
|---|---|---|
| `sharegpt` | ShareGPT V3, first human / assistant turn, the standard vLLM filter, Llama-3.1 tokenizer (`prepare_traces.py`) | 87 / 308 |
| `humaneval` | the 164 HumanEval problems: signature + docstring / canonical solution (`prepare_humaneval_trace.py`) | 131 / 54 |
| `arxiv_summarization` | long-document summarization trace (`data/traces/arxiv_summarization_*.csv`) | 2588 / 262 |
| `azure_conv_2024`, `azure_code_2024` | 100,000 requests sampled from each week of the Azure LLM inference trace 2024 (`prepare_azure_trace.py`), clipped to the 8192-token context | 1630 / 105 and 2464 / 22 |
| `fixed_128_128`, `fixed_256_256`, `fixed_512_512`, `fixed_1024_1024`, `fixed_2048_2048` | controlled 1:1 ladder | as named |
| `fixed_6144_1024` | long-context ablation (KV-bound at the cap) | 6144 / 1024 |

Prompts are clipped at 4096 and outputs at 2048 tokens (`lib/workloads.py`). Arrivals are
Poisson unless stated (`lib/arrivals.py` also offers bursty and gamma streams).

## Hardware

- H100 80 GB (SXM, 700 W limit), one card per single-GPU job, four or eight per fleet job on a SLURM
  cluster whose cgroups expose only the allocated cards; everything in the main text.
- V100 32 GB (SXM2, same cluster), RTX 8000 48 GB (an 8-GPU workstation) and RTX 5090 32 GB (a
  2-GPU workstation): the single-GPU model in the appendix, one physical card per model.
- Power is sampled by NVML at ~400 Hz in a background thread (`lib/power_sampler.py`); a window's
  power is the mean of the samples in it. A multi-card instance sums its cards.

## 1. Closed-loop concurrency sweep (`concurrency_sweep.py`)

`concurrency_sweep.py --model <slug> --workload <a,b,...>`; launchers
`slurm/concurrency_sweep.sbatch`, `slurm/concurrency_sweep_v100.sbatch`.

One engine per model (built once); for each level c of the model's grid (1 .. 512 for the small
models, 1 .. 256 for the 8B class, 1 .. 96 for the 24B) the client keeps exactly c requests in
flight, resubmitting on completion, and records over a steady window: the achieved mean resident
batch b, mean time between tokens (TBT), mean end-to-end time, completions, throughput, mean power,
KV-cache use and the waiting list. Warm-up / window 20 / 60 s on ShareGPT-class tasks, 30 / 90 s on
1024-token tasks, 60 / 180 s on 2048/2048, 45 / 120 s for the 70B (a level must hold at least two
request lifetimes); 20 s of warm idle before the grid. `--kv-guard` drops levels above 0.9 x KV pool /
(mean prompt + mean output) on small cards. Output: `levels.csv` (one row per level) and
`summary.json` (device, idle power, settings, the KV pool).

## 2. The calibration (`fit_power_model.py`)

Fits, per (model, workload) sweep, the power profile P(b) = P_idle + (P_busy - P_idle) g(b) with g
chosen among seven forms by held-out error (every third level withheld), the step-time line
tau(b) = t0 + t1 b (OLS of mean TBT on b) and `pack_cap` (the largest c meeting a 50 ms p95 TBT);
levels that completed no request are dropped, sweeps with fewer than 8 usable levels skipped.
`results/power_curves.json` + `power_curves_report.json` (which records the source sweep of each
curve) are the calibration the fleet model and the datacenter simulation use;
`refit_power_curves.py` reproduces them. The single-GPU figures refit the same quantities per cell
from the latest sweep of that cell (`analyze/plot_single_gpu_model.py`).

## 3. Open-loop capacity (`capacity_bisection.py`)

`capacity_bisection.py --model <slug> --workload <w> --max-num-seqs <cap>`; launchers
`slurm/capacity_bisection.sbatch` (`MAX_NUM_SEQS` sets the cap, by default the top of the model's
grid; the fleets use 64 for the large and 128 for the small models) and
`slurm/capacity_bisection_v100.sbatch` (V100; `SPEC=model@cap:...`, one cell per array task).

Capacity is the largest Poisson arrival rate one replica sustains without its waiting list growing:
bisection (6 iterations) between `--lo` and an upper bracket found by probing, each probe a fresh
Poisson stream at the rate for `--probe-s` seconds after `--warmup-s` of warm-up, with the engine
reset (every request aborted, the engine drained) between probes. Stability: the slope of a line
fitted to the waiting depth over the window is below max(1 % of the rate, 2 standard errors) -- no
absolute floor; the capacity is the highest stable probe below the lowest diverging one. Windows:
90 / 25 s on ShareGPT-class tasks, 180 / 60 s on 512- and 1024-token tasks, 240 / 60-90 s on arXiv
and 6144/1024, 300-540 / 90-240 s on 2048/2048, 180 / 60 s on the Azure lengths; 2-3x longer on
the V100 and RTX 8000 (`PROBE_S`, `WARMUP_S`). Output: `capacity.json` -- the capacity, the power
the replica drew at it, every probe (rate, waiting slope and standard error, running mean, power,
residual backlog at the reset) and the stability rule.

## 4. One replica under Poisson arrivals (`poisson_load.py`)

`poisson_load.py --model <slug> --workload <a,b,...> --max-num-seqs <cap>`; launcher
`slurm/poisson_load.sbatch`.

At each of 0.15, 0.30, 0.45, 0.60 and 0.75 of the cell's measured capacity (a workload without a
measured capacity is skipped), the replica warms up until at least 10 requests finished and two
mean residences passed (cap 900 s), is observed for max(120 s, 8 mean residences) <= 480 s while
every request's submit, first-token and completion times are logged and the scheduler's running and
waiting counts are sampled at 10 Hz, then admissions stop and the replica drains naturally with the
in-flight set snapshotted (prompt, output and tokens generated so far per request, so every
residual is known exactly); re-warm and drain three times per rate. Every
phase is judged by the capacity bisection's stability rule. Output: `summary.json` (per phase:
rate, verdict, warm-up actually run, window, mean resident count, power, drains) and
`requests.csv` (per request).

## 5. KV ceiling (`slurm/kv_ceiling_sweep.sbatch`)

`concurrency_sweep.py --concurrency <levels> --max-num-seqs <2.2x the ceiling>` on a
long-context workload; `CONCURRENCY=auto` places the levels at 0.4 .. 2x round(K / mean resident
context) from the pool the card measured (`analyze/kv_ceiling_levels.py`), 90 / 240 s windows. The
plateau of the mean resident count with the KV cache full is the measured ceiling.

## 6. Fleet routing sweeps (`fleet_routing_sweep.py`)

`fleet_routing_sweep.py`; launchers `slurm/fleet_sweep_4gpu.sbatch` (3 large + 1 small on four
H100s, loads u = 0.1 .. 0.9 in two array tasks per workload; `WL` selects another workload, e.g.
1024/1024) and `slurm/fleet_sweep_8gpu.sbatch` (eight H100s, 5 + 3 or 4 + 4).

A fleet is N large + M small vLLM replicas, each in its own process pinned to one GPU with its
own NVML sampler (`lib/fleet.py`), behind a router (`lib/router.py`) that **packs**: a new request
goes to the first replica of its pool holding fewer than the cap (64 large, 128 small). All
traffic is flexible; the routing fraction f is the share sent to the small pool. Offered load is
Poisson at u x C_fleet with C_fleet = N C_L + M C_S from the **measured** capacities
(`analyze/capacity_lookup.py`; the launcher refuses a missing, stale or inconsistent cell). At each
load and each f of the grid 0, 0.1, .. 1 (plus extra fractions inside the feasible window at high
load, `FGRID`), the fleet is drained to empty, then run for 150 s; power and lit-replica counts are
averaged over the second half, the waiting list's growth over that half is recorded. The analysis
(`analyze/score_fleet_model.py`) counts a probe as **held** if it started drained and its waiting
list did not grow; the **feasible window** at a load is the range of held fractions, the **band**
the power range over them. Output per run: `points.csv` (one row per (load, f)), `trace.csv` (1 Hz
fleet telemetry: power per replica, lit counts, queue depths), `bands.csv`, `summary.json`.

Fleets measured: Llama-3.1-8B x3 + Llama-3.2-1B and Qwen3-8B x3 + Qwen3-4B on ShareGPT; the Llama
pair on arXiv, 128/128, 256/256, 512/512 and 1024/1024; the Qwen pair on 1024/1024; on eight GPUs
the Llama pair as 5 + 3 (u = 0.1-0.9) and 4 + 4 (u = 0.1-0.5).

## 7. A large model over four GPUs

Llama-3.1-70B-Instruct is served as one vLLM instance with tensor parallelism 4 (set in
`lib/models.py`). Steps 1, 3, 4 and 5 run through the same launchers on four GPUs
(`sbatch --gres=gpu:h100:4 ...` with `MODEL=llama31-70b`, `PREFETCH=1`); power is the sum over the
four cards, each of which must hold its weight shard (`lib/power_sampler.assert_instance_resident`).
Results land under `results/h100-80gb-tp4/`; the capacities feed the second datacenter site.

## 8. The Azure request lengths

The Azure 2024 trace's own chat and code request lengths are the workloads `azure_conv_2024` and
`azure_code_2024` (built by `prepare_azure_trace.py`). Steps 1 (with the KV guard) and 3 (60 / 180 s
windows) on them, for the reference pair, Mistral-24B and the 70B, feed the datacenter simulation's
second site.

## 9. The datacenter-scale simulation (`analyze/datacenter_sim.py`)

No GPU. The analytical model built exactly as for the fleets (single-GPU calibration, measured
capacities, packed occupancy, the exact closed-form solve) with the replica count raised to a
10,000-GPU site: 7,500 large and 2,500 small replicas, the busiest five minutes of a week at 60 % of
the site's measured capacity, each pool allowed up to its full measured capacity, normal operation
splitting every query between the pools by capacity, a cut moving the flexible queries normal
operation placed on the large model to the small one. Facility power = PUE 1.2 x (GPU power + 300 W
of non-GPU IT per GPU). Inputs (`data/datacenter/`, fetched and reduced by
`analyze/fetch_datacenter_inputs.py`): arrival counts per five minutes of the Azure 2024 chat and
code traces (timing), EIA-930 balancing-authority load for PJM and ERCOT, and a curated table of grid
programs (`grid_programs.csv`, sources inside). The measured 1-s drop and return shape comes from the
open-loop drop runs (`results/power_drops/drop_traces.csv.gz`); drains of the second site are
projected with the offline drain predictor (`results/power_drops/predicted_drain.csv`). `--only`
selects scenarios (all are run by default; see the script's docstring); `--selfcheck` verifies the
closed form against the exact solve.
