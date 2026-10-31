# Creating Power Flexibility in Datacenters via Query Routing

Code and data release for the paper *Creating Power Flexibility in Datacenters via Query Routing*.
This repository contains the code and data to (1) regenerate every figure of the paper from the
result tables, (2) rebuild those tables from the measurements, and (3) re-run the measurements on
your own GPUs.

## 1. Reproducing the figures (no GPU)

```
pip install -r requirements.txt        # numpy, pandas, matplotlib, scipy (Python 3.12)
python make_figures.py                 # -> figures/main/ and figures/appendix/ (PDF + PNG)
python make_figures.py --list          # the figure map
```

`make_figures.py` draws the figures from the tables under `results/tables/` (`--out DIR` writes
elsewhere, `--only single_gpu,fleet,datacenter` draws a subset). `docs/FIGURES.md` lists, for every
figure, what it shows, which tables it reads and which experiment produced them; `paper_figures/`
holds the figure code, one function per figure.

## 2. Layout

| path | contents |
|---|---|
| `make_figures.py`, `paper_figures/` | the figure code |
| `results/tables/` | the tables the figures read: `single_gpu/`, `fleet_routing/`, `datacenter/` |
| `results/concurrency_sweep/`, `capacity_bisection/`, `poisson_load/`, `fleet_routing_sweep/` | the measurements on the H100, one directory per run |
| `results/rtx-8000-48gb/`, `results/v100-32gb/`, `results/rtx-5090-32gb/` | the single-GPU measurements on the other cards (appendix) |
| `results/h100-80gb-tp4/` | Llama-3.1-70B served as one instance over four H100s |
| `results/power_curves.json`, `results/power_curves_report.json` | the single-GPU power calibration the fleet model is built from |
| `results/power_drops/` | the measured 1-s power drop and return, and the predicted drain times, that the datacenter simulation uses |
| `concurrency_sweep.py`, `capacity_bisection.py`, `poisson_load.py`, `fleet_routing_sweep.py` | the experiments (GPU) |
| `fit_power_model.py`, `refit_power_curves.py`, `prepare_*.py` | the P(b) fit and its reproduction check; the trace builders |
| `lib/` | the experiment library: engine, workloads, arrivals, power sampling, fleet, router, feasibility, the analytical model (`power_model.py`) |
| `analyze/` | the table producers and the fleet model (`plot_single_gpu_model.py`, `plot_fleet_routing_figures.py`, `datacenter_sim.py`, `capacity_lookup.py`, ...) |
| `slurm/` | SLURM launchers, one per experiment, and a runner for a workstation without SLURM |
| `data/traces/` | the request-length traces (ShareGPT, HumanEval, arXiv summarization, Azure 2024 chat and code, splitwise) |
| `data/datacenter/` | the reduced public inputs of the simulation; `SOURCES.json` records each source's URL, access date, size and SHA-256 |
| `docs/` | `FIGURES.md` (figure -> tables -> experiment), `EXPERIMENTS.md` (protocols) |

## 3. From measurements to tables (no GPU)

```
python analyze/plot_single_gpu_model.py        # -> results/tables/single_gpu/    (5-15 min)
python analyze/plot_fleet_routing_figures.py   # -> results/tables/fleet_routing/ (~2 min)
python analyze/datacenter_sim.py               # -> results/tables/datacenter/    (a few minutes; --selfcheck tests the solver)
python refit_power_curves.py                   # re-fits results/power_curves.json from its recorded sweeps and compares
```

Each producer rewrites its tables (`--out-dir` / `--out` write elsewhere), together with tables
and diagnostic plots the figures do not read. What each one reads:

- **single GPU**: per run `levels.csv` + `summary.json` (concurrency sweeps), `capacity.json`
  (capacity bisections) and `summary.json` (Poisson-load runs), under `results/` for the H100 and
  `results/<card>/` for the other cards;
- **fleet**: per fleet run `summary.json`, `points.csv` and `trace.csv`; the measured capacities;
  the calibration (`results/power_curves.json` and its report);
- **datacenter**: `data/datacenter/`, the calibration, the measured capacities (including
  `results/h100-80gb-tp4/` and the Azure-length cells), the fleet probes and `results/power_drops/`.

## 4. Re-running the measurements on your hardware

**Environment.** Python 3.12, vLLM 0.11.0, PyTorch 2.8.0 (CUDA 12.6) and `nvidia-ml-py` for
NVML. `requirements-gpu.txt` is the exact package list of the environment the paper's
measurements used; `environment.yml` creates it with conda (the torch wheels come from the cu126
index). Set `CUDA_HOME` if the toolkit is not at `/usr/local/cuda` (flashinfer compiles its
sampling kernels at first use).

**Weights.** Hugging Face ids are in `lib/models.py`: Llama-3.2-1B / 3B, Llama-3.1-8B,
Llama-3.1-70B-Instruct (four cards), Qwen3-1.7B / 4B / 8B / 14B, Mistral-7B-Instruct-v0.2 and
Mistral-Small-24B-Instruct-2501. The Llama and Mistral repositories are gated: accept their terms
and `huggingface-cli login` before downloading. `DCE_HF_CACHE=/path` points the code at a shared
cache (default `~/.cache/huggingface`). The launchers run offline (`HF_HUB_OFFLINE=1`), so
download first, e.g.
`python -c "from lib import env, models; from huggingface_hub import snapshot_download; snapshot_download(models.get('llama31-8b').model_id)"`;
`slurm/prefetch_weights.sh <slug>` reads a cached checkpoint into the page cache before a job.

**Traces.** `data/traces/` contains every length trace used. To rebuild them: `prepare_traces.py`
(ShareGPT; downloads the dataset and needs the Llama-3.1 tokenizer), `prepare_humaneval_trace.py`,
and `prepare_azure_trace.py` after `analyze/fetch_datacenter_inputs.py --only azure`.

**The experiments, in order** (protocols in `docs/EXPERIMENTS.md`):

| step | what | script | launcher | GPUs |
|---|---|---|---|---|
| 1 | closed-loop concurrency sweep: TBT, throughput, power, KV use against the resident batch | `concurrency_sweep.py` | `slurm/concurrency_sweep.sbatch` (array over models; `concurrency_sweep_v100.sbatch` on V100) | 1 |
| 2 | fit P(b) and the step-time line | `fit_power_model.py results/concurrency_sweep/*/ --out results/power_curves.json` | -- (CPU) | 0 |
| 3 | open-loop capacity by bisection on the arrival rate | `capacity_bisection.py` | `slurm/capacity_bisection.sbatch` (`MAX_NUM_SEQS` sets the cap: 64 / 128 for the fleets' large / small models); `capacity_bisection_v100.sbatch` on V100 | 1 |
| 4 | one replica under Poisson arrivals: occupancy, power and natural drains (needs 3) | `poisson_load.py` | `slurm/poisson_load.sbatch` | 1 |
| 5 | KV-ceiling sweeps (step 1 past the memory limit) | `concurrency_sweep.py --max-num-seqs` | `slurm/kv_ceiling_sweep.sbatch` (`CONCURRENCY=auto` places the levels from the measured pool) | 1 |
| 6 | fleet routing sweeps: fleet power against the routing fraction at loads u = 0.1-0.9 of the measured capacity (needs 3 at caps 64 / 128) | `fleet_routing_sweep.py` | `slurm/fleet_sweep_4gpu.sbatch` (3 large + 1 small on four GPUs; `WL` selects the workload), `slurm/fleet_sweep_8gpu.sbatch` (eight GPUs) | 4 / 8 |

Llama-3.1-70B runs as one instance over four GPUs (tensor parallelism 4, set in `lib/models.py`):
steps 1, 3, 4 and 5 use the same launchers with `MODEL=llama31-70b` on four GPUs
(`sbatch --gres=gpu:h100:4 ...`), and its results land under `results/h100-80gb-tp4/`. The Azure
trace's own request lengths are the workloads `azure_conv_2024` and `azure_code_2024`. Without
SLURM, `slurm/run_single_gpu_pipeline.sh <model> <gpu>` runs steps 1, 3, 4 and 5 for one model on
one card of a workstation; the RTX 8000 and RTX 5090 cells were measured this way.

**Submitting.** Submit from the repository's root (`sbatch slurm/concurrency_sweep.sbatch`); the
launchers find the root through `SLURM_SUBMIT_DIR` (or `REPO_DIR`), run `python` from the PATH
(or `PY=/path/to/python`), write logs to `slurm_logs/`, and carry placeholder partition and
GPU-type names (`-p gpu`, `--gres=gpu:h100:<n>`, `--gres=gpu:v100:1`) -- set them for your cluster
in the headers or on the command line (`sbatch -p <partition> --gres=gpu:<type>:<n> -A <account>
...`). Lists passed through `--export` are colon-separated, because SLURM splits its value on
commas. Every launcher's header documents its protocol and measurement windows.

Results land in `results/<experiment>/<timestamp>__<tag>/` on the baseline H100 and in
`results/<card-slug>/<experiment>/` on any other card (`lib/hardware.py` names the card from NVML),
so runs from different cards are never pooled. Then §3 regenerates the tables and §1 the figures.

**Smoke test (one GPU, about 15 minutes).** Every experiment has a `--smoke` mode (three levels,
two bisection steps, two short Poisson phases) that exercises the whole path -- engine build, NVML
sampling, scheduler reads, outputs -- without measuring anything. From the repository's root on a
GPU node (run the launchers with `bash` in an interactive session, or `sbatch` them):

```
export DCE_HF_CACHE=/path/to/hf-cache            # if the weights are in a shared cache
SMOKE=1 WORKLOADS=sharegpt:fixed_128_128 bash slurm/concurrency_sweep.sbatch
SMOKE=1 MODEL=llama32-1b WORKLOAD=sharegpt MAX_NUM_SEQS=128 bash slurm/capacity_bisection.sbatch
SMOKE=1 MODEL=llama32-1b CAP=128 WORKLOADS=sharegpt RATES=5:15 bash slurm/poisson_load.sbatch
python fleet_routing_sweep.py --n-large 3 --n-small 1 --workload sharegpt --utils 0.3,0.5 \
    --f-grid 0,0.5,1 --residence-report results/power_curves_report.json --dry-run
```

The three smoke runs write run directories tagged `smoke_` under `results/`, which the table
producers and the capacity lookups skip. The dry run plans a fleet sweep from the measured
capacities without GPUs; a fleet sweep needs the launcher's four (or eight) GPUs.

## 5. What was measured

Nine single-GPU models (1.2 B to 24 B parameters, three families) and Llama-3.1-70B over four
GPUs; nine request-length distributions (ShareGPT, HumanEval, arXiv summarization, the Azure 2024
chat and code traces, and fixed 128/128 .. 2048/2048 and 6144/1024 prompt/output lengths); H100 80 GB
(bf16) for everything in the main text, and V100 32 GB (fp16), RTX 8000 48 GB (fp16) and RTX 5090
32 GB for the single-GPU model in the appendix. Fleets of 3 large + 1 small replica on four H100s
(Llama-3.1-8B / Llama-3.2-1B and Qwen3-8B / Qwen3-4B) and 5 + 3 on eight. The datacenter-scale
results are a simulation of the analytical model (`lib/power_model.py`, `analyze/datacenter_sim.py`)
at 10,000 GPUs driven by public data.

## 6. Data provenance

`data/datacenter/SOURCES.json` records the URL, access date, size and SHA-256 of every public input
of the datacenter simulation: the Azure LLM inference traces 2024 (CC-BY-4.0; Splitwise, ISCA'24 /
DynamoLLM, HPCA'25) and the EIA-930 hourly demand files. `docs/EXPERIMENTS.md` gives the origin of
every request-length trace under `data/traces/`.
