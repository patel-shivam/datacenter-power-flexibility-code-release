# The paper's figures: file, data, experiment

`python make_figures.py` draws every figure below from the CSV tables under `results/tables/`; nothing
in the figure code touches a raw run. Each figure is written as PDF and PNG to
`figures/main/` or `figures/appendix/`.

How the tables come about is in README.md §3; the experiments behind them in
`docs/EXPERIMENTS.md`.

## Main text

### Single-GPU model (H100 80 GB; four core models x six tasks)

Tables: `results/tables/single_gpu/` — `single_gpu_levels.csv` (every closed-loop level),
`single_gpu_pb_curves.csv` (the fitted P(b) on a b grid), `single_gpu_calibration.csv` (t0, t1,
s per cell), `single_gpu_points.csv` (every predicted-vs-measured point). Produced by
`analyze/plot_single_gpu_model.py` from the concurrency sweeps, the capacity bisections and the
Poisson-load runs (plus the KV-ceiling sweeps for the appendix table).

| file | what it shows |
|---|---|
| `single_gpu_power_vs_resident_requests_h100-80gb` | per-replica power P(b): the measured mean power of every closed-loop level against its achieved resident count, and the fitted profile (seven-form fit, every third level held out); warm idle ~115 W, a jump at the first request, concave beyond it |
| `single_gpu_step_time_vs_resident_requests_h100-80gb` | the step-time line tau(b) = t0 + t1 b per task against the measured mean time between tokens of the steady-state levels; right: t1 by task (ShareGPT, HumanEval, 128/128 at 0.03-0.10 ms per resident request, arXiv at 0.10-0.41) |
| `single_gpu_generation_time_vs_resident_requests_h100-80gb` | the residence law T(b) = s tau(b) per task (s = mean output + mean prompt / 2048) against the measured mean request residence time of every steady-state level, log axes |
| `single_gpu_resident_requests_vs_arrival_rate_h100-80gb` | the occupancy fixed point b(lambda) = lambda s t0 / (1 - lambda s t1), clipped at the cap, against the arrival rate, with every stable capacity probe and Poisson-load phase at its realised rate; dotted: the measured capacity |
| `single_gpu_completion_rate_vs_resident_requests_h100-80gb` | the request throughput mu(b) = b / T(b) per task against the measured closed-loop levels (completions corrected for the window-edge counting rule) |

### Fleet routing (3 large + 1 small replica on four H100s; 5 + 3 on eight)

Tables: `results/tables/fleet_routing/` — `fleet_meta.csv` (each fleet's composition, measured
capacity C_fleet and u* = n_S C_S / C_fleet), `fleet_probes.csv` (every held probe: offered
load, f, measured and predicted power and lit counts), `fleet_model_curves.csv` (the model's
P(f) at every load), `fleet_model_band_grid.csv` (the model's window and span on a fine load
grid), `fleet_window_band.csv` (model vs measured window and band per load),
`fleet_band_measured.csv` (the holdable band, with its p5-p95 whiskers `lo` / `hi` and its standard deviation `sd`,
from the 1 Hz power of its two end probes). Produced by
`analyze/plot_fleet_routing_figures.py` from the fleet routing sweeps (`slurm/fleet_sweep_4gpu.sbatch`,
`slurm/fleet_sweep_8gpu.sbatch`), the measured capacities and the single-GPU calibration
(`results/power_curves.json`).

| file | what it shows |
|---|---|
| `fleet_feasible_window_and_power_band_4gpu_pairs` | for the Llama 1B/8B and Qwen 4B/8B pairs on ShareGPT: the sustainable window of the routing fraction f against load (left) and the total-power band over it (right), the model's prediction (shaded) and the measured range at every load swept (bars, over the probes whose queue held) |
| `fleet_power_vs_routing_fraction_llama_sharegpt` | the Llama pair on ShareGPT at normalized arrival rates 0.1-0.9: total power against the fraction of queries sent to the small pool, measured (solid; feasible probes whose queue held) and model (dashed); right: the holdable power range against load, measured (whiskers: +-1 standard deviation, from the end probes' 1 Hz power) and model |

### Datacenter scale (a simulation of a 10,000-GPU site with the analytical model)

Tables: `results/tables/datacenter/` — `week_by_share[_70b24b_chat|_code].csv` (every 5-minute
step of the week at every flexible share), `event_by_share*.csv` and `event_summary_by_share*.csv`
(the event day), `step_response*.csv` / `step_summary*.csv` (the measured 1-s drop and return
shape, from the open-loop drop runs), `availability_by_share*.csv` and `week_lending*.csv` (GPUs
leaving), `week.csv` (the reference site's load). Produced by `analyze/datacenter_sim.py` from the
public inputs in `data/datacenter/` (Azure LLM inference trace 2024: arrival counts per 5 minutes;
EIA-930 hourly demand of PJM and ERCOT, for the curtailment hours), the single-GPU calibration, the
measured capacities and the fleet probes.
The three sites: the reference (Llama-3.1-8B -> Llama-3.2-1B at ShareGPT request lengths), and the
second site (Llama-3.1-70B over four GPUs -> Mistral-24B) at the Azure chat and code request
lengths, each on its trace's own week.

| file | what it shows |
|---|---|
| `datacenter_week_flexibility_8b_1b_sharegpt` | a week of real traffic: facility power in normal operation (black) and the floor routing reaches within seconds at every share of the queries flexible (10 % cool to 100 % warm); a duration curve (at least y MW can be shed during x % of the week); the power that can be shed by hour of day |
| `datacenter_week_flexibility_70b_24b_azure_chat` | the same, second site, Azure chat lengths |
| `datacenter_week_flexibility_70b_24b_azure_code` | the same, second site, Azure code lengths |
| `datacenter_event_day_8b_1b_sharegpt` | the week's busiest day with a 16:00-20:00 (Eastern) event at every flexible share; the event's first and last minute at 1 s resolution (the measured drop and return scaled to each share's cut) |
| `datacenter_event_day_70b_24b_azure_chat` | the same, second site, Azure chat lengths (the 1-s shape is the measured 8B -> 1B shape with the drain stretched by the 70B's projected drain) |
| `datacenter_event_day_70b_24b_azure_code` | the same, second site, Azure code lengths |
| `datacenter_gpus_leaving_8b_1b_sharegpt` | week-mean power reduction against the inference GPUs still available (traffic unchanged, GPUs leave both pools alike; dotted once some busy 5-minute steps go unserved); the load on the inference GPUs through the busiest weekday with every GPU and with 40 % of them training at night; the power reduction through that day |
| `datacenter_gpus_leaving_70b_24b_azure_chat` | the same, second site, Azure chat lengths |
| `datacenter_gpus_leaving_70b_24b_azure_code` | the same, second site, Azure code lengths |

## Appendix

| file | what it shows | data |
|---|---|---|
| `single_gpu_parity_all_gpus` | five predicted quantities (power at b, TBT, mean request residence time, request throughput, request concurrency) measured against predicted on the H100, RTX 5090, RTX 8000 and V100, with the MAPE per GPU | `results/tables/single_gpu/single_gpu_points.csv` (the per-GPU runs under `results/<gpu>/`) |
| `fleet_feasible_window_and_power_band_grid` | the feasible window and power band against load for the five other fleets with a full load sweep: Qwen 4B/8B on ShareGPT and 1024/1024, Llama 1B/8B on arXiv and 1024/1024, and the 8-GPU Llama 3S/5L fleet on ShareGPT | fleet tables |
| `fleet_power_vs_routing_fraction_grid` | total power against the routing fraction at every load, and the power range against load, for the same five fleets | fleet tables |
| `grid_programs_response_timescale` | what grid products and large-load agreements require (time to full response, and how long it is held) against routing's measured 90 % drop time by task | `results/tables/datacenter/grid_programs.csv` (curated from the programs' public rules, sources inside `data/datacenter/grid_programs.csv`), `step_summary.csv` |
