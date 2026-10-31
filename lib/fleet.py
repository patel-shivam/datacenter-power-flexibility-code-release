"""Multi-GPU fleet: one worker process per replica, one coordinator.

Process model
-------------
Each replica is a separate OS process with ``CUDA_VISIBLE_DEVICES`` pinned to its own
GPU, holding one vLLM engine and sampling its own GPU's NVML counters. The coordinator
owns the arrival stream, the routing decision, the controller, and the aggregation.

One process per replica rather than one process driving several engines, because the
engine core runs **in-process** here (see ``env.py``): a single Python process driving N
engines would serialize N serving loops behind one GIL, and the resulting slowdown
would be indistinguishable from a capacity limit. That is why the split is structural
rather than a tuning choice.

What crosses the queue
----------------------
Only indices and small telemetry dicts. Each worker generates its **own** request pool
from the shared workload and seed, so a submission is an integer, not a list of
thousands of token ids. Prompts differ per model (different vocabularies) but the
*shapes* are identical across replicas, which is what has to match.

Telemetry is pushed on a fixed cadence and drained by the coordinator without blocking.
A worker that stalls therefore shows up as stale telemetry rather than as a coordinator
that stops driving the fleet.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass, field

TELEMETRY_INTERVAL_S = 0.1

# How long a window each telemetry report averages NVML power over. By default it equals
# the report cadence above, so every tick's power is a 0.1 s average. The window sets how
# smoothed the power trace is, so a like-for-like noise comparison between two runs has to
# match it. Longer is not automatically better: the window is dead time in the control
# loop, and a 0.5 s lag on a 1 s tick is real phase margin spent. The 0.1 s default keeps
# results reproducible; set FLEET_POWER_WINDOW_S to compare.
POWER_WINDOW_S = float(os.environ.get("FLEET_POWER_WINDOW_S", TELEMETRY_INTERVAL_S))


@dataclass
class ReplicaConfig:
    logical_id: int
    gpu_index: int
    role: str                  # "large" | "small"
    model_slug: str
    max_num_seqs: int
    pack_cap: int
    workload: str
    pool_size: int = 4096
    seed: int = 0
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    max_num_batched_tokens: int | None = None


def worker(cfg: ReplicaConfig, cmd_q, res_q) -> None:
    """One replica. Runs until it receives ``{"op": "stop"}``."""
    # PCI_BUS_ID first, so CUDA's ordinals match NVML's. CUDA's default ordering is
    # FASTEST_FIRST, which need not agree with the PCI order NVML and nvidia-smi
    # enumerate in -- and the sampler below addresses its card by NVML index, so the
    # two orderings have to be the same one or telemetry lands on the wrong GPU.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_index)
    from lib import env  # noqa: F401  (bootstrap before vllm)
    from lib import engine as eng
    from lib import power_sampler as ps
    from lib import request_gen as reqlib
    from lib import workloads

    try:
        wl = workloads.get(cfg.workload)
        e = eng.build_engine(cfg.model_slug, max_num_seqs=cfg.max_num_seqs,
                             gpu_memory_utilization=cfg.gpu_memory_utilization,
                             max_model_len=cfg.max_model_len,
                             max_num_batched_tokens=cfg.max_num_batched_tokens)
        vocab, special = eng.tokenizer_info(e)
        rng = random.Random(cfg.seed + cfg.logical_id)
        specs = reqlib.pregenerate(wl, cfg.pool_size, vocab, rng, special)

        # NVML index 0 is NOT our pinned GPU. NVML enumerates every device on the node
        # and is entirely independent of CUDA_VISIBLE_DEVICES, so device_index=0 in
        # every worker would make every replica sample physical GPU 0 -- pool power
        # would read as N copies of one card, which is invisible when the cards are
        # evenly loaded and catastrophic exactly when they are not: idling a replica
        # would free nothing in the numbers, and the packing lever could not appear.
        # With CUDA_DEVICE_ORDER=PCI_BUS_ID set above, NVML index == cfg.gpu_index.
        sampler = ps.PowerSampler(device_index=cfg.gpu_index)
        sampler.start()
        kv = eng.kv_sizing(e)

        # Warm idle: weights resident, nothing running. This replica's P_idle.
        time.sleep(2.0)
        idle = sampler.average_over(time.monotonic() - 1.5, time.monotonic()) or {}
        res_q.put({"type": "ready", "id": cfg.logical_id, "role": cfg.role,
                   "kv": kv, "idle_power_w": idle.get("avg_power_w"),
                   "model": cfg.model_slug, "device": sampler.device_info()})

        live: dict[str, dict] = {}
        n_done = 0
        next_tel = time.monotonic()
        seq = 0
        stopping = False

        while True:
            # --- drain commands without blocking the serving loop
            while True:
                try:
                    msg = cmd_q.get_nowait()
                except Exception:
                    break
                if msg["op"] == "submit":
                    spec = specs[msg["idx"] % len(specs)]
                    rid = f"w{cfg.logical_id}-{seq}"
                    seq += 1
                    e.add_request(rid, {"prompt_token_ids": spec.prompt_token_ids},
                                  reqlib.sampling_params(spec))
                    live[rid] = {"t_submit": msg.get("t_submit", time.monotonic()),
                                 "cls": msg.get("cls", "flexible"),
                                 "P": spec.num_prefill_tokens,
                                 "D": spec.num_decode_tokens,
                                 "t_first": None, "n_tok": 0}
                elif msg["op"] == "stop":
                    stopping = True

            outs = e.step() if (live or e.has_unfinished_requests()) else []
            now = time.monotonic()
            finished = []
            for o in outs:
                rec = live.get(o.request_id)
                if rec is None:
                    continue
                n_tok = len(o.outputs[0].token_ids) if o.outputs else 0
                if n_tok > 0 and rec["t_first"] is None:
                    rec["t_first"] = now
                rec["n_tok"] = max(rec["n_tok"], n_tok)
                if o.finished:
                    n_done += 1
                    ttft = (rec["t_first"] - rec["t_submit"]) * 1e3 if rec["t_first"] else None
                    tbt = (((now - rec["t_first"]) / max(rec["n_tok"] - 1, 1)) * 1e3
                           if rec["t_first"] and rec["n_tok"] > 1 else None)
                    # t_done is on the same CLOCK_MONOTONIC base as the telemetry "t"
                    # below, and CLOCK_MONOTONIC is system-wide on Linux, so a request
                    # can be attributed to the control dwell it completed in. Without
                    # it requests.csv can only be summarised per RUN, and no latency
                    # can be tied to the power level it was served at.
                    finished.append({"ttft_ms": ttft, "tbt_ms": tbt,
                                     "e2e_ms": (now - rec["t_submit"]) * 1e3,
                                     "t_done": now,
                                     "cls": rec["cls"], "P": rec["P"], "D": rec["n_tok"]})
                    live.pop(o.request_id, None)

            if now >= next_tel or finished:
                st = eng.read_state(e)
                pw = sampler.average_over(now - POWER_WINDOW_S, now) or {}
                res_q.put({"type": "tel", "id": cfg.logical_id, "t": now,
                           "power_w": pw.get("avg_power_w"),
                           "mem_used_mib": pw.get("avg_mem_used_mib"),
                           "sm_clock_mhz": pw.get("avg_sm_clock_mhz"),
                           "gpu_util_pct": pw.get("avg_gpu_util_pct"),
                           "running": st.num_running, "waiting": st.num_waiting,
                           "state_ok": st.ok,
                           "kv_usage": st.kv_usage, "n_done": n_done,
                           "finished": finished})
                next_tel = now + TELEMETRY_INTERVAL_S

            if stopping and not live and not e.has_unfinished_requests():
                break
            if not outs and not live:
                time.sleep(0.001)      # nothing to do; do not spin a core

        samples = sampler.stop()
        res_q.put({"type": "done", "id": cfg.logical_id, "n_done": n_done,
                   "n_power_samples": len(samples),
                   "sample_rate_hz": round(sampler.achieved_rate_hz(), 1)})
        sampler.shutdown()
    except Exception as e:  # a worker crash must surface, not hang the coordinator
        import traceback
        res_q.put({"type": "error", "id": cfg.logical_id,
                   "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})


@dataclass
class ReplicaState:
    """Coordinator-side view of one replica."""

    cfg: ReplicaConfig
    power_w: float = 0.0
    running: int = 0
    waiting: int = 0
    kv_usage: float = 0.0
    n_done: int = 0
    inflight: int = 0          # coordinator's own count: submitted minus completed
    last_tel_t: float = 0.0
    idle_power_w: float = 0.0
    ready: bool = False
    device: dict = field(default_factory=dict)   # which card this replica samples
    state_ok: bool = True      # False => running/waiting below are NOT scheduler truth
    kv: dict = field(default_factory=dict)       # kv_sizing(): the pool this replica got


class Fleet:
    """Spawn, drive, and tear down a set of replica workers."""

    def __init__(self, configs: list[ReplicaConfig]):
        ctx = mp.get_context("spawn")
        self.configs = configs
        self.states = {c.logical_id: ReplicaState(c) for c in configs}
        self.cmd_qs = {c.logical_id: ctx.Queue() for c in configs}
        self.res_q = ctx.Queue()
        self.procs = {c.logical_id: ctx.Process(target=worker,
                                                args=(c, self.cmd_qs[c.logical_id], self.res_q),
                                                daemon=True)
                      for c in configs}
        self.completed: list[dict] = []
        self.errors: list[dict] = []

    def start(self, timeout_s: float | None = None) -> None:
        # 1800 s is enough for every pair up to 8B, and not enough for three qwen3-14b
        # replicas (27.5 GiB each) loading concurrently off a shared filesystem, hence
        # the 2700 s default. Startup cost scales with weights x replicas and with
        # whatever else is hitting the filesystem, so this is a wall-clock guess either
        # way; make it settable rather than pretend otherwise.
        if timeout_s is None:
            timeout_s = float(os.environ.get("FLEET_READY_TIMEOUT_S", 2700.0))
        for p in self.procs.values():
            p.start()
        t0 = time.monotonic()
        while not all(s.ready for s in self.states.values()):
            if time.monotonic() - t0 > timeout_s:
                not_ready = [i for i, s in self.states.items() if not s.ready]
                raise TimeoutError(
                    f"replicas {not_ready} did not become ready within {timeout_s:.0f}s. "
                    f"Large models loading in parallel can exceed this; raise "
                    f"FLEET_READY_TIMEOUT_S rather than assuming a hang.")
            self.poll(block_s=1.0)
            if self.errors:
                raise RuntimeError(f"replica failed during startup: {self.errors[0]}")
        self._assert_distinct_devices()

    def _assert_distinct_devices(self) -> None:
        """Every replica must sample its own physical card. Refuse to run otherwise.

        NVML ignores CUDA_VISIBLE_DEVICES, so a worker calling
        ``nvmlDeviceGetHandleByIndex(0)`` samples physical GPU 0 whichever card it runs
        on, and pool power comes back as N copies of one card. Nothing downstream can
        detect that -- the numbers stay plausible, the trace stays smooth, and the error
        vanishes whenever the cards happen to be evenly loaded, which is precisely the
        control arm of the packing experiment. Cheap assertion, whole classes of silent
        garbage.
        """
        seen: dict[str, int] = {}
        for lid, s in sorted(self.states.items()):
            bus = (s.device or {}).get("pci_bus_id")
            if not bus:
                print(f"  WARNING: replica {lid} reported no PCI bus id; cannot verify "
                      f"its power sampler is bound to its own GPU")
                continue
            if bus in seen:
                raise RuntimeError(
                    f"replicas {seen[bus]} and {lid} are both sampling {bus} -- pool "
                    f"power would be a multiple of one card, not the fleet total. "
                    f"NVML indices ignore CUDA_VISIBLE_DEVICES; sample by "
                    f"cfg.gpu_index with CUDA_DEVICE_ORDER=PCI_BUS_ID.")
            seen[bus] = lid
        self._sampler_devices = {lid: bus for bus, lid in seen.items()}
        if seen:
            print("  power samplers: " + ", ".join(
                f"replica {lid}->{bus.split(':', 1)[-1] if ':' in bus else bus}"
                for bus, lid in sorted(seen.items(), key=lambda kv: kv[1])))

    def sampling_provenance(self) -> dict:
        """What every run records about which card each replica sampled.

        With it a reader can verify that a run's power numbers are a fleet total and not
        N copies of one card.
        """
        devs = getattr(self, "_sampler_devices", {}) or {}
        return {
            "distinct_devices_verified": len(set(devs.values())) == len(self.states)
                                         and len(devs) == len(self.states),
            "pci_bus_id_by_replica": {str(k): v for k, v in sorted(devs.items())},
            "n_replicas": len(self.states),
            # Two runs with different smoothing are not comparable on any noise metric,
            # so the smoothing is recorded with every run.
            "power_window_s": POWER_WINDOW_S,
            "telemetry_interval_s": TELEMETRY_INTERVAL_S,
        }

    def poll(self, block_s: float = 0.0) -> None:
        """Drain telemetry. Never blocks the caller's control loop for long."""
        deadline = time.monotonic() + block_s
        while True:
            try:
                msg = self.res_q.get(timeout=0.01) if block_s else self.res_q.get_nowait()
            except Exception:
                if time.monotonic() >= deadline:
                    return
                continue
            t = msg.get("type")
            if t == "ready":
                s = self.states[msg["id"]]
                s.ready = True
                s.idle_power_w = msg.get("idle_power_w") or 0.0
                s.device = msg.get("device") or {}
                s.kv = msg.get("kv") or {}
            elif t == "tel":
                s = self.states[msg["id"]]
                if msg.get("power_w") is not None:
                    s.power_w = msg["power_w"]
                s.running, s.waiting = msg["running"], msg["waiting"]
                s.state_ok = msg.get("state_ok", True)
                s.kv_usage, s.last_tel_t = msg["kv_usage"], msg["t"]
                done_delta = msg["n_done"] - s.n_done
                s.n_done = msg["n_done"]
                s.inflight = max(0, s.inflight - done_delta)
                for f in msg.get("finished", []):
                    f["replica"] = msg["id"]
                    f["role"] = s.cfg.role
                    self.completed.append(f)
            elif t == "error":
                self.errors.append(msg)
            elif t == "done":
                self.states[msg["id"]].ready = False
            if time.monotonic() >= deadline:
                return

    def submit(self, logical_id: int, idx: int, cls: str) -> None:
        self.cmd_qs[logical_id].put({"op": "submit", "idx": idx, "cls": cls,
                                     "t_submit": time.monotonic()})
        self.states[logical_id].inflight += 1

    # --- aggregate observations -------------------------------------------------
    def total_power_w(self) -> float:
        return sum(s.power_w for s in self.states.values())

    def pool_power_w(self, role: str) -> float:
        return sum(s.power_w for s in self.states.values() if s.cfg.role == role)

    def total_running(self) -> int:
        return sum(s.running for s in self.states.values())

    def total_waiting(self) -> int:
        return sum(s.waiting for s in self.states.values())

    def kv_concurrency_ceiling(self, mean_context_tokens: float) -> list[dict]:
        """Per replica: how many sequences of this size the KV pool can actually hold.

        ``max_num_seqs`` is an *intention*. The KV pool is a hard ceiling on the sum of
        context lengths across resident sequences, so at long context a replica runs out
        of KV long before it runs out of sequence slots -- ``engine.kv_sizing`` puts the
        crossover between those two ceilings around 4k tokens.

        This matters beyond a slightly smaller batch. The router packs to ``pack_cap``
        before spilling to the next replica, so a cap the replica can never reach means
        first-fit never spills on time, and the power model's occupancy vectors are
        computed at a fill level that does not exist. Both are silent: the run still
        produces smooth, plausible numbers.
        """
        out = []
        for lid, s in sorted(self.states.items()):
            pool = (s.kv or {}).get("kv_pool_tokens")
            cap = s.cfg.max_num_seqs
            if not pool or mean_context_tokens <= 0:
                out.append({"replica": lid, "role": s.cfg.role, "cap": cap,
                            "kv_ceiling": None, "binds": "unknown"})
                continue
            ceil_ = pool / float(mean_context_tokens)
            out.append({"replica": lid, "role": s.cfg.role, "cap": cap,
                        "kv_pool_tokens": int(pool), "kv_ceiling": round(ceil_, 1),
                        "binds": "kv" if ceil_ < cap else "max_num_seqs"})
        return out

    def states_readable(self) -> bool:
        """True only if every replica's last telemetry was real scheduler truth.

        ``total_running``/``total_waiting`` return 0 both for an empty fleet and for one
        whose scheduler could not be read. Anything that treats 0 as "idle" -- the drain,
        the lit-replica count, a queue-stability check -- must consult this first.
        """
        return all(s.state_ok for s in self.states.values())

    def active_replicas(self, role: str | None = None) -> int:
        return sum(1 for s in self.states.values()
                   if s.running > 0 and (role is None or s.cfg.role == role))

    def kv_usage_max(self) -> float:
        """Highest KV-pool occupancy across replicas, 0..1. A replica sitting near 1.0
        while below its ``max_num_seqs`` is KV-bound, not slot-bound."""
        vals = [s.kv_usage for s in self.states.values() if s.kv_usage is not None]
        return max(vals) if vals else 0.0

    def occupancy(self) -> dict[int, int]:
        return {i: s.running for i, s in self.states.items()}

    def inflight(self, role: str | None = None) -> int:
        return sum(s.inflight for s in self.states.values()
                   if role is None or s.cfg.role == role)

    def idle_power_total_w(self) -> float:
        return sum(s.idle_power_w for s in self.states.values())

    def stop(self, drain_timeout_s: float = 180.0) -> None:
        for q in self.cmd_qs.values():
            q.put({"op": "stop"})
        t0 = time.monotonic()
        while time.monotonic() - t0 < drain_timeout_s:
            self.poll(block_s=0.2)
            if all(not p.is_alive() for p in self.procs.values()):
                break
        for p in self.procs.values():
            if p.is_alive():
                p.terminate()
            p.join(timeout=10)
