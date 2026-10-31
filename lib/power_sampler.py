"""NVML power/clock/utilization/memory sampler, plus the exact energy counter.

Sampling rate
-------------
40 Hz by default, and that is a measured choice, not a guess. NVML's power reading
refreshes at a hardware-clocked ~10 Hz under load (median inter-change period 100.3
ms) and ~20 Hz idle. At 435 Hz, 95.7% of consecutive reads are byte-identical. 40 Hz
keeps 4x margin over the loaded refresh and captures every distinct hardware reading;
a 1 kHz and a 40 Hz sampler run against the same load agreed to 0.002 W mean / 0.12 W
max while the 40 Hz trace was 24.7x smaller. Oversampling buys nothing and costs
memory in a trace that is held for the life of a run.

Two ways to get energy, and they are not equivalent
---------------------------------------------------
``average_over`` integrates the sampled power. ``energy_over`` reads NVML's
``TotalEnergyConsumption`` counter, which the driver accumulates continuously in mJ.
The counter is exact over the window; the integral is a 10 Hz reconstruction of a
signal with real structure above 10 Hz, so it mis-estimates energy across short or
transient windows (a prefill spike lasting 40 ms can fall entirely between two power
samples). Prefer ``energy_over`` for anything per-request or per-step; the power
trace is for *shape*, the counter is for *totals*.

``average_over`` locates its window by binary search. That is not a micro-optimization:
one sampler stays open for a whole multi-phase run while being polled ~10x/s, so a scan
of the whole trace costs time that grows with elapsed run time rather than with window
length -- 9.5 ms/call at 500 s of trace, 58 ms at 3000 s. At 10 calls/s that is a third
of a serving loop spent on telemetry late in a run, with the lock held, which reads as a
capacity collapse that worsens phase after phase.

One instance over several GPUs
------------------------------
A tensor-parallel instance is one replica drawing power on N cards, so its power is the
SUM over them, sampled in one sweep per tick (``device_indices``; ``for_model`` picks
them from the model spec). Energy is the sum of the N counters, memory used is the
instance total, clock and utilization are the means, and each card's own power rides
along (``avg_power_w_gpu<k>`` in every window, ``power_w_gpu<k>`` in the trace), so a
card that sits out of the instance shows up as a flat column rather than vanishing into
the sum. A one-card sampler reads its one card and carries no per-card columns.
"""

from __future__ import annotations

import bisect
import threading
import time
from dataclasses import dataclass

import pynvml

DEFAULT_SAMPLE_INTERVAL_S = 0.025  # 40 Hz


@dataclass
class PowerSample:
    timestamp: float           # time.monotonic()
    power_watts: float
    sm_clock_mhz: int
    gpu_util_pct: int          # fraction of time SMs were busy
    mem_util_pct: int          # fraction of time memory was read/written
    mem_used_mib: float
    energy_mj: float           # NVML lifetime energy counter, mJ (0 if unsupported)
    # Each card's power when one instance spans several (their sum is power_watts);
    # empty for a one-card sampler.
    per_device_w: tuple = ()


class PowerSampler:
    def __init__(self, device_index: int | None = None,
                 sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
                 device_indices: "list[int] | tuple[int, ...] | None" = None):
        # Several cards only when asked for (a tensor-parallel instance, see for_model);
        # a one-element list is the one-card sampler.
        if device_indices is not None and len(device_indices) > 1:
            if device_index is not None:
                raise ValueError("pass device_index or device_indices, not both")
            indices = [int(i) for i in device_indices]
            if len(set(indices)) != len(indices):
                raise ValueError(f"duplicate NVML indices {indices}")
        else:
            if device_indices:
                device_index = int(device_indices[0])
            # Default to the physical card this process is actually pinned to. NVML
            # ignores CUDA_VISIBLE_DEVICES, so a hard-coded 0 samples the wrong GPU
            # the moment more than one run shares a node -- see hardware.nvml_index.
            if device_index is None:
                from lib import hardware
                device_index = hardware.nvml_index()
            indices = [device_index]
        self.device_indices = indices
        self.device_index = indices[0]
        self.sample_interval_s = sample_interval_s
        pynvml.nvmlInit()
        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices]
        self._handle = self._handles[0]
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._samples: list[PowerSample] = []
        self._lock = threading.Lock()
        self._energy_supported = True
        try:
            for h in self._handles:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        except Exception:
            self._energy_supported = False
        if len(self._handles) > 1:
            # The N-copies-of-one-card failure, refused at construction: an instance total
            # summed over one card read N times looks like a plausible instance.
            buses = [self._device_facts(h, i)["pci_bus_id"]
                     for h, i in zip(self._handles, indices)]
            if None in buses or len(set(buses)) != len(buses):
                raise RuntimeError(f"NVML indices {indices} do not resolve to distinct cards "
                                   f"(bus ids {buses})")

    @property
    def num_devices(self) -> int:
        return len(self._handles)

    # --- device facts -----------------------------------------------------------
    def device_info(self) -> dict:
        info = self._device_facts(self._handle, self.device_index)
        if len(self._handles) == 1:
            return info
        devs = [self._device_facts(h, i) for h, i in zip(self._handles, self.device_indices)]
        try:
            count = pynvml.nvmlDeviceGetCount()
        except Exception:
            count = None
        return {**info, "num_devices": len(devs), "device_indices": list(self.device_indices),
                "pci_bus_ids": [d["pci_bus_id"] for d in devs],
                "power_limit_total_w": sum(d["power_limit_w"] for d in devs),
                "total_memory_mib_instance": sum(d["total_memory_mib"] for d in devs),
                "nvml_device_count": count, "devices": devs}

    def device_memory_used_mib(self) -> list[float]:
        """Memory in use on each sampled card right now, in device_indices order."""
        return [pynvml.nvmlDeviceGetMemoryInfo(h).used / 1048576.0 for h in self._handles]

    def _device_facts(self, h, index: int) -> dict:
        def _try(fn, *a, default=None):
            try:
                return fn(h, *a)
            except Exception:
                return default
        name = _try(pynvml.nvmlDeviceGetName)
        pci = _try(pynvml.nvmlDeviceGetPciInfo)
        bus = getattr(pci, "busId", None) if pci else None
        return {
            "name": name.decode() if isinstance(name, bytes) else name,
            # Which physical card this sampler is actually bound to. NVML enumerates
            # every device on the node and ignores CUDA_VISIBLE_DEVICES, so a process
            # pinned to one GPU can silently sample another; the bus id is the only
            # identifier that settles which one, and callers running one sampler per
            # replica must check these are distinct.
            "device_index": index,
            "pci_bus_id": bus.decode() if isinstance(bus, bytes) else bus,
            "power_limit_w": (_try(pynvml.nvmlDeviceGetPowerManagementLimit, default=0) or 0) / 1000.0,
            "power_limit_max_w": (_try(pynvml.nvmlDeviceGetPowerManagementLimitConstraints,
                                       default=(0, 0)) or (0, 0))[1] / 1000.0,
            "total_memory_mib": (_try(pynvml.nvmlDeviceGetMemoryInfo).total / 1048576.0
                                 if _try(pynvml.nvmlDeviceGetMemoryInfo) else 0.0),
            "sm_clock_max_mhz": _try(pynvml.nvmlDeviceGetMaxClockInfo, pynvml.NVML_CLOCK_SM,
                                     default=0),
            "energy_counter_supported": self._energy_supported,
        }

    # --- sampling ---------------------------------------------------------------
    def _poll(self) -> PowerSample:
        if len(self._handles) > 1:
            return self._poll_instance()
        h = self._handle
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        energy = 0.0
        if self._energy_supported:
            try:
                energy = float(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
            except Exception:
                self._energy_supported = False
        return PowerSample(
            timestamp=time.monotonic(),
            power_watts=pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
            sm_clock_mhz=pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            gpu_util_pct=util.gpu,
            mem_util_pct=util.memory,
            mem_used_mib=mem.used / 1048576.0,
            energy_mj=energy,
        )

    def _poll_instance(self) -> PowerSample:
        """One sweep over every card of the instance, stamped at the sweep's midpoint."""
        t0 = time.monotonic()
        pw, clk, gu, mu = [], [], [], []
        mem = energy = 0.0
        for h in self._handles:
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem += pynvml.nvmlDeviceGetMemoryInfo(h).used / 1048576.0
            if self._energy_supported:
                try:
                    energy += float(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
                except Exception:
                    self._energy_supported = False
            pw.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
            clk.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
            gu.append(util.gpu)
            mu.append(util.memory)
        n = len(pw)
        return PowerSample(
            timestamp=0.5 * (t0 + time.monotonic()),
            power_watts=sum(pw),
            sm_clock_mhz=int(round(sum(clk) / n)),
            gpu_util_pct=int(round(sum(gu) / n)),
            mem_util_pct=int(round(sum(mu) / n)),
            mem_used_mib=mem,
            # A partial sum is not an energy: once any counter fails, report none.
            energy_mj=energy if self._energy_supported else 0.0,
            per_device_w=tuple(pw),
        )

    def _run(self) -> None:
        nxt = time.monotonic()
        while not self._stop.is_set():
            s = self._poll()
            with self._lock:
                self._samples.append(s)
            nxt += self.sample_interval_s
            delay = nxt - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                nxt = time.monotonic()   # fell behind; resync rather than catch-up spin

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("PowerSampler already started")
        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[PowerSample]:
        if self._thread is None:
            raise RuntimeError("PowerSampler was not started")
        self._stop.set()
        self._thread.join()
        self._thread = None
        with self._lock:
            return list(self._samples)

    def samples(self) -> list[PowerSample]:
        with self._lock:
            return list(self._samples)

    # --- queries ----------------------------------------------------------------
    def _window(self, t0: float, t1: float) -> list[PowerSample]:
        with self._lock:
            s = self._samples
            lo = bisect.bisect_left(s, t0, key=lambda x: x.timestamp)
            hi = bisect.bisect_left(s, t1, key=lambda x: x.timestamp)
            return s[lo:hi]

    def average_over(self, t0: float, t1: float) -> dict | None:
        """Mean of every channel over ``[t0, t1)``. None if the window holds no sample."""
        w = self._window(t0, t1)
        if not w:
            return None
        n = len(w)
        out = {
            "avg_power_w": sum(x.power_watts for x in w) / n,
            "peak_power_w": max(x.power_watts for x in w),
            "min_power_w": min(x.power_watts for x in w),
            "avg_sm_clock_mhz": sum(x.sm_clock_mhz for x in w) / n,
            "avg_gpu_util_pct": sum(x.gpu_util_pct for x in w) / n,
            "avg_mem_util_pct": sum(x.mem_util_pct for x in w) / n,
            "avg_mem_used_mib": sum(x.mem_used_mib for x in w) / n,
            "peak_mem_used_mib": max(x.mem_used_mib for x in w),
            "num_samples": n,
        }
        if len(self._handles) > 1:
            out["num_devices"] = len(self._handles)
            for k in range(len(self._handles)):
                out[f"avg_power_w_gpu{k}"] = sum(x.per_device_w[k] for x in w) / n
        return out

    def energy_over(self, t0: float, t1: float) -> float | None:
        """Joules over ``[t0, t1)`` from NVML's own counter -- exact, not integrated.

        None if the counter is unsupported or the window has < 2 samples."""
        if not self._energy_supported:
            return None
        w = self._window(t0, t1)
        if len(w) < 2:
            return None
        return (w[-1].energy_mj - w[0].energy_mj) / 1000.0

    def percentiles_over(self, t0: float, t1: float, qs=(5, 50, 95)) -> dict | None:
        """Power percentiles. The p5-p95 spread is the excursion metric: a slow limit
        cycle has a small step-to-step sigma and a large excursion, so sigma (or
        std(diff)) hides exactly the oscillation worth measuring."""
        w = self._window(t0, t1)
        if not w:
            return None
        vals = sorted(x.power_watts for x in w)
        out = {}
        for q in qs:
            k = min(len(vals) - 1, max(0, int(round((q / 100.0) * (len(vals) - 1)))))
            out[f"p{q}_power_w"] = vals[k]
        out["excursion_p5_p95_w"] = out.get("p95_power_w", 0) - out.get("p5_power_w", 0)
        return out

    def achieved_rate_hz(self) -> float:
        with self._lock:
            s = self._samples
        if len(s) < 2:
            return 0.0
        span = s[-1].timestamp - s[0].timestamp
        return (len(s) - 1) / span if span > 0 else 0.0

    def shutdown(self) -> None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def for_model(spec) -> PowerSampler:
    """The sampler for ONE instance of ``spec``: its own card, or every card it spans."""
    tp = int(getattr(spec, "tensor_parallel_size", 1) or 1)
    if tp <= 1:
        return PowerSampler()  # physical index resolved from CUDA_VISIBLE_DEVICES
    from lib import hardware
    return PowerSampler(device_indices=hardware.nvml_indices(tp))


def assert_instance_resident(sampler: PowerSampler, spec, min_frac: float = 0.5) -> list[float]:
    """Refuse to measure unless every sampled card holds its share of the weights.

    Distinct bus ids prove the sampler reads N different cards, not that they are the N
    the instance runs on: NVML ignores CUDA_VISIBLE_DEVICES, so a wrong index-to-card map
    would sum the right number of the wrong GPUs, at idle and plausibly. A card holding
    less than ``min_frac`` of its weight shard after the engine is built is not part of
    this instance. Returns each card's MiB in use.
    """
    used = sampler.device_memory_used_mib()
    tp = int(getattr(spec, "tensor_parallel_size", 1) or 1)
    need = min_frac * spec.weights_gib * 1024.0 / tp
    bad = [(i, round(u)) for i, u in zip(sampler.device_indices, used) if u < need]
    if len(used) != tp or bad:
        raise RuntimeError(f"power sampler is not on this instance's {tp} GPUs: NVML "
                           f"{sampler.device_indices} hold {[round(u) for u in used]} MiB, "
                           f"need >= {need:.0f} MiB each (bad: {bad})")
    return used


def write_trace_csv(samples: list[PowerSample], path) -> None:
    """Write the raw trace. A ``.gz`` suffix compresses it, which is the default.

    A 40 Hz trace over a multi-hour run is tens of MB of numeric text per GPU -- 71 MB
    across one nine-model sweep -- and it compresses about 15x. The trace is kept because
    the *shape* of the power signal is the evidence for anything about excursion, but it
    does not need to be stored raw.
    """
    import csv
    import gzip

    path = str(path)
    opener = (lambda: gzip.open(path, "wt", newline="")) if path.endswith(".gz") \
        else (lambda: open(path, "w", newline=""))
    n_dev = len(samples[0].per_device_w) if samples else 0
    with opener() as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "power_w", "sm_clock_mhz", "gpu_util_pct",
                    "mem_util_pct", "mem_used_mib", "energy_mj"]
                   + [f"power_w_gpu{k}" for k in range(n_dev)])
        for s in samples:
            w.writerow([f"{s.timestamp:.4f}", f"{s.power_watts:.2f}", s.sm_clock_mhz,
                        s.gpu_util_pct, s.mem_util_pct, f"{s.mem_used_mib:.1f}",
                        f"{s.energy_mj:.0f}"]
                       + [f"{p:.2f}" for p in s.per_device_w[:n_dev]])
