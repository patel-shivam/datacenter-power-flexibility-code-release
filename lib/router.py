"""The router: which replica each arriving request goes to.

Two decisions, kept separate on purpose.

**1. Which pool** -- large or small. Requests carry a class:

  ``large_only``  must run on the large pool (quality-critical).
  ``small_only``  must run on the small pool.
  ``flexible``    may run on either. This is the only traffic the knob can move, and
                  ``f`` -- the fraction of flexible traffic sent to the small pool --
                  is the single control variable of the whole system.

**2. Which replica within the pool** -- and this is where the power lever lives.

  ``pack``       First-fit by index: send to the lowest-index replica with fewer than
                 ``pack_cap`` in flight. k-1 replicas end at the cap and the k-th holds
                 the remainder -- N=70 at cap 20 gives [20, 20, 20, 10].
  ``spread``     Equal over the **active set**. The active count is chosen from the load
                 exactly as packing would, k = ceil(N/cap), and the load is then split
                 evenly among those k -- N=70 at cap 20 gives [18, 18, 17, 17]. This is
                 the control arm: the same replicas are lit, only the arrangement
                 differs, so any power difference is the concavity of P(b) and nothing
                 else.
  ``spread_all`` Least-loaded across every replica in the pool, lighting all of them
                 regardless of load. The naive balancer; kept because it is the common
                 default, but it confounds two effects at once.

Why first-fit is the better power knob, at equal k: with packing, k-1 replicas sit at
the cap and the k-th sweeps the whole per-replica power range as the remainder grows,
so the pool total moves continuously between rungs. With equal spreading every active
replica sits in the saturated upper part of the concave curve, so the pool total barely
responds to load and power moves only when k itself changes -- coarse steps of roughly
one replica's busy power.

``pack_cap`` vs ``max_num_seqs``
--------------------------------
``max_num_seqs`` is the engine's hard admission ceiling -- exceed it and requests queue
inside vLLM. ``pack_cap`` is the router's *soft* preference for when to start using the
next replica, and it should sit below ``max_num_seqs``: it is chosen from the latency
sweep (``concurrency_sweep``) as the concurrency at which this model still meets its TBT target,
not from what the engine will physically accept.

One caveat worth carrying: a cap counted in **requests** is shape-dependent. 48 short
requests and 48 long ones are not the same load, the same power, or the same drain
time. Where the workload has heavy length variance, ``pack_cap_tokens`` caps on
expected remaining decode work instead, which is invariant to shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Replica:
    idx: int              # global replica index
    role: str             # "large" | "small"
    pool_pos: int         # position within its own pool
    pack_cap: int
    inflight: int = 0
    inflight_decode_tokens: int = 0   # sum of expected remaining decode work


@dataclass
class Router:
    replicas: list[Replica]
    policy: str = "pack"                  # "pack" | "spread" | "spread_all"
    cap_on: str = "requests"              # "requests" | "tokens"
    _active_target: dict = field(default_factory=dict)

    def pool(self, role: str) -> list[Replica]:
        return [r for r in self.replicas if r.role == role]

    def _load(self, r: Replica) -> int:
        return r.inflight_decode_tokens if self.cap_on == "tokens" else r.inflight

    def set_active_target(self, role: str, k: int) -> None:
        """Set how many replicas of ``role`` the spread policy may use.

        This has to come from the *offered load*, not from instantaneous occupancy.
        Sizing the active set from what is currently in flight makes spreading collapse
        into packing: the newest replica is always the least loaded, so least-loaded
        routing fills it to the cap before the set grows again, and the result is
        [8, 8, 4] where an even split over three replicas would be [7, 7, 6]. The driver
        therefore computes k from the arrival rate (Little's Law gives the steady-state
        N; k = ceil(N/cap)) and sets it here each tick.
        """
        self._active_target[role] = max(1, int(k))

    def active_target(self, role: str) -> int:
        pool = self.pool(role)
        k = self._active_target.get(role)
        if k is None:                    # not set: fall back to the current load
            total = sum(self._load(r) for r in pool)
            cap = max(pool[0].pack_cap, 1) if pool else 1
            k = max(1, -(-int(total) // cap))
        return min(len(pool), max(1, k))

    def pick(self, role: str) -> Replica:
        """Choose a replica within ``role``'s pool."""
        pool = self.pool(role)
        if not pool:
            raise ValueError(f"no replicas with role {role!r}")
        if self.policy == "spread_all":
            return min(pool, key=self._load)
        if self.policy == "spread":
            # Least-loaded within the active set. k comes from the offered load via
            # set_active_target; the +1 is a safety valve so a burst that overruns the
            # planned set spills rather than queueing behind a full one.
            k = self.active_target(role)
            active = pool[:k]
            if all(self._load(r) >= r.pack_cap for r in active) and k < len(pool):
                active = pool[:k + 1]
            return min(active, key=self._load)
        for r in pool:                     # first-fit, by pool position
            if self._load(r) < r.pack_cap:
                return r
        # Everything is at the cap. Never hold the request: holding changes the offered
        # load, which would confound every power number measured against it.
        return min(pool, key=self._load)

    def route(self, req_class: str, f: float, rng) -> Replica:
        """Full decision for one arrival. ``f`` is the flexible->small fraction."""
        if req_class == "large_only":
            role = "large"
        elif req_class == "small_only":
            role = "small"
        elif req_class == "flexible":
            role = "small" if rng.random() < f else "large"
        else:
            raise ValueError(f"unknown request class {req_class!r}")
        return self.pick(role)

    # --- bookkeeping ------------------------------------------------------------
    def admit(self, r: Replica, decode_tokens: int = 0) -> None:
        r.inflight += 1
        r.inflight_decode_tokens += decode_tokens

    def complete(self, r: Replica, decode_tokens: int = 0) -> None:
        r.inflight = max(0, r.inflight - 1)
        r.inflight_decode_tokens = max(0, r.inflight_decode_tokens - decode_tokens)

    # --- observation ------------------------------------------------------------
    def occupancy(self) -> list[int]:
        return [r.inflight for r in self.replicas]

    def active_count(self, role: str | None = None) -> int:
        rs = self.replicas if role is None else self.pool(role)
        return sum(1 for r in rs if r.inflight > 0)


def build_router(n_large: int, n_small: int, pack_cap_large: int, pack_cap_small: int,
                 policy: str = "pack", cap_on: str = "requests") -> Router:
    reps: list[Replica] = []
    for i in range(n_large):
        reps.append(Replica(len(reps), "large", i, pack_cap_large))
    for i in range(n_small):
        reps.append(Replica(len(reps), "small", i, pack_cap_small))
    return Router(reps, policy=policy, cap_on=cap_on)


class ClassSplit:
    """Traffic mix by class. Only ``flexible`` is routable, so it bounds the knob."""

    def __init__(self, large_only: float, small_only: float, flexible: float):
        total = large_only + small_only + flexible
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"class split must sum to 1, got {total}")
        self.large_only, self.small_only, self.flexible = large_only, small_only, flexible

    def draw(self, rng) -> str:
        u = rng.random()
        if u < self.large_only:
            return "large_only"
        if u < self.large_only + self.small_only:
            return "small_only"
        return "flexible"

    def __repr__(self) -> str:
        return (f"ClassSplit(large_only={self.large_only:.2f}, "
                f"small_only={self.small_only:.2f}, flexible={self.flexible:.2f})")


def _selfcheck() -> None:
    import random

    rng = random.Random(0)

    # --- first-fit fills replica 0 before touching replica 1
    r = build_router(3, 1, pack_cap_large=8, pack_cap_small=8)
    for _ in range(20):
        rep = r.route("large_only", 0.0, rng)
        r.admit(rep)
    assert r.occupancy()[:3] == [8, 8, 4], r.occupancy()
    assert r.active_count("large") == 3

    # --- spread lights the SAME number of replicas as pack, but fills them evenly
    r2 = build_router(3, 1, 8, 8, policy="spread")
    r2.set_active_target("large", 3)          # k chosen from the load: ceil(20/8) = 3
    for _ in range(20):
        r2.admit(r2.route("large_only", 0.0, rng))
    occ2 = r2.occupancy()[:3]
    assert sum(occ2) == 20 and max(occ2) - min(occ2) <= 1, occ2
    assert r2.active_count("large") == r.active_count("large"), (occ2, r.occupancy()[:3])

    # --- spread_all lights every replica regardless of load
    r2b = build_router(3, 1, 64, 64, policy="spread_all")
    for _ in range(6):
        r2b.admit(r2b.route("large_only", 0.0, rng))
    assert min(r2b.occupancy()[:3]) > 0, "spread_all never idles a replica"

    # --- overflow past every cap spills rather than holding
    r3 = build_router(2, 0, 4, 4)
    for _ in range(20):
        r3.admit(r3.route("large_only", 0.0, rng))
    assert sum(r3.occupancy()) == 20, r3.occupancy()

    # --- the knob moves flexible traffic and only flexible traffic
    for f, want_small in ((0.0, False), (1.0, True)):
        r4 = build_router(2, 2, 64, 64)
        for _ in range(200):
            r4.admit(r4.route("flexible", f, rng))
        small = sum(x.inflight for x in r4.pool("small"))
        assert (small > 0) == want_small, (f, small)
    r5 = build_router(2, 2, 64, 64)
    for _ in range(200):
        r5.admit(r5.route("large_only", 1.0, rng))
    assert sum(x.inflight for x in r5.pool("small")) == 0, "large_only must ignore f"

    # --- token-capped packing tracks work, not request count
    r6 = build_router(2, 0, 100, 100, cap_on="tokens")
    r6.admit(r6.pool("large")[0], decode_tokens=90)
    nxt = r6.route("large_only", 0.0, rng)
    assert nxt.pool_pos == 0, "90 < cap 100, still fits"
    r6.admit(nxt, decode_tokens=90)
    assert r6.route("large_only", 0.0, rng).pool_pos == 1, "180 > cap, must spill"

    split = ClassSplit(0.3, 0.2, 0.5)
    counts = {}
    for _ in range(20000):
        c = split.draw(rng)
        counts[c] = counts.get(c, 0) + 1
    assert abs(counts["flexible"] / 20000 - 0.5) < 0.02, counts

    print("router selfcheck OK")
    print(f"  first-fit occupancy after 20 arrivals, cap 8: {r.occupancy()[:3]} "
          f"({r.active_count('large')} of 3 replicas lit)")
    print(f"  spread (equal over active set), same arrivals: {r2.occupancy()[:3]} "
          f"({r2.active_count('large')} of 3 replicas lit -- same k as pack)")
    print(f"  spread_all after only 6 arrivals:             {r2b.occupancy()[:3]} "
          f"({r2b.active_count('large')} of 3 lit -- lights everything)")
    print(f"  class mix drawn: {counts}")


if __name__ == "__main__":  # python -m lib.<module>
    _selfcheck()
