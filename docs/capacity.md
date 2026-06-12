# Capacity: can optimus run a single 800,000-member server?

**Verdict: yes — with the right configuration, and with one caveat to watch.**

A single 800k-member Discord server fits comfortably within optimus on a small
deployment, provided you (1) run NATS with a raised `max_payload` (now the
shipped default, see [Exp 3](#experiment-3-nats-payload-limit)) and (2) scale
detection replicas to your sustained image rate. The index-scaling axis that was
previously the headline limit is no longer one: the phash index now uses
multi-index hashing (MIH), whose query cost stays low-ms into the hundreds of
thousands of entries ([Exp 1](#experiment-1-index-scaling)).

This document records *measured* evidence, not estimates. The harnesses live in
[`benchmarks/index_scaling.py`](../benchmarks/index_scaling.py) and
[`benchmarks/burst_absorption.py`](../benchmarks/burst_absorption.py) and are
reproducible.

## Load model

The numbers below assume one very large, very active server:

| Quantity | Assumption |
| -------- | ---------- |
| Messages/day | up to ~1,000,000 (~12 msg/s avg, ~100 msg/s peak) |
| Fraction with images | 5–10% → ~1–8 img/s sustained |
| Raid burst | 50–100 img/s for minutes |
| Discord REST global | 50 req/s (500 req/s for verified big bots) |

## Summary of findings

| Axis | Result | Limit / action |
| ---- | ------ | -------------- |
| Image burst absorption | **PASS** | in-flight bounded, no redelivery storm; surplus buffers in JetStream |
| NATS payload | **BUG FOUND + FIXED** | 8 MiB inline images failed publish on the 1 MiB NATS default; now validated at startup + compose raised to 12 MiB |
| Index scaling (query latency) | **FIXED** | switched BK-tree → multi-index hashing (MIH); radius-18 query now **2.8 ms p50 at 10k, 24 ms at 100k, 134 ms at 500k** (was 7 / 118 / 623 ms) on uniform-random hashes — 2.6×/4.9×/4.6× faster, and far better on real clustered hashes |
| Index memory / build | **FINE** | MIH ~406 B/entry (~194 MB at 500k, *lower* than the BK-tree's ~238 MB); cold build 4.7 s at 500k |
| REST budget under raid | **FINE with tuning** | per-guild bucket, not the 50 req/s global, is the binding limit; PROTECT actions never dropped |
| Postgres growth | **FINE** | ~31–62 MB/day of detections at this scale; inserts trivial; enable retention to bound it |

---

## Experiment 1: index scaling

**The axis that was the headline limit — now fixed.** The throughput harness seeds
the index from a tiny corpus, so per-guild/global index *size* was never
characterized. A single mature deployment's **global** scam-hash index
(cross-guild promoted hashes) grows over time and is queried on every image
alongside the guild index
([`matcher.py`](../src/optimus/services/detection/matcher.py)).

The original BK-tree degraded toward a linear scan at the production candidate
radius 18 and was the documented scaling limit (~118 ms p50 at 100k, ~623 ms at
500k). It has been **replaced by multi-index hashing** (MIH; Norouzi, Punjani &
Fleet, CVPR 2012, [`hashing/mih.py`](../src/optimus/hashing/mih.py)). MIH splits
each 64-bit phash into `m=4` disjoint 16-bit substrings with one exact-match hash
table per substring. By the pigeonhole principle, two hashes within total Hamming
distance `r` must agree to within `floor(r/m)` on at least one substring — so a
radius-18 query enumerates each substring's `floor(18/4)=4` Hamming ball
(`sum(C(16,k), k=0..4)=2,517` keys), unions the bucket ids, and verifies the true
64-bit distance. Results are **identical to a linear scan** (exact, not
approximate — proven by property tests in
[`tests/unit/test_mih.py`](../tests/unit/test_mih.py) and
[`tests/unit/test_index_equivalence.py`](../tests/unit/test_index_equivalence.py)),
so matcher semantics, mirror siblings, and `hash_id`/`campaign_id` mapping are
unchanged.

Measured with [`benchmarks/index_scaling.py`](../benchmarks/index_scaling.py),
building a real `HashIndex` from N synthetic 64-bit hash sets (half carrying a
mirror sibling, as the production builder produces) and running 1,000–2,000
`candidates()` queries at the **production candidate radius 18**
([`DEFAULT_CANDIDATE_RADIUS`](../src/optimus/services/detection/matcher.py)):

| Entries | Index nodes | Build (s) | Index RSS | B/entry | q p50 | q p95 | q p99 | mean cands |
| ------- | ----------- | --------- | --------- | ------- | ----- | ----- | ----- | ---------- |
| 10,000  | 15,022      | 0.05      | 12.0 MB   | 1,260   | 2.8 ms | 3.7 ms | 5.1 ms | 5.0 |
| 100,000 | 149,701     | 1.06      | 59.7 MB   | 626     | 23.7 ms | 28.6 ms | 36.4 ms | 46.8 |
| 500,000 | 749,656     | 4.74      | 193.6 MB  | 406     | 134.4 ms | 275.9 ms | 323.1 ms | 231.3 |

**Before vs after** (radius 18, p50 query latency):

| Entries | BK-tree p50 | MIH p50 | Speed-up |
| ------- | ----------- | ------- | -------- |
| 10,000  | 7.1 ms      | 2.8 ms  | 2.6×     |
| 100,000 | 117.8 ms    | 23.7 ms | 5.0×     |
| 500,000 | 623.0 ms    | 134.4 ms| 4.6×     |

(Synthetic hashes are uniform random — the **worst case** for MIH: uniform
substrings spread evenly across the per-table buckets, maximizing both the
candidate union each query must verify *and* the spurious near-collisions
verified — `mean cands` itself grows with N because random 64-bit hashes
genuinely collide within radius 18 more often as the corpus grows. Real
scam-campaign hashes cluster into far fewer buckets, so production latency is
lower than these upper bounds. The 10k B/entry figure is small-sample RSS noise;
the asymptotic cost is ~406 B/entry, *below* the BK-tree's ~500.)

**Why m=4, not m=8.** Tuning at radius 18 by measurement: m=8 (8-bit substrings,
`floor(18/8)=2` ball of 37 keys) wins at 10k but **loses badly at scale** — its
256-bucket tables are ~50× more populated than m=4's 65,536-bucket tables, so each
of the 8 probes drags in a far larger candidate set to verify. Measured p50 at
100k: m=4 **23.7 ms** vs m=8 **59 ms**; at 500k: m=4 **134 ms** vs m=8 **391 ms**.
m=4 is the production choice ([`DEFAULT_SUBSTRING_COUNT`](../src/optimus/hashing/mih.py)).

**Memory and build improved.** MIH is ~406 B/entry (the four substring tables cost
more per node than tree nodes, but storing only one full 64-bit value per id keeps
it under the BK-tree); 500k entries is ~194 MB (vs ~238 MB) and rebuilds cold in
**4.7 s** (vs 6.6 s). Well under the 2 KB/entry budget.

**What this means for 800k members.** Index size is *not* member count — it is the
number of distinct scam-image hashes. A single guild's own index stays small. The
**global** index is the one that grows, and it is shared across the fleet. At 100k
global entries the per-image global lookup is now ~24 ms p50 (was ~118 ms); even
500k is ~134 ms, comfortably within a single replica's budget at sustained image
rates. The previous advice to keep the global index aggressively pruned is no
longer load-bearing — the lookup scales sub-linearly in the regime that matters
(clustered campaign hashes) and stays low-ms on the worst case well past 100k.

---

## Experiment 2: burst absorption

A single detection replica sustains ~10 img/s; a raid offers 50–100 img/s. Does
the surplus melt the replica, or buffer safely?

Measured with [`benchmarks/burst_absorption.py`](../benchmarks/burst_absorption.py),
which drives the **real** [`EventBus.consume`](../src/optimus/bus/nats.py)
pull-consumer loop against a faithful fake JetStream reproducing the two
behaviours that decide the outcome:

* **Pull-fetch delivery.** A message only starts its `ack_wait` redelivery timer
  when the consumer `fetch()`-es it. The loop clamps each fetch to the spare
  in-flight budget, so surplus stays buffered *un-delivered* in the stream — and
  an un-delivered message cannot time out.
* **`ack_wait` redelivery.** A delivered-but-unacked message past `ack_wait` is
  redelivered. The harness counts these to flag a redelivery storm.

Result of a 100 img/s offered burst into a 10 img/s replica
(`detection_max_inflight=10`, `ack_wait=60s`):

| Metric | Observed | Expectation |
| ------ | -------- | ----------- |
| peak in-flight | **10** | ≤ `max_inflight` (10) — bounded ✓ |
| redeliveries | **0** | no redelivery storm ✓ |
| peak backlog | surplus, buffered in JetStream | lives in the stream, not replica RAM ✓ |
| drain after burst | matches `(offered−capacity)·duration / capacity` | analytical ✓ |

**No redelivery storm.** The default `ack_wait=60s` is far above the ~1 s a
fetched image spends in-handler at burst depth, so it never fires. Critically,
the **un-fetched backlog is shielded from `ack_wait`** by pull-consumer semantics:
the consume loop never fetches beyond its spare in-flight budget
(`want = min(batch, max_inflight − len(tasks))`), so queued-but-unfetched messages
have no running timer and cannot spuriously redeliver. This is also proven by the
unit test [`test_consume_bounds_inflight_under_burst`](../tests/unit/test_bus.py)
(`assert all(size <= max_inflight for size in sub.fetch_sizes)`).

**Conclusion:** bursts are absorbed safely. Memory stays bounded at the in-flight
ceiling; the surplus lives in JetStream's bounded stream; drain time is
predictable. The lever for faster drain is more detection replicas (§ scaling.md).

---

## Experiment 3: NATS payload limit

**A real bug — found, reproduced, and fixed in this audit.**

Images ride **inline as base64** inside `image_fetched.v1` events. The ingest
default `OPTIMUS_INGEST_MAX_INLINE_BYTES` is **8 MiB**, but base64 inflates raw
bytes by ~4/3 plus a JSON/header envelope, so an 8 MiB image becomes ~11 MiB on
the NATS wire. The NATS server default `max_payload` is **1 MiB**. A raw image
larger than ~0.73 MiB would therefore **fail to publish and never be scanned** —
silently, from the operator's point of view.

**Fix (robust, both layers):**

1. **Compose default raised.** [`docker-compose.yml`](../docker-compose.yml) now
   starts NATS with `--max_payload 12582912` (12 MiB), covering the 8 MiB inline
   cap with headroom. Documented inline and in [`.env.example`](../.env.example).
2. **Fail-fast startup validation.** The bus now captures the server's negotiated
   `max_payload` on connect and the ingest service calls
   `bus.validate_inline_capacity(settings.ingest_max_inline_bytes)` at startup
   ([`ingest/service.py`](../src/optimus/services/ingest/service.py)). If the
   configured inline cap would exceed the server's wire limit, it raises
   `PayloadLimitError` and refuses to start — so a mismatch can never silently
   drop images again. If the server limit is unknown, it skips rather than guesses.

Covered by tests in [`tests/unit/test_bus.py`](../tests/unit/test_bus.py)
(`inline_wire_size`, capacity pass/reject/skip, and `connect` capturing
`max_payload`).

**Self-hosters running their own NATS** must set `max_payload` to at least
`ceil(OPTIMUS_INGEST_MAX_INLINE_BYTES · 4/3) + 4 KiB`. The startup check enforces
this for you.

---

## Experiment 4: REST budget under raid

Discord's global REST limit is 50 req/s (500 for verified big bots). Does a raid
on one 800k server blow it?

**The global limit is not the binding constraint for a single guild — the
per-guild action rate limiter is.** Moderation actions go through a token bucket
keyed `modact:{guild_id}` ([`actions.py`](../src/optimus/services/moderation/actions.py))
with **capacity 5, refill 1/s** (`mod_action_rate_capacity=5.0`,
`mod_action_rate_refill=1.0`). For one guild that is a burst of 5 then a sustained
**1 action/s**. Discord's 50 req/s global is enforced transparently by hikari's
own REST client; optimus adds no separate global throttle.

A raid generating, say, 100 protect actions/min (~1.67/s) offers faster than the
1/s drain, so the queue grows ~0.67/s. This is handled gracefully:

* The [`PriorityDispatcher`](../src/optimus/services/moderation/priority.py)
  classifies DELETE/BAN/KICK as **PROTECT** (priority 0). PROTECT actions are
  **always admitted past `mod_dispatch_max_queue`** (1000) — only droppable
  NOTIFY/COURTESY are rejected when full.
* An **aging guard** (`mod_dispatch_aging_seconds=5.0`) promotes waiting items one
  class per interval, so nothing starves; PROTECT is rescored to the front before
  each pop.
* `mod_dispatch_concurrency=4` workers drain the heap.

So during a raid, member-protecting actions (deletes, bans) are never dropped and
are always served first; lower-priority notifications shed load gracefully. The
**recommendation for an 800k server** is to raise the per-guild bucket so protect
actions keep pace with a sustained raid:

```
OPTIMUS_MOD_ACTION_RATE_CAPACITY=20     # bigger burst headroom
OPTIMUS_MOD_ACTION_RATE_REFILL=5        # 5 sustained actions/s for the one big guild
OPTIMUS_MOD_DISPATCH_CONCURRENCY=8
```

A verified big bot (500 req/s global) has ample room for this. Stay well under the
global ceiling: even 5/s for one guild plus fleet traffic is far below 50/s. If
you run multiple replicas, switch the ratelimit backend to Redis so the bucket is
shared and the effective limit is not multiplied (see [scaling.md](scaling.md) §3).

---

## Experiment 5: Postgres growth

One `Detection` row is written **per scanned image, including CLEAN verdicts**
([`detection/service.py`](../src/optimus/services/detection/service.py)) — one
INSERT per image (no bulk batching; JetStream `batch` is fetch-level only).

Row width is ~410 B (BIGINT ids, a small `distances` JSON, short varchars, a
128-char unique `idempotency_key`, `created_at`). With heap + index overhead
(pk, `guild_id`, `ix_detections_created_at`, the unique key) reckon ~1.5×.

| Load | Images/day | Detections/day (with indexes) | 30-day | 1-year |
| ---- | ---------- | ----------------------------- | ------ | ------ |
| 1M msgs/day @ 5% img | 50,000 | ~31 MB | ~0.9 GB | ~11 GB |
| 1M msgs/day @ 10% img | 100,000 | ~62 MB | ~1.8 GB | ~22 GB |

**Insert throughput is a non-issue:** 1–8 INSERT/s sustained is trivial for
Postgres. **Growth is bounded only if you enable retention** —
`OPTIMUS_DETECTION_RETENTION_DAYS` is **unset (disabled) by default**, so a
self-host keeps everything forever. For an 800k server set a retention window;
the deployment-wide purge batches deletes (`retention_batch_size=1000`,
`retention_batch_pause_seconds=0.5`) to avoid long locks. The `Appeal` table
(~60 B/row) and `ModAction` (~250 B/row, admin actions only) grow far slower and
are not a concern.

---

## Recommended deployment recipe for one 800k server

A single very large, very active server does **not** need a large fleet. Start
here and scale the one axis that turns red:

| Setting | Value | Why |
| ------- | ----- | --- |
| Detection replicas | **2–3** | one replica clears ~10 img/s; 2–3 gives raid drain headroom and lets you survive a replica restart mid-burst |
| Gateway shards | auto (1) unless other guilds push you past ~2,500 | one server needs no sharding for guild count; heavy single-server gateway load is the trigger ([sharding.md](sharding.md)) |
| `OPTIMUS_INGEST_MAX_INLINE_BYTES` | 8 MiB (default) | covered by the 12 MiB NATS `max_payload` now shipped |
| NATS `--max_payload` | **12 MiB** (shipped default) | inline base64 + envelope; do not lower below `ceil(inline·4/3)+4KiB` |
| `OPTIMUS_DETECTION_MAX_INFLIGHT` | 10–20 | bounds per-replica concurrency; raise if a replica's CPU is underused at burst |
| `OPTIMUS_MOD_ACTION_RATE_CAPACITY` / `_REFILL` | **20 / 5** | let protect actions keep pace with a sustained raid on the one big guild |
| `OPTIMUS_MOD_DISPATCH_CONCURRENCY` | 8 | drain the priority heap faster under raid |
| `OPTIMUS_RATELIMIT_BACKEND` | **redis** | required once you run >1 replica so limits aren't multiplied |
| `OPTIMUS_DETECTION_RETENTION_DAYS` | **set it** (e.g. 90) | growth is otherwise unbounded; ~62 MB/day at peak |
| Monitoring profile | **on** | watch detection in-flight, queue depth, and the global-index lookup latency |

## Known limits

* **Global hash-index query latency is no longer the headline scaling limit.** The
  phash index uses multi-index hashing (MIH); at candidate radius 18 the lookup is
  ~24 ms p50 at 100k and ~134 ms p50 at 500k on uniform-random hashes (was ~118 ms
  / ~623 ms with the BK-tree), and lower on real clustered campaign hashes. Pruning
  the **global** index is now an optimization, not a requirement; add detection
  replicas to parallelize lookups under heavy sustained load. The candidate radius
  18 (border-crop recall, see [detection-eval.md](detection-eval.md)) is unchanged
  — MIH gives the recall *and* the speed, so it is no longer a precision/latency
  trade.
* **Per-guild action rate, not the Discord global limit, throttles one big
  guild.** Tune `mod_action_rate_*` for the server; PROTECT actions are never
  dropped regardless.
* **Retention is off by default.** Detections accumulate one row per image
  (including CLEAN) forever until you enable a retention window.
* These index numbers are a single-replica, worst-case (uniform-random hash)
  characterization. Real clustered campaign hashes land in fewer MIH buckets and
  have fewer spurious near-collisions to verify, so they are faster; measure your
  own global index with [`benchmarks/index_scaling.py`](../benchmarks/index_scaling.py).
