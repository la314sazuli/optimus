# Performance & async-correctness notes

Findings from the Cycle 6 performance/async pass. Each entry is either **fixed**,
**documented** (real but deferred), or **ruled out** (investigated, no change).

## Throughput baseline

Measured by `python -m benchmarks.load`, a throughput load-test harness that
pushes N synthetic images (the same deterministic corpus the accuracy benchmark
uses) through the real `DetectionWorker.handle` with a configurable number
concurrently in flight. It exercises the full production path — the sandboxed
decode *subprocess*, perceptual hashing (both offloaded via `asyncio.to_thread`),
BK-tree candidate gather, and the ensemble vote — over in-process fakes for the
index/whitelist/sensitivity/idempotency hooks (no NATS/Redis/Postgres). Queue
arrival is simulated with a saturated asyncio job pool, so each row is a
sustained, always-busy worst case. See `benchmarks/load/`.

**Machine caveat:** numbers below were captured in the CI-class sandbox this repo
develops on — **2 vCPU, 8 GB RAM**, Linux. They characterize *one* detection
replica on that hardware; absolute images/sec scales with core count (the decode
subprocess is CPU-bound), so production replicas on larger instances will differ.
Use these for relative reasoning and onboarding sizing, not as an SLO.

Command: `python -m benchmarks.load --concurrency 1 2 4 8 --images 200`
(corpus: 66 distinct images, cycled to 200 per level).

| Concurrency | Images | Wall (s) | Images/s | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) | max (ms) | Peak RSS (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 200 | 28.77 | 7.0 | 141.8 | 161.2 | 176.2 | 143.8 | 181.1 | 116.8 |
| 2 | 200 | 21.40 | 9.3 | 212.2 | 252.8 | 266.0 | 213.7 | 285.9 | 127.2 |
| 4 | 200 | 21.46 | 9.3 | 434.8 | 495.6 | 509.2 | 427.4 | 538.7 | 127.2 |
| 8 | 200 | 20.67 | 9.7 | 781.7 | 1191.5 | 1300.0 | 818.8 | 1311.3 | 127.2 |

**Reading the table.** Sustained throughput is **~7 images/sec single-flight and
saturates at ~9–10 images/sec by concurrency 2** — exactly the 2-vCPU ceiling,
since each image's dominant cost is the CPU-bound decode subprocess (~140 ms
single-flight, the p50 at concurrency 1). Pushing concurrency past the core count
does **not** raise throughput; it only deepens the in-flight queue, so per-image
end-to-end latency climbs ~linearly (p50 141 ms → 782 ms from c=1 to c=8) while
images/sec stays flat. Peak RSS is modest and stable (~117–127 MB) and does not
grow with concurrency, so a replica is CPU- not memory-bound. **Sizing rule of
thumb:** budget roughly `~3.5 images/sec per vCPU` per detection replica and set
the in-flight concurrency near the replica's core count — higher only trades
latency for nothing.

## Fixed

### Blocking decode + hashing on the detection event loop
- **Location:** `src/optimus/services/detection/worker.py` `DetectionWorker.handle`
- **Issue:** `decode()` (a synchronous `subprocess.run` that blocks up to the
  decode wall timeout, default 5s) and `all_frame_hashes()` (numpy/Python
  perceptual hashing over up to `max_frames` frames) ran directly on the event
  loop. The detection bus consumer (`bus.consume`) processes messages
  sequentially and shares the loop with the health server and NATS heartbeats,
  so a single slow/large image stalled all detection progress and could delay
  liveness/readiness responses.
- **Fix:** the decode+hash block is offloaded with `asyncio.to_thread` via a new
  `DetectionWorker._decode_and_hash` helper. `decode()` stays synchronous (it is
  unit-tested that way and used at one async call site), so the change is
  surgical. Covered by `test_worker_offloads_decode_and_hash_to_thread`.

### `IndexManager` per-guild index cache is now LRU-bounded (Cycle 14)
- **Location:** `src/optimus/services/detection/index.py` `IndexManager`
- **Was:** `_guilds: dict[int, HashIndex]` retained one BK-tree index per guild
  ever queried, never evicting. Growth was bounded by the number of guilds the
  bot is in (not per-message/per-user churn), and each index is expensive to
  rebuild (a Postgres query per guild) — a legitimate hot-path cache, but
  unbounded for very large fleets.
- **Fix:** the cache is now an `OrderedDict` LRU bounded by
  `Settings.detection_guild_index_cap` (default 1024; `None` = unbounded). Each
  read touches the entry to most-recent; once the cap is exceeded the
  least-recently-used guild index is dropped and rebuilt on demand the next time
  it is queried (the rebuild-on-demand path is unchanged). Eviction runs
  synchronously *after* a freshly built index is stored and moved to most-recent,
  so within the single event loop it can never evict the index a caller is about
  to return. Each eviction increments `optimus_detection_guild_index_evicted_total`
  and logs `guild_index_evicted`. Covered by
  `test_guild_index_lru_evicts_least_recently_used`,
  `test_guild_index_invalidate_respects_lru_cap`, and
  `test_guild_index_unbounded_by_default`.

### In-memory rate-limiter fallback is now swept in the ingest service (Cycle 14)
- **Location:** `src/optimus/services/ingest/service.py` `build_worker`;
  `src/optimus/core/ratelimit.py` `InMemoryRateLimiter`
- **Was:** `InMemoryRateLimiter` exposed `evict_idle` to bound its `_buckets`
  map, but nothing called it for the ingest fallback. Keys are `guild:{id}`, so
  growth is bounded by guild count rather than user churn, and this path is only
  reached in a degraded (no-Redis) mode.
- **Fix:** `InMemoryRateLimiter` gained an optional `sweep_interval`; when set,
  `acquire` opportunistically runs `evict_idle` at most once per interval (a
  time-gated compare on the single event loop, so the sweep never races an
  in-flight `acquire`). The ingest fallback now passes
  `Settings.ingest_inmemory_sweep_seconds` (default 300s). `sweep_interval=None`
  (the default) preserves the old behavior for every other caller. Covered by
  `test_in_memory_sweep_interval_triggers_eviction_on_use` and
  `test_in_memory_sweep_disabled_by_default`.

### `mod_circuit_*` settings are now wired to the moderation breaker (Cycle 14)
- **Location:** `src/optimus/services/moderation/service.py` `build_coordinator`
- **Was:** `mod_circuit_failure_threshold` / `mod_circuit_recovery_seconds`
  existed in `Settings` but `ActionExecutor` was constructed without a `breaker`,
  so it fell back to `CircuitBreaker()`'s own defaults (5 / 30.0). Those defaults
  happened to match the config defaults, so behavior was correct but the settings
  were inert.
- **Fix:** `build_coordinator` now constructs the `CircuitBreaker` from
  `Settings.mod_circuit_failure_threshold` and `mod_circuit_recovery_seconds` and
  injects it. Defaults are preserved exactly (5 / 30.0). Covered by
  `test_build_coordinator_wires_circuit_settings`.

## Documented (real, deferred — too invasive for a single-fix pass)

### Unused `_use_embedding` flag on the detection worker
- **Location:** `src/optimus/services/detection/worker.py` (`use_embedding` /
  `_use_embedding`) and `src/optimus/hashing/embedding.py`
- **Observation:** the worker stores `_use_embedding` but never consults it;
  `embedding.embed()` (blocking ONNX inference) is not called from any async
  path. So embedding is not currently a blocking-on-loop risk. If embedding
  confirmation is wired into `handle` later, `embed()` MUST be offloaded the
  same way as decode/hash (it runs synchronous `session.run`).

## Ruled out (investigated, no change needed)

- **`hashing/bktree.py` query/add:** sub-linear (triangle-inequality pruning),
  pure-Python int Hamming ops, only a handful of lookups per frame. Not a
  meaningful loop-blocker relative to decode/hash.
- **`hashing/perceptual.py` (called inside the offloaded helper):** CPU-bound
  but now runs off-loop via the `_decode_and_hash` offload above.
- **`ingest/fetcher.py`:** fully async aiohttp with a total `ClientTimeout`,
  bounded chunked streaming with a hard size cap, and per-hop `ClientSession`
  created/closed via `async with` (intentional for SSRF re-pinning). No session
  leak, no missing timeout.
- **`services/detection/swarm.py`, `core/ratelimit.py` (Redis), `moderation/cooldown.py`:**
  single atomic Redis round trips (Lua `eval` / `SET NX EX`); no N+1 round trips,
  no in-memory growth. TTLs bound Redis-side state.
- **`core/guild_config.py` `GuildConfigCache`:** Redis cache with TTL + DB
  fallback; no per-message recompute, no unbounded process-local map.
- **`services/gateway/extract.py`:** URL regex compiled once at import; per-message
  dedup set is request-scoped and bounded by the message. No O(n^2).
- **`core/config.py` `get_settings`, `i18n/catalog.py`:** `lru_cache`d — parsed
  once, not per message.
- **Fire-and-forget tasks:** every `asyncio.create_task` in the service
  entrypoints (scheduler, moderation, ingest, detection) keeps a strong
  reference (held in a list and awaited via `gather`, or tracked in
  `GatewayService._inflight` and drained on shutdown). No GC'd-task risk.
- **`gather` usage:** `return_exceptions` is used only where appropriate
  (`GatewayService.drain` over best-effort in-flight publishes). The supervisor
  `gather`s in `_amain` intentionally propagate so a dead consumer triggers
  shutdown.
