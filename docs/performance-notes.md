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

## Scaling

### Distributed rate limiting for multi-replica deployments (`scale/redis-ratelimit`)
- **Location:** `src/optimus/core/ratelimit.py` (`RedisRateLimiter`,
  `build_rate_limiter`); `src/optimus/core/config.py` (`RateLimitBackend`,
  `ratelimit_backend`); service wiring in `ingest`, `interactions`, `moderation`.
- **Problem:** the in-memory token buckets are per-process, so running N replicas
  of a service multiplies the effective limit by N — a "20 fetches/guild/s" cap
  becomes 20·N once you scale out.
- **Fix:** `RedisRateLimiter` evaluates a token bucket inside a single Redis Lua
  `EVAL`. The read-modify-write is atomic on the server, so concurrent
  acquisitions from any number of replicas share one bucket with no read/write
  race (`test_redis_limiter_concurrent_acquire_is_atomic` asserts exactly
  `capacity` allows out of a concurrent burst). The bucket key carries a TTL of
  `ceil(capacity/refill)+1`s so idle keys expire server-side.
- **Backend selection:** `Settings.ratelimit_backend` is `memory` (default) or
  `redis`, wired through the shared `build_rate_limiter`. **Defaulting to
  `memory` means single-node self-hosters see zero behavioural change**; only
  multi-replica operators opt into `redis`.
- **Graceful degradation:** the Redis backend is built with an in-memory
  `fallback`. If Redis errors at runtime (connection loss, timeout, script
  error) `acquire` **falls back to a process-local bucket rather than failing
  open** (which would allow unlimited traffic) **or crashing the request path.**
  This bounds load per replica during the outage; the only cost is that the
  shared limit is temporarily multiplied by replica count — strictly safer than
  fail-open. Each fallback increments `optimus_ratelimit_redis_fallback_total`
  and logs `ratelimit_redis_fallback`, so an outage is observable. Covered by
  `test_redis_limiter_falls_back_to_in_memory_on_error`. With no fallback wired
  the error is re-raised (`test_redis_limiter_reraises_when_no_fallback`), so the
  policy is always explicit at the call site.

### Idempotency & back-pressure under JetStream redelivery (`scale/idempotency-backpressure`)
- **Location:** `src/optimus/bus/nats.py` (`EventBus.publish`, `EventBus.consume`,
  `ensure_stream`); `src/optimus/services/detection/service.py` (`_persist`,
  consumer wiring); `src/optimus/services/{ingest,moderation}/service.py` (publish
  msg-ids, consumer wiring); `src/optimus/core/config.py` (bus settings).
- **Problem (raid / image-flood on a huge server):** JetStream is at-least-once.
  If a detection replica acks slowly (CPU-bound decode under a flood) past
  `ack_wait`, or naks on a transient error, the same `image_fetched` message is
  **redelivered**. Two failure modes follow: (1) the redelivery could re-run the
  pipeline and double-act (a second ban, a duplicate detection row, a duplicate
  `verdict`/`action_result` on the stream); (2) the consumer fetched a fixed
  batch and processed it with no explicit in-flight ceiling, and `ack_wait` was
  hardcoded at 30s — under a flood, queued messages in a batch could exceed
  `ack_wait` and trigger a **spurious redelivery storm**, compounding load.

- **Idempotency — three layers, so redelivery is a no-op:**
  1. *Consumer-side guard (primary).* `DetectionWorker.handle` already claims a
     Redis `SET NX` on the deterministic `idempotency_key`
     (`optimus:idem:{message_id}:{attachment_id}`) before doing any work; a
     redelivered image finds the key claimed and returns `None`, so **no second
     verdict is emitted and no second action runs**. The moderation
     `ActionExecutor` has its own independent `SET NX` keyed per
     `(idempotency_key, action)` as a backstop on the action path.
  2. *Publisher dedup (defense in depth).* `EventBus.publish` now accepts a
     `msg_id` sent as the JetStream `Nats-Msg-Id` header; `ensure_stream` sets a
     `duplicate_window` (default 2h) so a republish of the same id is collapsed
     **server-side** before it reaches any consumer. Wired on the three events
     that carry a business key: `image_fetched` (ingest), `verdict` (detection),
     `action_result` (moderation). The id is namespaced by subject so the same
     key on two subjects never cross-dedups.
  3. *DB unique constraint (authority).* `detections.idempotency_key` is `UNIQUE`.
     `DetectionService._persist` does a read-check first, but two replicas can
     race past it; the loser's INSERT now runs inside a **savepoint**
     (`begin_nested`) and a raised `IntegrityError` is swallowed as a no-op. The
     savepoint scopes the rollback to just the failed row, so the surrounding
     transaction (and the row the winner committed) is untouched — and critically
     the consumer does **not** nak a message whose row already exists (which would
     redeliver forever).

- **Back-pressure — bounded in-flight, JetStream buffers the surplus:**
  `EventBus.consume` takes a `max_inflight` (settings-driven
  `detection_max_inflight`, default **10**, informed by the load-harness ceiling
  above: a 2 vCPU replica saturates near ~10 img/s, so deeper in-flight only adds
  latency and redelivery risk, not throughput). It is enforced two ways: an
  `asyncio.Semaphore` caps concurrently-processing handlers per replica, and the
  same value is set as the consumer's `max_ack_pending` so the **server** stops
  delivering once a replica holds that many unacked. The pull `fetch` is clamped
  to the spare in-flight budget each loop, so a slow replica leaves messages
  **buffered in JetStream rather than ballooning its own memory**. `ack_wait` is
  now configurable (`detection_ack_wait_seconds`, default **60s** — comfortably
  above the worst-case decode+hash of `max_frames` frames) so slow processing
  buffers instead of tripping redelivery; `max_deliver` is configurable too.
  `optimus_bus_messages_inflight` (gauge) makes the live in-flight depth
  observable per subject.

- **Tested by:** `tests/unit/test_bus.py` (msg-id header wiring + namespacing,
  ack/nak/term dispatch, `max_inflight` bounds concurrent handlers under a 50-msg
  burst, `max_ack_pending`/`ack_wait` reach the consumer config) and
  `tests/integration/test_pipeline.py`
  (`test_redelivered_image_does_not_double_act`,
  `test_redelivered_verdict_dedups_on_msg_id`, `test_persist_swallows_unique_race`).

### Image payload hardening for huge-server scale (`scale/payload-hardening`)
- **Location:** `src/optimus/core/config.py` (`ingest_max_inline_bytes`,
  `gateway_max_attachments`, inline-cap validator); `src/optimus/services/gateway/extract.py`
  (`build_events` attachment cap + `optimus_gateway_images_dropped_total`);
  `src/optimus/services/ingest/worker.py` (inline-size cap + `oversize_inline`
  rejection); `src/optimus/services/detection/worker.py` (base64-decode guard +
  `optimus_detection_payload_rejected_total`).
- **The flow (audited).** Gateway publishes `message_image.v1` carrying only the
  attachment/embed **URL** (no bytes). Ingest fetches once through the streaming,
  SSRF-pinned `fetch_image` (DNS pinned, redirects re-validated, body read in
  64 KiB chunks and **aborted mid-stream** the instant it passes `ingest_max_bytes`,
  Content-Length pre-checked, header allowlist + magic-byte sniff). It then
  publishes `image_fetched.v1` with the validated bytes **inline as base64** plus
  a SHA-256. Detection base64-decodes and decodes in a **sandboxed subprocess**
  (CPU/AS/FSIZE rlimits, Pillow `MAX_IMAGE_PIXELS` pixel cap enforced *before* a
  full decode, parent-side wall-clock timeout); any decode/limit failure is a
  `NON_DECISION`. So raw bytes do **not** flow on the gateway hop, but they **do**
  ride inline on `image_fetched` (base64, ~+33%).
- **Why keep bytes inline rather than re-fetch from the CDN URL in detection.**
  Discord CDN URLs are now **signed and time-limited** (`ex`/`is`/`hm` query
  params, ~24h). Under a raid the `image_fetched` queue can buffer deep in
  JetStream (back-pressure is *designed* to let it); if detection re-fetched from
  the URL, a message that waited out the queue could find its URL **expired** →
  unfetchable → forced non-decision, i.e. a raid could make us silently stop
  inspecting exactly when it matters most. Re-fetching would also double outbound
  bandwidth and re-expose the SSRF surface in a second service. The pipeline
  deliberately fetches **once** in ingest; we keep that and instead **hard-bound
  the inline payload** so it can never balloon NATS or replica memory.
- **The bounds added (all settings-driven, sensible defaults):**
  - *Inline size cap* — `ingest_max_inline_bytes` (default **8 MiB**, validated
    `<= ingest_max_bytes`). The fetcher may stream up to `ingest_max_bytes`, but
    anything larger than the inline cap is **dropped in the ingest worker**
    (counted `optimus_ingest_images_rejected_total{reason="oversize_inline"}`,
    returns `None` → the message is **acked**, never nak-looped). This is the
    bound that actually caps what a single `image_fetched` message puts on the
    stream and into a detection replica's memory.
  - *Per-message attachment cap* — `gateway_max_attachments` (default **10**).
    `build_events` stops after that many inspectable images per message and counts
    the rest (`optimus_gateway_images_dropped_total{reason="attachment_cap"}`), so
    one message cannot fan out an unbounded number of fetch/decode jobs.
  - *Download size / timeout / content-type / pixel-count / redirect caps* already
    existed (`ingest_max_bytes`, `fetch_image` `total_timeout`, `ALLOWED_CONTENT_TYPES`
    + magic-byte sniff, `DecodeLimits.max_image_pixels`, `ingest_max_redirects`);
    this pass verified them and folds them into one documented contract.
- **Resolve, never nak-loop.** Every oversize/timeout/bomb/malformed-payload case
  resolves the message (ack + reason-labeled metric/log), consistent with the
  PR #14 redelivery/idempotency design. The detection worker now base64-decodes
  with `validate=True` inside a guard: a corrupt inline payload becomes a
  `NON_DECISION` (counted `optimus_detection_payload_rejected_total{reason="decode"}`)
  instead of raising — a raise would nak and redeliver the same poison until
  `max_deliver`, pure wasted work under a flood.
- **Tested by:** `tests/unit/test_gateway.py` (attachment cap counts dropped
  extras, spans attachments+content URLs, no cap when unset),
  `tests/unit/test_ingest_worker.py` (oversize-inline drop + boundary publish),
  `tests/unit/test_detection.py` (malformed-base64 resolves not raises,
  decompression-bomb pixel-cap → non-decision, decode-timeout → non-decision,
  normal image still flows), `tests/unit/test_config_and_health.py` (defaults +
  inline-cap-must-not-exceed-download-cap validator), plus the existing
  `tests/integration/test_fetcher.py` streaming/size-cap/timeout coverage and
  `tests/unit/test_decoder.py` pixel-cap/timeout coverage.

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
