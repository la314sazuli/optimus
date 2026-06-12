# Scaling optimus for huge servers and large fleets

This is the consolidated operator guide for running optimus at scale — whether
that means a handful of very large, very active Discord servers or a fleet of
thousands of guilds. It ties together the individual scaling levers (each
documented in depth elsewhere) into one place: what to turn, in what order, and
how to watch the result.

optimus is six independent services communicating over JetStream-backed NATS
(see [architecture.md](architecture.md)). Each scales independently, so the goal
is to find the bottleneck, scale only that, and confirm with metrics before
moving on. Almost nothing here is required for a small self-host — the defaults
are correct for a single replica.

For a worked, measured capacity study of one very large server (can optimus run a
single 800,000-member Discord guild?), including index-scaling, burst-absorption,
REST-budget and Postgres-growth numbers and a tuned deployment recipe, see
[capacity.md](capacity.md).

## At a glance

| Pressure | Symptom | Lever | Section |
| -------- | ------- | ----- | ------- |
| Many guilds (>2,500) | Discord rejects the gateway identify | Gateway sharding | [Sharding](#1-gateway-sharding) |
| High image volume | Detection in-flight pinned at max; throughput flat | More detection replicas | [Detection replicas](#2-detection-replicas-the-throughput-bottleneck) |
| Multiple replicas | Effective rate limits multiplied by replica count | Redis rate-limit backend | [Distributed rate limiting](#3-distributed-rate-limiting) |
| Detection backlog | In-flight queue deepens, latency rises | `detection_max_inflight` tuning | [In-flight tuning](#4-in-flight-concurrency) |
| Unbounded data growth | DB grows forever | Retention purge | [Retention](#5-retention) |
| DB connection exhaustion | Pool timeouts at high replica count | Pool sizing | [Connection pooling](#6-connection-pooling) |
| Need visibility | Flying blind under load | Monitoring profile | [Monitoring](#7-monitoring) |
| Need to be paged | Problems found too late | Alerting | [Alerting](#8-alerting) |

## 1. Gateway sharding

The **gateway** holds the only Discord gateway connection and is the one
component whose load grows directly with guild count. Discord *requires* sharding
past ~2,500 guilds, and heavy single servers can saturate one connection well
before that.

Configuration-only via `OPTIMUS_SHARD_COUNT` (fleet-wide total, every replica
must agree) and `OPTIMUS_SHARD_IDS` (which shards this replica runs). Small
deployments should leave both unset and let hikari auto-negotiate a single
shard. The full mechanics — shard assignment, multi-replica splits, `/readyz`
shard checks, and `max_concurrency` identify pacing — are in
[sharding.md](sharding.md).

## 2. Detection replicas (the throughput bottleneck)

Detection is CPU-bound on image decode and is the component that limits
end-to-end throughput. The load harness (`python -m benchmarks.load`, see
[performance-notes.md](performance-notes.md)) characterizes one replica: a 2-vCPU
replica sustains ~7 images/sec single-flight and saturates at ~9–10 images/sec.

**Sizing rule of thumb: budget roughly `~3.5 images/sec per vCPU` per detection
replica.** To handle a target rate `R` images/sec on `C`-vCPU instances:

```
replicas ≈ ceil(R / (3.5 * C))
```

Detection is horizontally scalable: every replica is a competing pull consumer
on the same JetStream stream, so adding replicas adds throughput linearly until
some other resource (DB, Redis, Discord) becomes the limit. The
[`detection`](../src/optimus/services/detection/) service is stateless beyond its
in-memory hash index, which each replica rebuilds independently.

**How to know you need more:** the *Detection in-flight vs max* panel pins at the
configured ceiling and *Pipeline throughput* plateaus. That is the saturation
signature — add replicas, do not just raise `detection_max_inflight` (see below).

## 3. Distributed rate limiting

The default rate limiter (`OPTIMUS_RATELIMIT_BACKEND=memory`) uses per-process
token buckets. That is correct for a single replica, but with N replicas the
effective limit is multiplied by N because each process limits independently.

For multi-replica deployments set:

```bash
OPTIMUS_RATELIMIT_BACKEND=redis
```

This shares one bucket across all replicas via Redis, so effective limits do not
multiply with replica count. If Redis becomes unreachable the limiter falls back
to in-memory buckets (bounding load per replica during the outage) and increments
`optimus_ratelimit_redis_fallback_total`; the shared limit is temporarily
multiplied by replica count again until Redis recovers — strictly safer than
failing requests. Watch the *Ratelimit Redis fallbacks* panel and the
`OptimusRatelimitRedisFallback` alert. Details and the rationale are in
[performance-notes.md](performance-notes.md) (§ distributed rate limiting).

## 4. In-flight concurrency

`OPTIMUS_DETECTION_MAX_INFLIGHT` (default 10) bounds how many messages a
detection replica processes concurrently. Set it **near the replica's vCPU
count**. Because detection is CPU-bound, raising it past the core count does
*not* raise throughput — it only deepens the in-flight queue and adds per-image
latency. Excess buffering belongs in JetStream, not in replica memory, where it
would balloon RSS under a raid. Scale throughput with replicas (§2), not with a
deeper in-flight setting.

## 5. Retention

By default optimus keeps everything: `OPTIMUS_DETECTION_RETENTION_DAYS` is unset,
which disables the scheduler's `retention_purge` job. On a large deployment the
`detections` and appeal tables grow unbounded, so set a positive value to enable
bounded cleanup:

```bash
OPTIMUS_DETECTION_RETENTION_DAYS=90
OPTIMUS_RETENTION_BATCH_SIZE=1000          # rows per DELETE, keeps locks short
OPTIMUS_RETENTION_BATCH_PAUSE_SECONDS=0.5  # pause between batches on huge tables
```

The purge runs in the **scheduler** service on a daily cadence, deleting in
bounded batches so locks and transactions stay short even on huge tables. Purge
volume is observable via `optimus_scheduler_rows_affected_total{task="retention_purge"}`
(the *Retention purges* panel). This operator-level floor is independent of the
per-guild `retention_days` config consumed by the legacy retention job.

> Retention is also a privacy lever: it bounds how long detection metadata and
> appeals are stored. See [privacy-policy-template.md](privacy-policy-template.md).

## 6. Connection pooling

Each service opens a SQLAlchemy async pool: `OPTIMUS_DB_POOL_SIZE` (default 5)
plus `OPTIMUS_DB_MAX_OVERFLOW` (default 10) burst connections. The real cap on
the database is `replicas * (pool_size + max_overflow)`, so at large fleets this
is how you exhaust Postgres connection slots. Size the pool down per replica as
you scale replicas up, or raise Postgres `max_connections` (and front it with a
pooler such as PgBouncer) to match `total_replicas * per_replica_cap`. Postgres
operational guidance is in [operations.md](operations.md).

## 7. Monitoring

An **optional** Prometheus + Grafana stack ships in
[`docker-compose.yml`](../docker-compose.yml) behind the `monitoring` compose
profile. A plain `docker compose up` does **not** start it. Bring it up with:

```bash
docker compose --profile monitoring up -d prometheus grafana
```

- **Prometheus** (`:9090`) scrapes all six services on the compose network at
  `/metrics` (port 8080) every 15s and evaluates the alert rules.
- **Grafana** (`:3000`, default login `admin`/`admin` — change it) is provisioned
  with the Prometheus datasource and the **Optimus Overview** dashboard, no
  manual import. Config lives in [`monitoring/`](../monitoring/).

The dashboard covers pipeline throughput (msgs/s), detection in-flight vs max,
p95 dispatch latency, per-priority queue depth, circuit breaker states, ratelimit
fallbacks, reject/drop counters, and retention purges — i.e. every lever above
has a panel to confirm the change worked.

Every service exposes `/metrics`, `/healthz`, and `/readyz` on
`OPTIMUS_HEALTH_PORT` (default 8080) regardless of the monitoring profile, so you
can point an existing Prometheus at them instead.

## 8. Alerting

The Prometheus rules in [`monitoring/alerts.yml`](../monitoring/alerts.yml) cover
the conditions that warrant operator attention, with conservative `for:` windows
to avoid flapping on deploys and bursts:

| Alert | Fires when |
| ----- | ---------- |
| `OptimusServiceDown` | `up == 0` for a service for >2m |
| `OptimusConsumerStalled` | service up but acking nothing while messages in-flight for >10m (readiness-failing proxy) |
| `OptimusModerationCircuitOpen` | circuit breaker OPEN >5m (actions not reaching Discord) |
| `OptimusModerationQueueDepthHigh` | per-priority queue depth >100 sustained 10m |
| `OptimusRatelimitRedisFallback` | Redis ratelimit fallback active over 5m |
| `OptimusGatewayDropsIncreasing` / `…Ingest…` / `…Detection…` | reject/drop counters rising >1/s over 5m |
| `OptimusBusMessagesDropped` | bus discarding undecodable/poison messages |

Thresholds are starting points — tune per deployment. Each rule carries an inline
comment explaining the intent so you can adjust with context.

## Recommended scale-up order

1. **Shard the gateway** if guild count requires it (§1).
2. **Turn on monitoring** (§7) so the next steps are measurable.
3. **Scale detection replicas** to your target image rate (§2), confirming with
   the in-flight and throughput panels.
4. **Switch the ratelimit backend to Redis** once you run >1 replica of anything
   rate-limited (§3).
5. **Size pools and Postgres** for `replicas * per-replica-cap` (§6).
6. **Enable retention** to bound data growth (§5).
7. **Wire alerts** to your pager (§8).
