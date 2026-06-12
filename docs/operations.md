# Database operations

Operational guidance for running optimus's Postgres at scale: data retention,
connection pooling, and backups. See [architecture.md](architecture.md) for the
wider system picture and [sharding.md](sharding.md) for the gateway tier.

## Data retention

optimus stores a row per detection (plus cascading appeals and evidence
references) and an append-only `mod_actions` audit log. Left unbounded these
tables grow forever, which eventually hurts query latency, backup size, and
disk usage. Two independent mechanisms trim them:

1. **Per-guild retention** (`enforce_retention`, the legacy `retention` job).
   Each guild's `retention_days` config (default 30) is honoured: detections,
   appeals, and mod-actions older than that window are deleted on every run.
   This is per-tenant policy and always on.
2. **Deployment-wide purge** (`purge_old_data`, the `retention_purge` job).
   An operator-level floor governed by `OPTIMUS_DETECTION_RETENTION_DAYS`.
   **Disabled by default (`None`)** so self-hosters keep everything; set a
   positive day count to enable it.

### The deployment-wide purge

When enabled, the `retention_purge` scheduler job deletes detections and appeals
created more than `detection_retention_days` ago, in **bounded batches**:

- Each batch is a single `DELETE` of at most `OPTIMUS_RETENTION_BATCH_SIZE` rows
  (default 1000), run in its own transaction. Short transactions keep locks
  brief and avoid bloating the WAL or blocking foreground writes on huge tables.
- Between batches the job sleeps `OPTIMUS_RETENTION_BATCH_PAUSE_SECONDS`
  (default 0.5) to yield to live traffic and give autovacuum room to reclaim
  the freed tuples.
- **FK order is respected:** appeals are purged before detections. Appeals that
  hang off a still-retained detection are removed on their own `created_at`
  schedule; appeals and evidence under a purged detection cascade away via
  `ON DELETE CASCADE`.
- Each run emits the Prometheus counter
  `optimus_scheduler_rows_affected_total{task="retention_purge"}` (rows purged)
  and logs `retention_purge_complete` with `rows_purged` and `retention_days`.

The `created_at < cutoff` scan is backed by `ix_detections_created_at` and
`ix_appeals_created_at` (migration 0005). On a very large live table, build
these with `CREATE INDEX CONCURRENTLY` manually to avoid the build-time lock.

| Setting | Default | Meaning |
| --- | --- | --- |
| `OPTIMUS_DETECTION_RETENTION_DAYS` | _unset_ (off) | Purge rows older than N days. Unset keeps everything. |
| `OPTIMUS_RETENTION_BATCH_SIZE` | 1000 | Rows deleted per batch/transaction. |
| `OPTIMUS_RETENTION_BATCH_PAUSE_SECONDS` | 0.5 | Sleep between batches. |
| `OPTIMUS_SCHEDULER_RETENTION_PURGE_INTERVAL` | 86400 | Seconds between purge runs. |

## Connection pooling

Each replica's SQLAlchemy async engine keeps its own `QueuePool`. The settings:

| Setting | Default | Meaning |
| --- | --- | --- |
| `OPTIMUS_DB_POOL_SIZE` | 5 | Persistent connections kept open per replica. |
| `OPTIMUS_DB_MAX_OVERFLOW` | 10 | Extra connections opened under burst, closed when idle. |
| `OPTIMUS_DB_POOL_RECYCLE` | 1800 | Reconnect a pooled connection after this many seconds. |
| `OPTIMUS_DB_POOL_PRE_PING` | true | Liveness-check a connection before handing it out. |

**The footprint is per replica.** The real load on Postgres is

```
replicas * (db_pool_size + db_max_overflow)
```

connections against a single `max_connections`. A modest per-replica pool
multiplied across a large fleet can exhaust the server. Size the pool so the
product stays comfortably under `max_connections` (leaving headroom for
migrations, admin, and monitoring), or front Postgres with an external pooler.

`pool_recycle` guards against connections being severed by a server-side
`idle_in_transaction_session_timeout`, a proxy, or a load balancer; recycle
below the shortest such timeout. `pool_pre_ping` transparently replaces a
connection the server has already dropped, at the cost of a cheap round-trip
per checkout.

### External pooling (pgbouncer) — asyncpg caveat

At large fleets, front Postgres with **pgbouncer in transaction-pooling mode**
so thousands of client connections share a small server-side pool. There is one
sharp edge with the asyncpg driver optimus uses:

- **asyncpg uses server-side prepared statements**, and in transaction mode a
  prepared statement created on one server connection may not exist on the next
  connection pgbouncer hands you — yielding errors like
  `prepared statement "__asyncpg_stmt_1__" does not exist`.
- Mitigations: disable statement caching on the asyncpg side
  (`statement_cache_size=0`), and/or set unique prepared-statement names per
  connection. With SQLAlchemy these are passed via the URL/`connect_args`, e.g.
  `?prepared_statement_cache_size=0` or
  `connect_args={"statement_cache_size": 0, "prepared_statement_name_func": ...}`.
- In **session-pooling mode** pgbouncer pins one server connection per client
  for its lifetime, so prepared statements work unchanged — but you lose most of
  the connection-multiplexing benefit.

Keep the in-process `db_pool_size` small when pgbouncer is in front: the pooler,
not SQLAlchemy, is doing the real multiplexing.

## Backups

- **Logical dumps (`pg_dump`).** Simplest for small/medium databases and for
  portable, version-independent snapshots:

  ```
  pg_dump --format=custom --no-owner --dbname="$OPTIMUS_DATABASE_URL" \
    --file=optimus-$(date +%F).dump
  # restore:
  pg_restore --clean --if-exists --no-owner --dbname=optimus optimus-YYYY-MM-DD.dump
  ```

  Run against a replica or during low traffic; a custom-format dump restores
  selectively and in parallel (`pg_restore -j`). Note `pg_dump` captures a
  consistent snapshot but does **not** give point-in-time recovery.

- **Continuous archiving / PITR (WAL-G or pgBackRest).** For large or
  high-value deployments, ship WAL continuously so you can restore to any
  moment. With [WAL-G](https://github.com/wal-g/wal-g):

  ```
  wal-g backup-push $PGDATA      # periodic base backup (e.g. nightly)
  # archive_command = 'wal-g wal-push %p'   in postgresql.conf
  # restore: fetch a base backup, then replay WAL to a target time
  wal-g backup-fetch $PGDATA LATEST
  ```

  Combine periodic base backups with continuous WAL archiving to a separate
  object store (S3/GCS). Test restores regularly — an unverified backup is not
  a backup.

- **Interaction with retention.** Purged rows leave dead tuples; autovacuum
  reclaims them but the on-disk size only shrinks after a `VACUUM FULL` (which
  locks). For backups, the practical effect is that enabling retention keeps
  dump and WAL volume bounded over time rather than ever-growing.
