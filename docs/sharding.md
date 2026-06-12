# Gateway sharding

The **gateway** service ([`services/gateway/`](../src/optimus/services/gateway/))
holds the only Discord gateway connection in optimus. A single connection
receives every event for every guild the bot is in, so it is the one component
whose load grows directly with guild count and per-guild traffic. Sharding lets
that load be split across multiple gateway connections — and, optionally, across
multiple replicas.

This is configuration-only: the gateway is publish-only and stateless (it owns
no state beyond in-flight publishes), so adding shards changes nothing about how
events flow through the rest of the system. See [architecture.md](architecture.md)
for the wider picture.

## When to shard

- **Discord requires it past ~2,500 guilds.** A single gateway connection may
  serve at most 2,500 guilds; identifying with more is rejected. Past that point
  you *must* run `shard_count >= ceil(guilds / 2500)`.
- **Heavy single servers.** A few very large, very active guilds can saturate one
  connection's event budget well before 2,500 guilds. Sharding spreads guilds
  across connections (Discord assigns a guild to shard `(guild_id >> 22) %
  shard_count`), so splitting into more shards reduces the events per connection.

Small deployments should shard *nothing*: leave both settings unset and hikari
auto-negotiates a single shard. There is zero behavioural change for self-hosters.

## Settings

Two environment variables (prefix `OPTIMUS_`), both optional:

| Setting              | Env var               | Meaning |
| -------------------- | --------------------- | ------- |
| `shard_count`        | `OPTIMUS_SHARD_COUNT` | Total shards across the **whole deployment**. Every replica must agree on this value. Unset → hikari automatic sharding. |
| `shard_ids`          | `OPTIMUS_SHARD_IDS`   | Which shards **this replica** runs. A spec string of comma-separated ids and inclusive ranges (`"0,1"`, `"0-3"`, `"0-1,4,6-7"`). Unset → this replica runs all shards. |

Validation happens at startup (in [`core/config.py`](../src/optimus/core/config.py))
and fails fast with a clear error:

- `shard_ids` set but `shard_count` unset → error (a subset needs a known total).
- any id `>= shard_count` → error (out-of-range shard).
- empty `shard_ids` → error.
- negative ids or inverted ranges (`"3-1"`) → error.

## Example: 4 shards across 2 replicas

Run two gateway processes, each owning half of a 4-shard fleet. Both agree on
`shard_count=4`; each takes a disjoint pair of ids.

Replica A:

```bash
OPTIMUS_SHARD_COUNT=4
OPTIMUS_SHARD_IDS=0-1
```

Replica B:

```bash
OPTIMUS_SHARD_COUNT=4
OPTIMUS_SHARD_IDS=2-3
```

Each process is a normal `python -m optimus.services.gateway` invocation; only
the two env vars differ. Scale out by adding more replicas with more shards and
bumping `shard_count` everywhere in lockstep.

## Readiness

`/readyz` on each gateway replica gains a `shards` check: it reports ready only
when **every** shard that replica runs is alive and connected
([`core/readiness.py`](../src/optimus/core/readiness.py)). This keeps the existing
fail-closed semantics — a replica whose shards have not finished connecting (or
have dropped) is pulled out of rotation rather than serving as healthy. A replica
running shards `0-1` is unaffected by the state of shards `2-3` on another replica.

## `max_concurrency` for identify

Discord limits how many shards may `IDENTIFY` concurrently via the
`max_concurrency` field of the gateway-bot endpoint (the "large bot sharding"
bucket; 1 for most bots, higher for verified bots at scale). hikari reads this
value and schedules shard startup in buckets accordingly, so you do not configure
it here — but it is why bringing up many shards is paced rather than instantaneous.
If you split shards across replicas, each process still respects the same global
bucketing, so stagger or simply let hikari's `startup_window_delay` pace the
identifies; do not try to force all shards up at once.
