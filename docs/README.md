# Optimus docs

**You don't need any of this to run the bot.** A bot token and one command
(`uvx optimus` or `docker run`) is the whole story — see the
[README quickstart](../README.md#quickstart). These docs are for when you want to
understand the internals, run the distributed topology, or operate at scale.

## Start here

- [simple-mode.md](simple-mode.md) — how the default single-process mode composes
  the whole bot with zero external services, and the one durability trade-off.

## Design

- [architecture.md](architecture.md) — the system design: message flow, the
  detection pipeline, where state lives, and the resilience controls.
- [detection-eval.md](detection-eval.md) — how detection quality is measured;
  the offline benchmark and what the numbers mean. Headline results live in
  [eval/baseline.md](eval/baseline.md).
- [security-audit.md](security-audit.md) — the security audit record: findings,
  what was verified clean, and low-severity observations.

## Running at scale (distributed mode)

- [scaling.md](scaling.md) — the consolidated operator guide: what to scale, in
  what order, and how to confirm it worked. Start here when one self-hosted
  process is no longer enough.
- [capacity.md](capacity.md) — a measured capacity study (can one 800k-member
  server run on Optimus?) with the throughput baseline and a tuned recipe.
- [sharding.md](sharding.md) — gateway sharding mechanics for large fleets.
- [operations.md](operations.md) — Postgres operations: retention, connection
  pooling, the pgbouncer/asyncpg caveat, and backups.
- [performance-notes.md](performance-notes.md) — the throughput baseline and the
  scale-hardening internals (distributed rate limiting, idempotency &
  back-pressure, payload hardening) behind the levers in scaling.md.

## Compliance

- [privacy-policy-template.md](privacy-policy-template.md) — a privacy-policy
  template for Discord bot verification.
