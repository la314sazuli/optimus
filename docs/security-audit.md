# Security Audit — Cycle 4

Date: 2026-06-11
Scope: full source tree (`src/optimus`), locked dependencies (`uv.lock`),
secrets/`.gitignore`/`.env` handling, and the branch git history. This audit
verified each claim of the README "Security model" against the actual code
paths and looked for concrete, exploitable issues — not style.

Overall the codebase is well-hardened: the SSRF guard, streaming fetcher, decode
sandbox, Redis TTL discipline, server-side permission re-checks, and Ed25519
signing all hold up under inspection. One real authorization gap was found and
**fixed in this cycle**; the remaining items are low-severity observations that
are either intentional design trade-offs or not worth changing tonight.

## Dependency audit

`uvx pip-audit` against both the active synced environment and the exported
locked requirements (`uv export --frozen --all-extras`) reported **no known
vulnerabilities**. No upgrades were required and `uv.lock` was not modified, so
`uv sync --frozen` is unaffected.

## Findings

### F1 — Appeal button did not verify detection ownership — Medium — FIXED

**Location:** `src/optimus/services/interactions/handlers.py` (`handle_component`,
`APPEAL_OPEN` branch); `src/optimus/services/interactions/service.py` (`DbDeps`);
`src/optimus/db/repositories.py` (`DetectionRepository`).

**Issue.** The `/appeal` *slash command* derives the detection being appealed
server-side via `recent_detection_for(guild_id, user_id)`, which ties it to the
invoking user. The `APPEAL_OPEN` *button* path did not: it took the detection id
straight from the button's `custom_id` (`om:v1:appeal_open:<detection_id>`) as
`ref_id` and passed it to `open_appeal(guild_id, ref_id, user_id)` with no
ownership or guild-existence check. Component `custom_id`s are client-echoed and
forgeable, so a member could open an appeal referencing a detection that is not
theirs (e.g. another member's detection in the same guild, or a guessed
sequential id). If a moderator then approved that appeal,
`reverse_detection_action` would reverse the moderation action on the *other*
user's detection.

**Why it was bounded (not High).** All downstream writes (`set_action_taken`,
`resolve_appeal`, `get_appeal`) are already guild-scoped, so blast radius was
confined to the appealing user's own guild, and `appeal_cooldown_ok` rate-limits
appeals to one per hour per user — limiting flooding and cross-guild reach.

**Fix.** Added `DetectionRepository.belongs_to(detection_id, user_id)` (scoped by
guild + uploader), exposed it on `InteractionDeps` as `detection_belongs_to`,
and made the `APPEAL_OPEN` handler reject the interaction (returning
`command.appeal_none`) when the referenced detection does not belong to the
clicking user in the interaction's guild. The handler also now rejects
`APPEAL_OPEN` in DMs (`GUILD_ONLY`) before any side effect. Covered by new unit
tests in `tests/unit/test_interactions_handlers.py` and
`tests/unit/test_repositories.py`.

## Verified clean (claims that held up)

- **SSRF guard** (`ingest/ssrf.py`): resolves DNS once and pins the IP for the
  connection; validates *every* resolved address (fail-closed); blocks
  loopback/private/link-local/CGNAT/multicast/reserved/unspecified/metadata for
  both IPv4 and IPv6, unwraps IPv4-mapped IPv6, and re-validates literal-IP URLs.
  Non-Discord hosts are HTTPS-only.
- **Fetcher** (`ingest/fetcher.py`): connects to the pinned IP via a static
  resolver, follows redirects manually and re-validates each hop, enforces the
  size cap by streaming (aborts mid-stream; never buffers an oversize body),
  checks both the `Content-Type` header allowlist and magic bytes. Config
  (`ingest_max_bytes`, `ingest_max_redirects`) is wired through from `Settings`.
- **Decode sandbox** (`hashing/decoder.py`): untrusted bytes are decoded in a
  subprocess under `RLIMIT_CPU`, `RLIMIT_AS` (memory), `RLIMIT_FSIZE=0`, a wall
  timeout, a Pillow pixel cap, and a frame cap. Any failure is a non-decision
  (`None`) — the pipeline never acts on an image it could not safely decode.
  No `shell=True`; the child is `[sys.executable, "-c", <inline source>]`.
- **Permissions** (`interactions/`): `default_member_permissions` is only a
  client hint. Every state-changing command and button re-checks the invoker's
  *effective* permissions server-side (`has_permission`), with `ADMINISTRATOR`
  implying all. Report buttons require `MANAGE_GUILD` on *this* click; the GDPR
  purge requires `ADMINISTRATOR`. `member.permissions` is resolved by Discord.
- **Import validation** (`interactions/logic.py`): byte cap before parse, strict
  schema (version pin, unknown-key rejection, type/range checks, note length,
  entry-count cap). No `eval`/`pickle`/`yaml.load`.
- **Redis** (`core/ratelimit.py`, `idempotency.py`, `cooldown.py`,
  `guild_config.py`, `detection/swarm.py`, `moderation/safemode.py`): every key
  is prefixed and built from numeric ids; every writer sets a TTL (token bucket
  `EXPIRE`, `SET ... EX`, swarm `EXPIRE`), so an attacker cannot grow the keyspace
  without bound. Token-bucket and swarm mutations are atomic Lua. The in-memory
  fallback limiter has `evict_idle` to bound memory, and the ingest fallback now
  drives it on a time-gated opportunistic sweep (`sweep_interval`) so the
  process-local map stays bounded in the degraded no-Redis path too (Cycle 14).
  (`redis.eval` is server-side Lua, not Python `eval`.)
- **Signing** (`globaldb/signing.py`): Ed25519 over a canonical sorted-JSON
  encoding; `verify_record` is fail-closed (returns `False`, never raises, on a
  missing/short/bad signature or key).
- **Evidence** (`evidence/store.py`): off by default and opt-in; TTL clamped to
  `[1, 24h]`; SSE on write; short-lived presigned GET URLs; keys derived from
  numeric ids.
- **DB scoping** (`db/repositories.py`, `db/engine.py`): per-guild repositories
  filter every query by `guild_id`; no raw SQL / `text()` interpolation. In
  multi-tenant builds the session scope sets the transaction-local
  `optimus.guild_id` GUC (`SELECT set_config(..., true)`) for guild-scoped work,
  so migrations `0002`/`0007`'s PostgreSQL row-level security actually enforces
  tenant isolation at the database (not merely defence in depth). An unset GUC
  evaluates to NULL, so an RLS-subject role with no GUC sees zero rows.
  Single-tenant/simple mode never sets the GUC and does not depend on RLS.
- **Two-role least privilege** (`docker-compose.yml`,
  `deploy/postgres/10-init-rls-roles.sh`): because FORCE ROW LEVEL SECURITY
  returns *zero* rows (not all rows) for an unscoped query, a single role cannot
  do both per-tenant isolation and cross-tenant maintenance. Request/detection
  services connect as `optimus_app` (NOBYPASSRLS — the RLS-subject role);
  genuinely account-wide work connects as `optimus_maintenance` (BYPASSRLS) via
  `OPTIMUS_MAINTENANCE_DATABASE_URL`: the scheduler's retention/purge/rollup/
  enumeration jobs and the DM-only `/forget_me` GDPR erasure (which spans every
  guild a user appears in). The compose stack no longer runs the superuser as the
  effective app role, so the isolation is real out of the box; the superuser is
  used only by the one-shot `migrate` job that owns the schema. Verified by
  `tests/integration/test_rls_isolation.py` (skipped unless
  `OPTIMUS_TEST_POSTGRES_URL` points at a real Postgres), which asserts both
  per-tenant isolation on the RLS-subject role and correct cross-tenant effects
  on the BYPASSRLS role.
- **Secrets** (`core/config.py`, `core/logging.py`): all secrets come from
  `OPTIMUS_`-prefixed env via pydantic-settings; `.env`/`.env.*` are gitignored
  (with `.env.example` whitelisted and containing only empty placeholders); no
  token/secret is logged; no secret appears anywhere in the branch git history.

## Low-severity observations (no change this cycle)

- **O1 — Pillow decompression-bomb threshold.** Pillow only raises
  `DecompressionBombError` above *2×* `MAX_IMAGE_PIXELS`; between the limit and
  2× it merely warns. An image in that band still decodes. This is acceptable
  because the decode subprocess's `RLIMIT_AS`, `RLIMIT_CPU`, and wall timeout
  independently bound the cost — the pixel cap is one layer of several. No action.
- **O2 — No `RLIMIT_NPROC` on the decode child.** The child cannot meaningfully
  fork-bomb under `RLIMIT_AS`, and `RLIMIT_NPROC` is per-user (would count the
  whole service's threads), so adding it risks breaking legitimate decoding.
  Left as-is intentionally.
- **O3 — Embedded message-content URLs are fetched.** `gateway/extract.py`
  extracts `http(s)` URLs from message text and embed image URLs, which are then
  fetched. This is by design (scammers post links, not just attachments) and is
  fully gated by the SSRF guard, the per-guild fetch rate limit, and the size
  cap, so it is not an SSRF or DoS vector beyond those controls.
