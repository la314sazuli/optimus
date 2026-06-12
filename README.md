# Optimus

[![CI](https://github.com/la314sazuli/optimus/actions/workflows/ci.yml/badge.svg)](https://github.com/la314sazuli/optimus/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Optimus is a Discord bot that automatically detects and removes scam, phishing,
and fraud **images** — fake giveaways, fake Nitro/Steam gifts, fake exchange
screenshots, wallet-drainer QR codes — within seconds of them being posted. It
matches every uploaded image against a database of known scam images using
perceptual hashing, so it still catches a scam after the usual re-share tricks
(cropping, re-coloring, re-compression, resizing, watermarking) while keeping a
zero-false-positive bias so it never punishes legitimate users.

## Quickstart

Three steps: create a bot, run Optimus, tell it what to catch. No database, no
infrastructure — just a bot token.

### 1. Create your Discord bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
   and click **New Application**. Give it a name (e.g. "Optimus") and create it.
2. Open the **Bot** tab. Click **Reset Token**, then **Copy** — this is your
   `OPTIMUS_DISCORD_TOKEN`. Keep it secret.
3. Still on the **Bot** tab, scroll to **Privileged Gateway Intents** and turn on
   **Message Content Intent**. Optimus needs this to see the images people post.
   (Leave Presence and Server Members off.)
4. Open **OAuth2 → URL Generator**. Under **Scopes** tick `bot` and
   `applications.commands`. Under **Bot Permissions** tick:
   **View Channels**, **Read Message History**, **Manage Messages** (to delete
   scams), **Moderate Members** (timeouts), and **Kick Members** / **Ban Members**
   if you want those actions available.
5. Copy the generated URL at the bottom, open it in your browser, and invite the
   bot to your server.

### 2. Run it

Optimus runs as a single process with zero external services. Pick one:

**With [uv](https://docs.astral.sh/uv/):**

```bash
# from a checkout of this repo:
OPTIMUS_DISCORD_TOKEN=your-token-here uv run optimus

# or, once published to an index, with no checkout at all:
OPTIMUS_DISCORD_TOKEN=your-token-here uvx optimus
```

**With Docker (one container, nothing else):**

```bash
docker build -t optimus .
docker run --rm -e OPTIMUS_DISCORD_TOKEN=your-token-here \
  -v optimus-data:/app optimus
```

The image defaults to simple mode, so a single `docker run` is the whole bot.
The volume keeps the SQLite database (registered scam hashes, per-server config)
across restarts; drop it for an ephemeral run.

On startup you'll see a line like `optimus_online bot=YourBot#1234` once it has
connected and registered its slash commands. If the token is missing or
malformed, Optimus prints a single clear line telling you exactly what to fix.

### 3. Teach it which images are scams

Optimus only acts on images you've shown it. In your server, use the slash
commands (registered automatically on first run):

- **`/scamhash add`** — attach a scam image (or paste a hash) to register it.
  From now on Optimus catches re-posts of that image and variants of it.
- **`/scamhash list`** / **`/scamhash remove`** — review or drop registered hashes.
- **`/scamhash import`** / **`/scamhash export`** — share hash sets as JSON between
  servers.
- **`/config set`** — choose what happens on a match (report / delete / timeout /
  ban), the detection **sensitivity** (`strict` / `balanced` / `permissive`), and
  the moderator review channel. **`/config view`** shows the current settings.
- **`/stats`** — see detection activity for your server.

Members can run **`/appeal`** on a detection, and **`/forget_me`** to erase their
data. That's the whole product — register scam images, pick an action, done.

## How detection works (the short version)

Every image is reduced to a four-hash **perceptual fingerprint** (pHash, dHash,
wHash, aHash) and matched against your registered scam hashes plus an optional
shared global database. Perceptual hashing is robust to the transforms scammers
use to dodge exact-match filters, and the ensemble vote plus a tunable
sensitivity preset is what gives the zero-false-positive bias. Detection quality
is measured against a fixture corpus — see
[docs/detection-eval.md](docs/detection-eval.md).

## More

**You don't need any of this to run the bot.** These cover the internals and
running at scale:

- [docs/](docs/README.md) — full documentation index.
- [docs/architecture.md](docs/architecture.md) — system design and the
  six-service distributed topology (`OPTIMUS_MODE=distributed`) for very large
  fleets.
- [docs/scaling.md](docs/scaling.md) — operating at scale: sharding, detection
  replicas, distributed rate limiting, retention, pooling, monitoring, alerting.
- [docs/security-audit.md](docs/security-audit.md) — the security model and audit
  record (SSRF defense, decode sandboxing, signed global DB, multi-tenant RLS).
- [CONTRIBUTING.md](CONTRIBUTING.md) — developer workflow, tests, and PR
  expectations.
- [.env.example](.env.example) — every setting, with the simple-mode keys first.

## License

MIT — see [LICENSE](LICENSE).
