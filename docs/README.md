# Optimus — Documentation

Optimus is a Discord bot that detects and removes scam, phishing, and fraud
**images** — fake giveaways, fake Nitro/Steam gifts, fake exchange screenshots,
wallet-drainer QR codes — within seconds of posting. Every uploaded image is
matched against a database of known scam images using perceptual hashing, so it
still catches a re-share after the usual evasion tricks (cropping, re-coloring,
re-compression, resizing, watermarking) while keeping a zero-false-positive bias.

For the quickstart (bot creation, `uv run` / `docker run`), see the
[top-level README](../README.md). This document covers the features shipped
today: the moderator commands, the detection intelligence pipeline, admin
configuration, and the architecture behind it.

> **You don't need any of this to run the bot.** A bot token and one command is
> the whole story. These docs are for moderators, admins, and anyone who wants to
> understand how Optimus works.

---

## Moderator Guide

All `/scamhash` and `/config` commands require the **Manage Server** permission
(enforced server-side, not just greyed out client-side). `/delete_server_data`
requires **Administrator**. `/appeal` and `/forget_me` are open to any member.

### Commands

| Command | What it does |
| --- | --- |
| `/scamhash reviewmsg message:<link\|id>` | Hash a message's images, register them as scams, and apply the server's action policy to the author (delete + ban by default). The fastest way to act on a posted scam. |
| `/scamhash scanmsg message:<link\|id>` | **Preview mode**: analyze a message's images (hashes, QR codes, lookalikes, phishing signals) without storing or acting. |
| `/scamhash recent` | Show the last 10 moderation events in this server. |
| `/scamhash explain detection_id:<id>` | Show why a detection was flagged — verdict, matched hash, confidence, action taken. |
| `/scamhash undo [detection_id:<id>]` | Reverse a moderation action. Without an id, reverses the most recent one. |
| `/scamhash help` | Quick reference for moderators (the table above, in-chat). |
| `/scamhash add image:\|phash:\|dhash:\|whash:` | Register a scam hash from an attachment or typed hex. |
| `/scamhash remove hash_id:<id>` | Remove a registered hash. |
| `/scamhash list` | List this server's registered hashes. |
| `/scamhash import file:<json>` | Import a hash set (JSON export) — up to 1000 hashes. |
| `/scamhash export` | Export this server's hashes as a JSON file. |

### Context menu

**Right-click a message → Apps → "Review as scam"** runs the same flow as
`/scamhash reviewmsg` — hash the images, register them, and act on the author.
This is the recommended way to handle a posted scam.

### Embed responses

Every moderator response is a color-coded ephemeral embed:

| Color | Meaning |
| --- | --- |
| 🟢 Green | Success — hash added, action undone, image clean, data erased. |
| 🟡 Yellow | Informational — preview result, intel found, import summary, nothing to undo. |
| 🔴 Red | Error — not found, no permission, rate limited, all attachments failed. |
| ⚪ Gray | Reference — help text, hash list, stats, config view, explain result. |

### Interactive buttons

When a detection is reported to the moderator review channel, the report embed
carries action buttons (each re-checks the clicker's permission server-side):

- **Confirm scam** · **False positive** (reverses the action) · **Ban uploader**
  · **Unban** · **Whitelist image** · **Submit to global**

Appeals, safe-mode resume, and the destructive `/delete_server_data` confirm are
also driven by buttons.

---

## Detection Intelligence

Beyond perceptual-hash matching, Optimus extracts intelligence from each image to
flag scam intent even on images it has never seen before. This runs for both live
detections and `/scamhash reviewmsg` / `scanmsg`.

### Perceptual hashes

Each image is reduced to a four-hash 64-bit fingerprint:

| Hash | Method |
| --- | --- |
| pHash | Low-frequency DCT coefficients vs. their median. |
| dHash | Horizontal brightness gradient. |
| wHash | Haar wavelet approximation coefficients vs. their median. |
| aHash | Average (mean) luminance. |

Each is also computed for the **horizontally-mirrored** image (`mphash`, `mdhash`,
`mwhash`, `mahash`) so a mirrored re-share matches its source at zero distance.
Matching is a Hamming-distance ensemble vote against the registered hashes plus an
optional shared global database, tuned by the server's sensitivity preset.

### QR code extraction

`cv2.QRCodeDetector` decodes QR codes (single and multi) from each image. Decoded
URLs/text are surfaced in scan results and reviewed alongside the OCR signals — a
QR pointing to a wallet-drainer is a strong scam signal.

### OCR text extraction

Tesseract OCR runs in multiple passes per image to maximize recall:

1. **Preprocessing variants** — grayscale, CLAHE contrast enhancement, and Otsu
   binarization. Small images are upscaled; oversized images are capped.
2. **Confidence filtering** — `image_to_data` per-word confidence, dropping tokens
   below 50%; falls back to plain `image_to_string` if confidence data is missing.
3. **Deduplication** — variants' results are merged, duplicates removed.

### URL repair

Scammers defang URLs to evade text filters. Optimus repairs them before analysis:

- `hxxps://` → `https://`
- `[.]`, `(.)`, and the word `dot` → `.`
- Stray whitespace inside domains removed

### Domain lookalike detection

Extracted URLs are normalized and compared against a watchlist of **20 official
AI-company domains** (perplexity.ai, openai.com, anthropic.com, claude.ai,
gemini.google.com, midjourney.com, x.ai, mistral.ai, meta.ai, …). A domain is
flagged as a lookalike if it:

- shares the second-level label but uses a different TLD (e.g. `openai.xyz`),
- is a homoglyph/typo within Levenshtein distance ≤ 2,
- or embeds the official label as a prefix/suffix.

### Phishing signal engine

OCR text is scored by a weighted rule set. Each matched category contributes its
weight to a running total:

| Signal | Weight | Examples |
| --- | --- | --- |
| `free_offer` | 1 | free, airdrop, giveaway, reward |
| `urgency` | 1 | limited, expires, act now, hurry |
| `impersonation` | 1 | official, support, admin, staff |
| `claim` | 2 | claim, redeem, collect, get your |
| `ai_community` | 2 | sora, gpt-5, claude beta, free pro |
| `credentials` | 3 | login, password, api key, connect wallet |
| `wallet` | 3 | seed phrase, private key, recovery phrase |
| `scam_phrase` | 3 | "claim your free", "scan qr to claim", "connect your wallet" |
| `crypto_address` | 4 | `0x` + 40 hex chars |

### Risk levels

The total score maps to a risk level:

| Score | Risk level |
| --- | --- |
| ≥ 7 | critical |
| ≥ 4 | high |
| ≥ 2 | medium |
| ≥ 1 | low |
| 0 | none |

### URL-signal correlation

A scam-text signal co-occurring with a URL **or** a lookalike domain adds **+3**
each to the score — scam text plus a link is a much stronger indicator than either
alone. The final risk level and the matched signals are reported on scan/review
results.

---

## Admin Guide

### Server configuration

| Command | Purpose |
| --- | --- |
| `/config view` | Show the current configuration for this server. |
| `/config set field:<name> value:<val>` | Set one config field. |

Configurable fields (the same names shown by `/config view`):

| Field | Values |
| --- | --- |
| `sensitivity` | `strict` · `balanced` · `permissive` |
| `action_policy` | `report` · `delete` · `timeout` · `ban` |
| `mod_queue_threshold` | score above which a detection is queued for review |
| `review_channel` | channel id for moderator reports |
| `safe_mode` | on/off — pauses automated actions after repeated failures |
| `retention_days` | purge detections/appeals older than N days (empty = keep all) |
| `locale` | `en` · `sr` |
| `optin_global_db` | share hashes with / consume from the global database |
| `optin_scan_bots` | scan messages from other bots |
| `optin_evidence_storage` | store image evidence (S3/MinIO) for appeals |

### Statistics

`/stats` shows detection activity for this server over the last 24 hours.

### Hash database export

`/scamhash export` returns this server's registered hashes as a JSON file, which
can be imported into another server with `/scamhash import`.

### Deployment

**Docker** (default — simple mode, one container, no external services):

```bash
docker build -t optimus .
docker run --rm -e OPTIMUS_DISCORD_TOKEN=your-token \
  -v optimus-data:/data optimus
```

The image installs `tesseract-ocr` and `opencv-python-headless` for the OCR and
QR pipelines, pins Python 3.12, and runs as an unprivileged user. The SQLite
database lives under the `/data` volume.

**Railway** — a `railway.json` is included; deploy from this repo and set
`OPTIMUS_DISCORD_TOKEN`. Attach a Railway Volume at `/data` to persist the
database across deploys.

**uv** (from a checkout):

```bash
OPTIMUS_DISCORD_TOKEN=your-token-here uv run optimus
```

### Environment

The only **required** setting in simple mode:

| Variable | Purpose |
| --- | --- |
| `OPTIMUS_DISCORD_TOKEN` | Your Discord bot token. |

`OPTIMUS_SIMPLE_DATABASE_URL` (default `sqlite+aiosqlite:///optimus.db`) sets the
SQLite path. Every other setting is advanced/distributed-mode tuning — see
[.env.example](../.env.example) for the full list.

---

## Architecture

Optimus is a [Hikari](https://github.com/hikari-py/hikari)-based Discord bot. In
**simple mode** (the default) the whole bot runs as a single process with zero
external services; **distributed mode** (`OPTIMUS_MODE=distributed`) splits it
into six single-purpose services communicating over a NATS JetStream event bus.

| Layer | Simple mode | Distributed mode |
| --- | --- | --- |
| State store | SQLite (`aiosqlite`) | PostgreSQL (`asyncpg`) + Redis |
| Transport | in-process bus | NATS JetStream |
| Processes | 1 | gateway · ingest · detection · moderation · interactions · scheduler |

**Perceptual hashing pipeline** — fetch (SSRF-hardened) → sandboxed decode
(Pillow + numpy, CPU/memory/time bounded) → grayscale → compute pHash/dHash/
wHash/aHash + mirror variants → ensemble Hamming match against the hash index.

**OCR / intelligence pipeline** — image bytes → multi-pass Tesseract (preprocess
variants → confidence filtering → dedup) → URL repair → URL extraction → domain
lookalike detection → phishing signal scoring → risk level. The same pipeline
feeds both live detections and the `reviewmsg` / `scanmsg` commands.

**i18n** — user-facing strings live in per-locale JSON catalogs under
[`i18n/locales/`](../src/optimus/i18n/locales) (English `en` and Serbian `sr`).
Lookups fall back to English when a key or locale is missing.

For the full system design (event contracts, idempotency, RLS multi-tenancy,
circuit breakers, safe mode, sharding), see
[architecture.md](architecture.md). For operating at scale, start with
[scaling.md](scaling.md).

---

## Further reading

These cover the internals and running at scale — none are required to run the bot:

- [architecture.md](architecture.md) — system design, the six-service topology, resilience controls.
- [simple-mode.md](simple-mode.md) — how the default single-process mode composes the whole bot.
- [detection-eval.md](detection-eval.md) — how detection quality is measured; baseline in [eval/baseline.md](eval/baseline.md).
- [security-audit.md](security-audit.md) — security model and audit record.
- [scaling.md](scaling.md) — the consolidated operator guide for large fleets.
- [capacity.md](capacity.md) — a measured capacity study.
- [sharding.md](sharding.md) — gateway sharding mechanics.
- [operations.md](operations.md) — Postgres operations: retention, pooling, backups.
- [performance-notes.md](performance-notes.md) — throughput baseline and scale-hardening internals.
- [privacy-policy-template.md](privacy-policy-template.md) — a privacy-policy template for bot verification.
