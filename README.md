# optimus

Open-source Discord moderation bot focused on detecting and removing scam,
phishing, and fraud **images** (fake giveaways, fake Nitro/Steam gifts, fake
exchange screenshots, wallet-drainer QR codes) in near-real-time, built to scale
to very large guilds.

> This is an early scaffold. A full README — architecture diagram, quickstart,
> configuration reference, scaling guide, and security model — will land as the
> project matures.

## Status

Initial scaffold: core utilities, event contracts, NATS bus helper, perceptual
hashing pipeline, and the async data layer.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run mypy
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).
