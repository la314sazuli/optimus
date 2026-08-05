"""Backfill missing ``guilds`` rows for servers the bot currently belongs to.

Guild config rows are normally created the moment the bot is invited into a
server, in response to Discord's one-shot ``GuildJoinEvent`` (see
``ModerationService.on_guild_joined``). Discord never refires that event for a
server the bot is already a member of. If the event was missed or failed to
persist for any guild (for example, because the bot was invited during a
deployment outage), that guild's config row never gets created, and every
config-touching command (``/config set``, ``/config view`` writes, etc.) fails
with ``KeyError`` forever with no natural retry path.

As of this fix, ``GuildRepository.get_or_create`` makes config access
self-healing going forward (a missing row is lazily provisioned on first use),
but any guild already missing a row needs a one-time backfill to close the
gap immediately rather than waiting for its next config command.

This script fetches every guild the bot is *currently* a member of from
Discord's REST API and ensures each one has a row, using the same
``get_or_create`` used by the self-healing fix. It is idempotent: guilds that
already have a row are left untouched, and it's safe to run multiple times or
concurrently.

Requires ``OPTIMUS_DISCORD_TOKEN`` in the environment (same variable used by
``scripts/register_commands.py``) plus the configured ``OPTIMUS_DATABASE_URL``
(or equivalent settings) so it can reach the same database the bot itself
uses.

Run against the Railway deployment with:

    railway run python scripts/backfill_missing_guilds.py

Or locally against a database you can reach directly:

    uv run python scripts/backfill_missing_guilds.py
"""

from __future__ import annotations

import asyncio

import hikari

from optimus.core.config import get_settings
from optimus.core.logging import configure_logging, get_logger
from optimus.db.engine import create_engine, create_session_factory, session_scope
from optimus.db.repositories import GuildRepository

_log = get_logger(__name__)


async def _fetch_current_guild_ids(token: str) -> list[int]:
    """Return the ids of every guild the bot is currently a member of."""
    rest_app = hikari.RESTApp()
    await rest_app.start()
    try:
        async with rest_app.acquire(token, hikari.TokenType.BOT) as rest:
            guild_ids: list[int] = []
            async for guild in rest.fetch_my_guilds():
                guild_ids.append(int(guild.id))
            return guild_ids
    finally:
        await rest_app.close()


async def _backfill() -> None:
    settings = get_settings()
    if not settings.discord_token:
        raise SystemExit("OPTIMUS_DISCORD_TOKEN must be set")

    guild_ids = await _fetch_current_guild_ids(settings.discord_token)
    _log.info("guilds_fetched", count=len(guild_ids))

    engine = create_engine(settings.effective_database_url, settings=settings)
    factory = create_session_factory(engine)

    checked = 0
    created = 0
    try:
        async with session_scope(factory) as session:
            repo = GuildRepository(session)
            for guild_id in guild_ids:
                checked += 1
                existed = await repo.get(guild_id) is not None
                await repo.get_or_create(guild_id)
                if not existed:
                    created += 1
                    _log.info("guild_backfilled", guild_id=guild_id)
    finally:
        await engine.dispose()

    _log.info("backfill_complete", checked=checked, created=created)
    print(f"Checked {checked} guild(s), created {created} missing row(s).")


def main() -> None:
    """CLI entrypoint."""
    configure_logging(level="INFO", service_name="optimus-backfill")
    asyncio.run(_backfill())


if __name__ == "__main__":
    main()
