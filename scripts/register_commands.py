"""Register optimus's global slash commands with Discord.

Registers every command declared in
:mod:`optimus.services.interactions.commands` as a *global* application command
(propagation can take up to an hour). Requires ``OPTIMUS_DISCORD_TOKEN`` and
``OPTIMUS_DISCORD_CLIENT_ID`` in the environment.

Run with: ``python scripts/register_commands.py`` (optionally
``--guild <id>`` to register instantly to one test guild instead of globally).
"""

from __future__ import annotations

import argparse
import asyncio

import hikari

from optimus.core.config import get_settings
from optimus.core.logging import configure_logging, get_logger
from optimus.services.interactions.commands import build_command_builders

_log = get_logger(__name__)


async def _register(guild_id: int | None) -> None:
    settings = get_settings()
    if not settings.discord_token or not settings.discord_client_id:
        raise SystemExit("OPTIMUS_DISCORD_TOKEN and OPTIMUS_DISCORD_CLIENT_ID must be set")

    rest_app = hikari.RESTApp()
    await rest_app.start()
    try:
        async with rest_app.acquire(settings.discord_token, hikari.TokenType.BOT) as rest:
            application = hikari.Snowflake(int(settings.discord_client_id))
            builders = build_command_builders()
            await rest.set_application_commands(
                application,
                builders,  # type: ignore[arg-type]
                guild=hikari.Snowflake(guild_id) if guild_id is not None else hikari.UNDEFINED,
            )
            scope = f"guild {guild_id}" if guild_id is not None else "global"
            _log.info("commands_registered", count=len(builders), scope=scope)
    finally:
        await rest_app.close()


def main() -> None:
    """CLI entrypoint."""
    configure_logging(level="INFO", service_name="optimus-register")
    parser = argparse.ArgumentParser(description="Register optimus slash commands.")
    parser.add_argument(
        "--guild",
        type=int,
        default=None,
        help="Register to a single guild (instant) instead of globally.",
    )
    args = parser.parse_args()
    asyncio.run(_register(args.guild))


if __name__ == "__main__":
    main()
