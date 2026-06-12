"""First-run validation for ``OPTIMUS_MODE=simple``.

Simple mode is the front door: someone with a bot token and nothing else. The
checks here turn the handful of mistakes that front door invites — no token, a
token with stray quotes, a SQLite path the process cannot write — into one-line
actionable messages instead of a pydantic dump or a mid-boot traceback. Each
failure raises :class:`StartupError`, which :mod:`optimus.__main__` prints
without a traceback so the very first thing a new user sees is what to fix.
"""

from __future__ import annotations

from pathlib import Path

from optimus.core.config import Settings


class StartupError(RuntimeError):
    """A misconfiguration caught before boot, carrying a user-facing message.

    The message is the whole error — :mod:`optimus.__main__` prints it as-is and
    exits non-zero, with no traceback, so simple-mode users get one actionable
    line rather than a stack dump.
    """


def _check_token(settings: Settings) -> None:
    """Reject a missing or obviously malformed Discord token.

    We do not validate the token against Discord here (that needs a network
    round-trip and happens at connect time); we only catch the local mistakes
    that otherwise surface as a confusing 401 much later: an unset token, and a
    value pasted with the surrounding quotes or whitespace still attached.
    """
    token = settings.discord_token
    if not token:
        raise StartupError(
            "OPTIMUS_DISCORD_TOKEN is not set. Create a bot at "
            "https://discord.com/developers/applications, copy its token, and set "
            "OPTIMUS_DISCORD_TOKEN (in your environment or a .env file). "
            "See the README quickstart for the exact steps."
        )
    if token != token.strip():
        raise StartupError(
            "OPTIMUS_DISCORD_TOKEN has leading or trailing whitespace. Paste just "
            "the token, with no surrounding spaces or newlines."
        )
    if (token[0], token[-1]) in {('"', '"'), ("'", "'")}:
        raise StartupError(
            "OPTIMUS_DISCORD_TOKEN looks quoted. Set it to the raw token value "
            "with no surrounding quotes."
        )


def _check_sqlite_writable(settings: Settings) -> None:
    """Ensure the simple-mode SQLite file lives somewhere we can write.

    Migrations create the file on first boot, so an unwritable directory (a typo
    in the path, a read-only mount) would otherwise fail deep inside Alembic.
    We resolve the target directory, create it if missing, and confirm we can
    open the database file for writing — all before anything touches the schema.
    """
    url = settings.effective_database_url
    prefix = "sqlite+aiosqlite:///"
    if not url.startswith(prefix):
        return
    raw = url[len(prefix) :]
    if not raw or raw == ":memory:":
        return
    db_path = Path(raw)
    parent = db_path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StartupError(
            f"Cannot create the directory for the SQLite database at {parent} "
            f"({exc.strerror or exc}). Point OPTIMUS_SIMPLE_DATABASE_URL at a "
            "writable path, e.g. sqlite+aiosqlite:///optimus.db in a directory you own."
        ) from exc
    try:
        with db_path.open("a"):
            pass
    except OSError as exc:
        raise StartupError(
            f"Cannot write the SQLite database file at {db_path} "
            f"({exc.strerror or exc}). Point OPTIMUS_SIMPLE_DATABASE_URL at a "
            "writable path, or fix the permissions on its directory."
        ) from exc


def validate_simple_startup(settings: Settings) -> None:
    """Run every preflight check for simple mode, raising :class:`StartupError`.

    Called once at the top of :func:`optimus.app.simple.run_simple`, before any
    migration or network work, so the common first-run mistakes are reported as
    a single clear line rather than a traceback.
    """
    _check_token(settings)
    _check_sqlite_writable(settings)
