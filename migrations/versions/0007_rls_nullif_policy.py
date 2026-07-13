"""re-apply tenant-isolation policy with NULLIF guard (idempotent)

Migration 0002 originally compared ``current_setting('optimus.guild_id', true)``
directly against the guild id. An *unset* GUC reads back as the empty string, so
``''::bigint`` raised ``invalid input syntax for type bigint`` on every unscoped
query instead of yielding zero rows. 0002 was later corrected in place to wrap
the read in ``NULLIF(..., '')`` — but an in-place edit never re-runs on a
database that already applied the old 0002. This forward migration drops and
recreates the ``tenant_isolation`` policy with the NULLIF guard on every
guild-scoped table so the fix reaches already-migrated deployments too. Fresh
databases pass through harmlessly (the policy simply gets recreated identically).

On non-Postgres backends (SQLite in tests) it is a no-op.

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-13

"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Must match the set enabled in migration 0002.
_GUILD_TABLES: tuple[str, ...] = (
    "guilds",
    "guild_channels_ignored",
    "guild_roles_ignored",
    "guild_trusted_users",
    "guild_hashes",
    "guild_whitelist",
    "detections",
    "appeals",
    "mod_actions",
    "stats_rollups",
)

_GUILD_COLUMN = {"guilds": "guild_id"}

_NULLIF_PREDICATE = "{column} = NULLIF(current_setting('optimus.guild_id', true), '')::bigint"
# The pre-fix predicate 0002 shipped before it was corrected in place.
_LEGACY_PREDICATE = "{column} = current_setting('optimus.guild_id', true)::bigint"


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    if not _is_postgres():
        return
    for table in _GUILD_TABLES:
        column = _GUILD_COLUMN.get(table, "guild_id")
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} USING "
            f"({_NULLIF_PREDICATE.format(column=column)})"
        )


def downgrade() -> None:
    if not _is_postgres():
        return
    for table in _GUILD_TABLES:
        column = _GUILD_COLUMN.get(table, "guild_id")
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} USING "
            f"({_LEGACY_PREDICATE.format(column=column)})"
        )
