"""row-level security for multi-tenant mode

Enables Postgres RLS on every guild-scoped table. Policies restrict rows to the
guild id carried in the ``optimus.guild_id`` session GUC, which the application
sets per request/transaction in multi-tenant mode. On non-Postgres backends
(e.g. SQLite in tests) this migration is a no-op.

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-11

"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Tables carrying a guild_id column that must be tenant-isolated.
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


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    if not _is_postgres():
        return
    for table in _GUILD_TABLES:
        column = _GUILD_COLUMN.get(table, "guild_id")
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        # NULLIF maps an unset GUC (which reads as '' not NULL) to NULL so the
        # comparison yields NULL -> zero rows, rather than raising on ''::bigint.
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} USING "
            f"({column} = NULLIF(current_setting('optimus.guild_id', true), '')::bigint)"
        )


def downgrade() -> None:
    if not _is_postgres():
        return
    for table in _GUILD_TABLES:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
