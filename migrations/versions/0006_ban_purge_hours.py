"""Add guilds.ban_purge_hours: ban-time message purge window.

Mirrors Discord's native ban dialog "delete message history" option: when a
ban executes, the banned user's messages from the last N hours are removed
across all channels. Defaults to 24 so a scam ban also sweeps the scammer's
other recent posts; 0 disables the purge.

Revision ID: 0006
Revises: 0005
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "guilds",
        sa.Column("ban_purge_hours", sa.Integer(), nullable=False, server_default="24"),
    )


def downgrade() -> None:
    op.drop_column("guilds", "ban_purge_hours")
