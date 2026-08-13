"""add campaign_id to guild_hashes for scam campaign tracking

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-13

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "guild_hashes",
        sa.Column("campaign_id", sa.String(32), nullable=True),
    )
    op.create_index(
        "ix_guild_hashes_campaign_id",
        "guild_hashes",
        ["campaign_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_guild_hashes_campaign_id", table_name="guild_hashes")
    op.drop_column("guild_hashes", "campaign_id")
