"""global submitter reputation + submitter provenance on global_hashes

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-11

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "global_hashes",
        sa.Column("submitter_user_id", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "global_hashes",
        sa.Column("submitter_guild_id", sa.BigInteger(), nullable=True),
    )
    op.create_index(
        "ix_global_hashes_submitter_user_id", "global_hashes", ["submitter_user_id"]
    )

    op.create_table(
        "global_submitters",
        sa.Column("user_id", sa.BigInteger(), primary_key=True),
        sa.Column("reputation", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("submitted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("confirmed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rejected", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )


def downgrade() -> None:
    op.drop_table("global_submitters")
    op.drop_index("ix_global_hashes_submitter_user_id", table_name="global_hashes")
    op.drop_column("global_hashes", "submitter_guild_id")
    op.drop_column("global_hashes", "submitter_user_id")
