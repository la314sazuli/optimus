"""transactional outbox table for the detection persist->publish dual write

Revision ID: 0006
Revises: 0005
Create Date: 2026-07-13

Detection persists its ``Detection`` row and the resulting ``verdict.v1`` (and
any ``swarm_alert.v1``) into this table in one transaction, then a relay drains
unpublished rows onto the bus with at-least-once semantics. This closes the
dual-write hole where a persisted detection could be left with no verdict on
the bus after a crash or broker outage.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "outbox",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("subject", sa.String(length=128), nullable=False),
        sa.Column("msg_id", sa.String(length=128), nullable=True),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_outbox_unpublished", "outbox", ["id", "published_at"])


def downgrade() -> None:
    op.drop_index("ix_outbox_unpublished", table_name="outbox")
    op.drop_table("outbox")
