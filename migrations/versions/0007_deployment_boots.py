"""Add deployment_boots: a persistence canary recorded once per process boot.

Each boot inserts one row. Because the count only grows and the first row's
timestamp never changes, moderators (via ``/scamhash stats``) and operators
(via the ``persistence_check`` startup log) can verify the database survives
redeploys — a reset counter means the volume is not mounted and data was lost.

Revision ID: 0007
Revises: 0006
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "deployment_boots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("booted_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("deployment_boots")
