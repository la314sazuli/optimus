"""created_at indexes for the deployment-wide retention purge

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-12

The settings-driven retention purge (scheduler ``retention_purge`` job) deletes
detections/appeals in bounded batches keyed on ``created_at < cutoff`` with no
guild filter. Without an index on ``created_at`` each batch is a full table
scan; on huge tables that defeats the point of batching. These indexes back
that range scan directly. They are additive and safe to apply online (use
``CREATE INDEX CONCURRENTLY`` manually on very large live tables if a brief
build-time lock is unacceptable).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index("ix_detections_created_at", "detections", ["created_at"])
    op.create_index("ix_appeals_created_at", "appeals", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_appeals_created_at", table_name="appeals")
    op.drop_index("ix_detections_created_at", table_name="detections")
