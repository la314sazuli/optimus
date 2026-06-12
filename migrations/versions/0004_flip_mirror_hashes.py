"""mirror (horizontal-flip) hashes on guild_hashes and global_hashes

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-12

Adds nullable columns holding the perceptual hashes of the horizontally-flipped
source image. Populated when the image is available at indexing time so mirrored
re-shares match their source; left NULL for pre-existing rows and for hex/import
adds that never saw an image. Stored as signed BIGINT via the same two's-complement
mapping the Uint64 type decorator applies (see optimus.db.models.Uint64).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for col in ("mphash", "mdhash", "mwhash", "mahash"):
        op.add_column("guild_hashes", sa.Column(col, sa.BigInteger(), nullable=True))
    for col in ("mphash", "mdhash", "mwhash"):
        op.add_column("global_hashes", sa.Column(col, sa.BigInteger(), nullable=True))


def downgrade() -> None:
    for col in ("mphash", "mdhash", "mwhash"):
        op.drop_column("global_hashes", col)
    for col in ("mahash", "mwhash", "mdhash", "mphash"):
        op.drop_column("guild_hashes", col)
