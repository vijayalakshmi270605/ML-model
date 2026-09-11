"""add Phase 13 monitoring tables and model_versions.baseline_metrics

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-09

Adds `input_statistics` (real-time data-drift signal - per-request
preprocessed-image statistics) and `ground_truth_labels` (delayed
model-performance-monitoring input - a true class attached to a past
prediction once it becomes available), plus a `baseline_metrics` column on
`model_versions` (the comparison point delayed performance is measured
against). See `ad_mri.db.models` and docs/monitoring.md.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from ad_mri.db.types import GUID

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("model_versions", sa.Column("baseline_metrics", sa.JSON(), nullable=True))

    op.create_table(
        "input_statistics",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column(
            "request_id",
            GUID(),
            sa.ForeignKey("prediction_requests.id"),
            nullable=False,
            unique=True,
        ),
        sa.Column("mean_intensity", sa.Float(), nullable=False),
        sa.Column("std_intensity", sa.Float(), nullable=False),
        sa.Column("p01_intensity", sa.Float(), nullable=False),
        sa.Column("p50_intensity", sa.Float(), nullable=False),
        sa.Column("p99_intensity", sa.Float(), nullable=False),
        sa.Column("nonzero_fraction", sa.Float(), nullable=False),
        sa.Column("voxel_count", sa.BigInteger(), nullable=False),
        sa.Column(
            "recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_input_statistics_recorded_at", "input_statistics", ["recorded_at"])

    op.create_table(
        "ground_truth_labels",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column(
            "request_id",
            GUID(),
            sa.ForeignKey("prediction_requests.id"),
            nullable=False,
            unique=True,
        ),
        sa.Column("true_class", sa.String(length=64), nullable=False),
        sa.Column("label_source", sa.String(length=64), nullable=False),
        sa.Column(
            "recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_ground_truth_labels_recorded_at", "ground_truth_labels", ["recorded_at"])


def downgrade() -> None:
    op.drop_index("ix_ground_truth_labels_recorded_at", table_name="ground_truth_labels")
    op.drop_table("ground_truth_labels")

    op.drop_index("ix_input_statistics_recorded_at", table_name="input_statistics")
    op.drop_table("input_statistics")

    op.drop_column("model_versions", "baseline_metrics")
