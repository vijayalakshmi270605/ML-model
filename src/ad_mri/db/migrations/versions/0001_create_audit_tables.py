"""create the Phase 11 audit-trail tables

Revision ID: 0001
Revises:
Create Date: 2026-09-08

Creates the five tables `db.models` defines: `model_versions`,
`prediction_requests`, `prediction_results`, `explanation_records`, and
`system_events`. Hand-written (not `alembic revision --autogenerate`, which
needs a reachable database this environment does not have) but kept in exact
column-for-column correspondence with `ad_mri.db.models` - see that module's
docstrings for what each column holds and why.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from ad_mri.db.types import GUID

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_versions",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("architecture", sa.String(length=64), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("model_config_hash", sa.String(length=128), nullable=False),
        sa.Column("preprocessing_version", sa.String(length=64), nullable=False),
        sa.Column("preprocessing_config_hash", sa.String(length=128), nullable=False),
        sa.Column("checkpoint_uri", sa.String(length=512), nullable=False),
        sa.Column("stage", sa.String(length=16), nullable=False),
        sa.Column("device", sa.String(length=16), nullable=False),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "model_config_hash", "preprocessing_config_hash", name="uq_model_versions_config_hash"
        ),
    )

    op.create_table(
        "prediction_requests",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("external_request_id", sa.String(length=128), nullable=False),
        sa.Column("model_version_id", GUID(), sa.ForeignKey("model_versions.id"), nullable=True),
        sa.Column("image_ref", sa.String(length=16), nullable=False),
        sa.Column("content_type", sa.String(length=128), nullable=True),
        sa.Column("api_key_hash", sa.String(length=64), nullable=True),
        sa.Column(
            "received_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_prediction_requests_received_at", "prediction_requests", ["received_at"])
    op.create_index(
        "ix_prediction_requests_external_request_id",
        "prediction_requests",
        ["external_request_id"],
    )

    op.create_table(
        "prediction_results",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column(
            "request_id",
            GUID(),
            sa.ForeignKey("prediction_requests.id"),
            nullable=False,
            unique=True,
        ),
        sa.Column("predicted_class", sa.String(length=64), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("probabilities", sa.JSON(), nullable=False),
        sa.Column("processing_time_ms", sa.Float(), nullable=False),
        sa.Column("explanation_available", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    op.create_table(
        "explanation_records",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("external_explanation_id", sa.String(length=128), nullable=False),
        sa.Column("request_id", GUID(), sa.ForeignKey("prediction_requests.id"), nullable=True),
        sa.Column("model_version_id", GUID(), sa.ForeignKey("model_versions.id"), nullable=True),
        sa.Column("methods", sa.String(length=32), nullable=False),
        sa.Column("target_class", sa.String(length=64), nullable=True),
        sa.Column("artifact_uri", sa.String(length=512), nullable=False),
        sa.Column("params_hash", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_explanation_records_created_at", "explanation_records", ["created_at"])
    op.create_index(
        "ix_explanation_records_external_explanation_id",
        "explanation_records",
        ["external_explanation_id"],
    )

    op.create_table(
        "system_events",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=True),
        sa.Column("detail", sa.JSON(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index(
        "ix_system_events_event_type_created_at", "system_events", ["event_type", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_system_events_event_type_created_at", table_name="system_events")
    op.drop_table("system_events")

    op.drop_index(
        "ix_explanation_records_external_explanation_id", table_name="explanation_records"
    )
    op.drop_index("ix_explanation_records_created_at", table_name="explanation_records")
    op.drop_table("explanation_records")

    op.drop_table("prediction_results")

    op.drop_index("ix_prediction_requests_external_request_id", table_name="prediction_requests")
    op.drop_index("ix_prediction_requests_received_at", table_name="prediction_requests")
    op.drop_table("prediction_requests")

    op.drop_table("model_versions")
