"""Shared Pydantic schemas used across API routers."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error envelope returned by all API error responses."""

    detail: str
    error_code: str | None = None


class NotImplementedResponse(ErrorResponse):
    """Returned by endpoints whose implementation lands in a later roadmap phase."""

    phase: str = Field(description="Roadmap phase that will implement this endpoint.")
