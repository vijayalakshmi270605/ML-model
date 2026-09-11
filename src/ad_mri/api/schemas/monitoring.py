"""Request/response contracts for `POST /api/v1/monitoring/ground-truth`.

See docs/monitoring.md "Delayed model-performance monitoring".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

#: A small, fixed vocabulary, enforced (not just documented) - a `Literal`,
#: not a free-text `str` field (see `db.models.GroundTruthLabelRecord`'s
#: docstring for why: free text here is exactly where PHI could accidentally
#: get typed). FastAPI/pydantic reject anything outside this set with a
#: normal `422`, before this value ever reaches the database.
LabelSource = Literal["clinician_review", "follow_up_diagnosis", "biopsy_confirmed", "other"]


class GroundTruthRequest(BaseModel):
    """Attaches a true class to a *past* prediction, once it becomes known -
    requirement: never assume this is available immediately."""

    request_id: str = Field(
        description="The request_id returned by the original POST /predictions call."
    )
    true_class: str = Field(description="The class this study was actually diagnosed as.")
    label_source: LabelSource = Field(description="How this label was obtained.")


class GroundTruthResponse(BaseModel):
    request_id: str
    true_class: str
    label_source: str
    recorded: bool = True
