"""Authentication architecture: a pluggable API-key scheme today, with the
seam a real per-user/OAuth2 scheme would slot into later.

`api_key_scheme` exists purely so FastAPI/OpenAPI documents the requirement
(the "Authorize" lock in `/docs`) — actual enforcement (`api.deps.require_api_key`)
does not depend on it and reads the configured header directly, so a
misconfigured/omitted OpenAPI declaration can never silently bypass a real
check. `auto_error=False`: a missing header is reported through
`require_api_key`'s own `HTTPException` (consistent `ErrorResponse` body),
not FastAPI's default `APIKeyHeader` error shape.

Swapping to a stronger scheme later (per-user API keys stored in Postgres,
OAuth2/JWT via `fastapi.security.OAuth2PasswordBearer`) only touches this
file and `api.deps.require_api_key` — no router imports either directly, they
all depend on `require_api_key`.
"""

from __future__ import annotations

from fastapi.security import APIKeyHeader

from ad_mri.core.constants import API_KEY_HEADER

api_key_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)
