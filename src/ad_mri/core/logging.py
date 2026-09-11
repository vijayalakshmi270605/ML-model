"""Structured logging configuration (stdlib logging + structlog).

Log level/format are driven entirely by Settings (environment variables), never
hardcoded. Sensitive data (raw pixel arrays, PHI, credentials) must never be
passed to a log call anywhere in this codebase — see
docs/TECHNICAL_SPECIFICATION.md §15.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Any

import structlog
import yaml

from ad_mri.core.config import Settings

_FALLBACK_LOGGING_CONFIG_PATH = Path("configs") / "logging.yaml"


def _load_dict_config(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def configure_logging(settings: Settings) -> None:
    """Configure stdlib logging + structlog for the current process.

    Idempotent — safe to call multiple times (e.g. once from the API factory,
    once from the CLI entry point).
    """
    dict_config = _load_dict_config(settings.config_dir / "logging.yaml") or _load_dict_config(
        _FALLBACK_LOGGING_CONFIG_PATH
    )
    if dict_config:
        logging.config.dictConfig(dict_config)
    else:
        logging.basicConfig(level=settings.log_level)

    logging.getLogger().setLevel(settings.log_level.upper())

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer = (
        structlog.processors.JSONRenderer()
        if settings.log_format == "json"
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        foreign_pre_chain=shared_processors,
    )
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog-bound logger. Call configure_logging() first at startup."""
    return structlog.get_logger(name)
