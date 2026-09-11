from __future__ import annotations

from ad_mri.core.config import Settings
from ad_mri.core.logging import configure_logging, get_logger


def test_configure_logging_does_not_raise(settings: Settings) -> None:
    configure_logging(settings)
    logger = get_logger(__name__)
    logger.info("logging_smoke_test", ok=True)
