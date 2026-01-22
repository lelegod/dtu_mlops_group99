from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    level: str | None = None,
    log_file: str | None = None,
) -> None:
    """
    Configure loguru once per process.

    - Console logs at `level` (default INFO)
    - Optional file logs at DEBUG with rotation
    """
    level = level or os.getenv("LOGURU_LEVEL", "INFO")

    logger.remove()
    logger.add(sys.stdout, level=str(level))

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level="DEBUG", rotation="100 MB")
