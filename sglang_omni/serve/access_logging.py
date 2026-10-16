# SPDX-License-Identifier: Apache-2.0
"""Status-aware Uvicorn access logging for pipeline servers and routers."""

import logging


class AccessLogFilter(logging.Filter):
    """Keep failed HTTP requests visible when successful access logs are disabled."""

    def __init__(self, minimum_level: int) -> None:
        super().__init__()
        self.minimum_level: int = minimum_level

    def filter(self, record: logging.LogRecord) -> bool:
        arguments = record.args
        if (
            isinstance(arguments, tuple)
            and len(arguments) == 5
            and isinstance(arguments[4], int)
        ):
            status = arguments[4]
            if status >= 500:
                severity = logging.ERROR
            elif status >= 400:
                severity = logging.WARNING
            else:
                severity = logging.INFO
        else:
            severity = record.levelno
        record.levelno = max(record.levelno, severity)
        record.levelname = logging.getLevelName(record.levelno)
        return record.levelno >= self.minimum_level


def configure_access_logging(log_level: str) -> None:
    """Apply after Uvicorn Config has installed its handlers and logger levels."""
    minimum_level = logging.getLevelName(log_level.upper())
    if not isinstance(minimum_level, int):
        raise ValueError(f"Unknown HTTP log level: {log_level}")
    access_logger = logging.getLogger("uvicorn.access")
    for log_filter in access_logger.filters[:]:
        if isinstance(log_filter, AccessLogFilter):
            access_logger.removeFilter(log_filter)
    access_logger.addFilter(AccessLogFilter(minimum_level))
    # note (Codex): Uvicorn emits every status at INFO, before filters can promote failures.
    access_logger.setLevel(min(logging.INFO, minimum_level))
