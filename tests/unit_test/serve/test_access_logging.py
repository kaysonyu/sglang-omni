# SPDX-License-Identifier: Apache-2.0
"""Successful requests are optional; HTTP failures retain their severity."""

import logging
from collections.abc import Iterator

import pytest
import uvicorn

from sglang_omni.serve.access_logging import configure_access_logging


@pytest.fixture
def access_logger(caplog: pytest.LogCaptureFixture) -> Iterator[logging.Logger]:
    loggers = [
        logging.getLogger(name)
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access")
    ]
    original = [
        (logger.level, logger.handlers[:], logger.filters[:], logger.propagate)
        for logger in loggers
    ]
    uvicorn.Config("unused:app", log_level="warning")
    logger = logging.getLogger("uvicorn.access")
    logger.handlers = [caplog.handler]
    try:
        yield logger
    finally:
        for logger, (level, handlers, filters, propagate) in zip(
            loggers, original, strict=True
        ):
            logger.setLevel(level)
            logger.handlers = handlers
            logger.filters = filters
            logger.propagate = propagate


@pytest.mark.parametrize(
    ("log_level", "status", "expected_level"),
    [
        ("warning", 200, None),
        ("warning", 307, None),
        ("warning", 404, logging.WARNING),
        ("warning", 503, logging.ERROR),
        ("error", 404, None),
        ("error", 500, logging.ERROR),
        ("info", 200, logging.INFO),
        ("debug", 200, logging.INFO),
    ],
)
def test_status_filter_runs_even_after_uvicorn_selected_warning(
    access_logger: logging.Logger,
    caplog: pytest.LogCaptureFixture,
    log_level: str,
    status: int,
    expected_level: int | None,
) -> None:
    configure_access_logging(log_level)
    access_logger.info(
        '%s - "%s %s HTTP/%s" %d', "127.0.0.1:1234", "POST", "/generate", "1.1", status
    )
    if expected_level is None:
        assert not caplog.records
    else:
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == expected_level
        assert f"{status}" in caplog.records[0].getMessage()


def test_diagnostic_setting_replaces_previous_filter(
    access_logger: logging.Logger, caplog: pytest.LogCaptureFixture
) -> None:
    configure_access_logging("warning")
    configure_access_logging("info")
    access_logger.info(
        '%s - "%s %s HTTP/%s" %d', "client", "GET", "/health", "1.1", 200
    )
    assert len(caplog.records) == 1


def test_unstructured_errors_are_preserved(
    access_logger: logging.Logger, caplog: pytest.LogCaptureFixture
) -> None:
    configure_access_logging("warning")
    access_logger.error("request transport failed")
    assert caplog.records[0].levelno == logging.ERROR
    assert caplog.records[0].getMessage() == "request transport failed"


def test_invalid_http_log_level_fails_before_installing_filter() -> None:
    with pytest.raises(ValueError, match="Unknown HTTP log level"):
        configure_access_logging("silent")
