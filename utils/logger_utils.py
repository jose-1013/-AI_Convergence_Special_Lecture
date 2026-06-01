from __future__ import annotations

import copy
import logging
import sys
from collections.abc import Iterable
from typing import Any, TextIO

LOG_FORMAT = "{asctime} | {lineno:4d} | {levelname_display} | {message}"
DATE_FORMAT = "%H:%M:%S"
LEVEL_WIDTH = 8
LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
}
RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def __init__(self, *, use_color: bool) -> None:
        super().__init__(fmt=LOG_FORMAT, datefmt=DATE_FORMAT, style="{")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        formatted_record = copy.copy(record)
        level_text = f"{formatted_record.levelname:<{LEVEL_WIDTH}}"
        if self.use_color and formatted_record.levelname in LEVEL_COLORS:
            color = LEVEL_COLORS[formatted_record.levelname]
            level_text = f"{color}{level_text}{RESET}"
        formatted_record.levelname_display = level_text
        return super().format(formatted_record)


def configure_logger(
    level: int | str = "INFO",
    *,
    stream: TextIO | None = None,
    use_color: bool | None = None,
) -> None:
    output = sys.stdout if stream is None else stream
    if use_color is None:
        use_color = bool(getattr(output, "isatty", lambda: False)())

    handler = logging.StreamHandler(output)
    handler.setFormatter(_ColorFormatter(use_color=use_color))
    handler._vlm_logger_utils = True  # type: ignore[attr-defined]

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def get_logger(name: str | None = None) -> logging.Logger:
    root_logger = logging.getLogger()
    has_repo_handler = any(
        getattr(handler, "_vlm_logger_utils", False) for handler in root_logger.handlers
    )
    if not has_repo_handler:
        configure_logger()
    return logging.getLogger(name)


def progress_iterable(
    iterable: Iterable[Any],
    *,
    desc: str | None = None,
    enabled: bool = True,
    logger: logging.Logger | None = None,
) -> Iterable[Any]:
    if not enabled or desc is None:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        (logger or get_logger(__name__)).info("%s", desc)
        return iterable
    return tqdm(iterable, desc=desc)


__all__ = ["configure_logger", "get_logger", "progress_iterable"]
