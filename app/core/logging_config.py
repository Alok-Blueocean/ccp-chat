from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Read from os.environ rather than Settings: logging must come up even when
# config validation fails, so we can actually see the validation error.
_DEFAULT_LOG_DIR = "logs"
_DEFAULT_LOG_FILE = "app.log"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
_BACKUP_COUNT = 5  # keep app.log.1 … app.log.5

_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

# Marker so a reload (or a second call) doesn't attach duplicate handlers.
_HANDLER_TAG = "ccp-file-handler"

logger = logging.getLogger(__name__)


def setup_logging() -> Path | None:
    """Send all logs to stderr *and* a rotating file. Returns the log file path.

    Set LOG_DIR to control where the file lands (default ./logs), LOG_LEVEL for
    verbosity. If the directory isn't writable, console logging still works and
    this returns None — logging is never a reason for the app to fail to boot.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(_FORMAT)

    already_attached = any(
        getattr(h, "_ccp_tag", None) == _HANDLER_TAG for h in root.handlers
    )
    if already_attached:
        return _current_log_path(root)

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

    log_dir = Path(os.getenv("LOG_DIR", _DEFAULT_LOG_DIR))
    log_path = log_dir / os.getenv("LOG_FILE", _DEFAULT_LOG_FILE)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("File logging disabled (%s not writable): %s", log_dir, exc)
        _adopt_uvicorn_loggers()
        return None

    file_handler.setFormatter(formatter)
    file_handler._ccp_tag = _HANDLER_TAG  # type: ignore[attr-defined]
    root.addHandler(file_handler)

    _adopt_uvicorn_loggers()
    logger.info("Logging to %s (level=%s, rotate at 10MB x5)", log_path, level_name)
    return log_path


def _adopt_uvicorn_loggers() -> None:
    """Route uvicorn's own loggers through the root handlers.

    Uvicorn installs its own handlers with propagate=False, so without this its
    startup and access lines only ever reach the console, never the file.
    """
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True


def _current_log_path(root: logging.Logger) -> Path | None:
    for handler in root.handlers:
        if getattr(handler, "_ccp_tag", None) == _HANDLER_TAG:
            return Path(handler.baseFilename)  # type: ignore[attr-defined]
    return None
