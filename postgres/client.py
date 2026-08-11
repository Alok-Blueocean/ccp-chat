from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.configs import get_settings

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# How long to wait for the first connection before giving up and running
# without chat memory. Keeps startup fast when no database is around.
_CONNECT_TIMEOUT_SECONDS = 3.0


def init_pool() -> None:
    """Open the chat-memory pool, or leave it disabled if Postgres is unreachable.

    A missing or unreachable database is never fatal: the app starts and every
    chat-memory operation degrades to a no-op.
    """
    global _pool
    settings = get_settings()
    if not settings.database_url:
        logger.info("DATABASE_URL not set; chat memory disabled.")
        return
    if _pool is not None:
        return
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        conninfo=settings.database_url,
        min_size=1,
        max_size=10,
        open=False,
        kwargs={"connect_timeout": int(_CONNECT_TIMEOUT_SECONDS)},
    )
    try:
        pool.open(wait=True, timeout=_CONNECT_TIMEOUT_SECONDS)
    except Exception as exc:
        # close() also stops the background reconnect worker, so we don't spam
        # "connection refused" warnings for the lifetime of the process.
        pool.close(timeout=1.0)
        logger.warning(
            "Postgres unreachable (%s); continuing without chat memory.", exc
        )
        return
    _pool = pool
    logger.info("Postgres connection pool opened for chat memory.")


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
        logger.info("Postgres connection pool closed.")


def get_pool() -> ConnectionPool | None:
    return _pool


def ensure_schema() -> None:
    """Apply schema from schema.sql if pool is available."""
    pool = get_pool()
    if pool is None:
        return
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    try:
        with pool.connection() as conn:
            for stmt in (s.strip() for s in sql.split(";")):
                if stmt:
                    conn.execute(stmt)
            conn.commit()
    except Exception as exc:
        logger.warning("Could not apply chat memory schema: %s", exc)
        return
    logger.info("Chat memory schema ensured.")
