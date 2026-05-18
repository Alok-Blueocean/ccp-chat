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


def init_pool() -> None:
    global _pool
    settings = get_settings()
    if not settings.database_url:
        logger.info("DATABASE_URL not set; chat memory disabled.")
        return
    if _pool is not None:
        return
    from psycopg_pool import ConnectionPool

    _pool = ConnectionPool(
        conninfo=settings.database_url,
        min_size=1,
        max_size=10,
        open=True,
    )
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
    with pool.connection() as conn:
        for stmt in (s.strip() for s in sql.split(";")):
            if stmt:
                conn.execute(stmt)
        conn.commit()
    logger.info("Chat memory schema ensured.")
