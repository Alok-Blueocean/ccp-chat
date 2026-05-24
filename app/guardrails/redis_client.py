from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import redis

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_redis() -> Optional[redis.Redis]:
    """Return a singleton Redis client using the URL from settings.

    Returns None (and logs a warning) when REDIS_URL is not configured so that
    callers can implement fail-open behaviour rather than crashing on startup.
    """
    from app.core.configs import get_settings

    settings = get_settings()
    if not settings.redis_url:
        logger.warning("REDIS_URL not set — Redis-backed features (rate limiting) are disabled.")
        return None

    return redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=10,
        socket_timeout=10,
    )
