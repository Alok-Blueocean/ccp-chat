from __future__ import annotations

import logging
import time

from fastapi import HTTPException, Request

from app.guardrails.redis_client import get_redis

logger = logging.getLogger(__name__)


def build_rate_limiter(endpoint: str, limit: int, window: int):
    """Return a FastAPI dependency that enforces a per-IP sliding-window rate limit.

    Uses a Redis sorted set keyed by ``rl:<endpoint>:<client_ip>``.
    Falls through (fail-open) when Redis is unavailable so the app keeps working
    even without a Redis connection.
    """

    def _check(request: Request) -> None:
        r = get_redis()
        if r is None:
            return  # Redis not configured — skip rate limiting

        ip = request.client.host if request.client else "unknown"
        key = f"rl:{endpoint}:{ip}"
        now = time.time()

        try:
            pipe = r.pipeline()
            pipe.zadd(key, {str(now): now})
            pipe.zremrangebyscore(key, 0, now - window)
            pipe.zcard(key)
            pipe.expire(key, window)
            results = pipe.execute()
            count = int(results[2])
        except Exception as exc:
            logger.error("Rate limiter Redis error (allowing request): %s", exc)
            return

        if count > limit:
            logger.warning(
                "Rate limit hit: endpoint=%s ip=%s count=%d limit=%d",
                endpoint, ip, count, limit,
            )
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: max {limit} requests per {window}s.",
                headers={"Retry-After": str(window)},
            )

    return _check


def _limits():
    from app.core.configs import get_settings
    s = get_settings()
    return s.rate_limit_chat, s.rate_limit_retriever, s.rate_limit_window


def chat_rate_limit(request: Request) -> None:
    chat, _, window = _limits()
    build_rate_limiter("chat", chat, window)(request)


def retriever_rate_limit(request: Request) -> None:
    _, retriever, window = _limits()
    build_rate_limiter("retriever", retriever, window)(request)
