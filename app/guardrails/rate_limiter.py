from __future__ import annotations

import logging
import time

from fastapi import HTTPException, Request

from app.guardrails.redis_client import get_redis

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    from app.core.configs import get_settings
    return get_settings().guardrail_rate_limit


def _limits():
    from app.core.configs import get_settings
    s = get_settings()
    return s.rate_limit_chat, s.rate_limit_retriever, s.rate_limit_window


def _check(endpoint: str, limit: int, window: int, request: Request) -> None:
    """Sliding-window rate limit check via Redis sorted set.

    Skipped entirely when GUARDRAIL_RATE_LIMIT=false or Redis is unavailable.
    """
    if not _enabled():
        return

    r = get_redis()
    if r is None:
        return  # Redis not configured — fail-open

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


def chat_rate_limit(request: Request) -> None:
    chat_limit, _, window = _limits()
    _check("chat", chat_limit, window, request)


def retriever_rate_limit(request: Request) -> None:
    _, retriever_limit, window = _limits()
    _check("retriever", retriever_limit, window, request)
