from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.core.logging_config import setup_logging

# Before any other app import, so module-level loggers inherit the config.
setup_logging()

from app.routers.retriever import router as retriever_router  # noqa: E402
from app.routers.chat import router as chat_router  # noqa: E402
from postgres.client import close_pool, ensure_schema, init_pool  # noqa: E402

logger = logging.getLogger(__name__)


def _prewarm_guards() -> None:
    """Load guardrail ML models at startup — only for guards that are enabled.

    Each scanner uses lru_cache, so this is a no-op on subsequent calls.
    Guards with their flag set to false skip model loading entirely.
    """
    from app.core.configs import get_settings
    s = get_settings()

    if s.guardrail_input:
        try:
            from app.guardrails.input_guard import _load_scanners
            logger.info("Pre-warming input guard…")
            _load_scanners()
        except Exception as exc:
            logger.warning("Input guard pre-warm skipped: %s", exc)
    else:
        logger.info("Input guard is OFF — skipping pre-warm.")

    if s.guardrail_pii:
        try:
            from app.guardrails.pii_guard import prewarm as _prewarm_pii
            _prewarm_pii()
        except Exception as exc:
            logger.warning("PII guard pre-warm skipped: %s", exc)
    else:
        logger.info("PII guard is OFF — skipping pre-warm.")

    if s.guardrail_output:
        try:
            from app.guardrails.output_guard import _load_scanners
            logger.info("Pre-warming output guard…")
            _load_scanners()
        except Exception as exc:
            logger.warning("Output guard pre-warm skipped: %s", exc)
    else:
        logger.info("Output guard is OFF — skipping pre-warm.")

    if not s.guardrail_rate_limit:
        logger.info("Rate limit guard is OFF.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_pool()
        ensure_schema()
    except Exception as exc:
        logger.warning("Chat memory unavailable: %s", exc)
    _prewarm_guards()
    yield
    close_pool()


app = FastAPI(
    title="RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(retriever_router, prefix="/retriever", tags=["Retriever"])
app.include_router(chat_router, prefix="/chat", tags=["Chat"])


@app.get("/")
def root():
    return {"message": "RAG API running"}
