from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.routers.retriever import router as retriever_router
from app.routers.chat import router as chat_router
from postgres.client import close_pool, ensure_schema, init_pool

logger = logging.getLogger(__name__)


def _prewarm_guards() -> None:
    """Load all guardrail ML models at startup so the first user request is fast.

    Each scanner uses lru_cache internally, so this is a no-op on subsequent calls.
    Order matches the request hot-path: input scanners → PII guard → output scanners.
    """
    try:
        from app.guardrails.input_guard import _load_scanners as _load_input
        logger.info("Pre-warming input guard scanners (TokenLimit, PromptInjection, Toxicity)…")
        _load_input()
    except Exception as exc:
        logger.warning("Input guard pre-warm skipped: %s", exc)

    try:
        from app.guardrails.pii_guard import prewarm as _prewarm_pii
        _prewarm_pii()
    except Exception as exc:
        logger.warning("PII guard pre-warm skipped: %s", exc)

    try:
        from app.guardrails.output_guard import _load_scanners as _load_output
        logger.info("Pre-warming output guard scanners (Sensitive, Toxicity)…")
        _load_output()
    except Exception as exc:
        logger.warning("Output guard pre-warm skipped: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_pool()
    ensure_schema()
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
