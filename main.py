
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql+psycopg://operator:CastAIP@localhost:2284/airflow"
)

with engine.connect() as conn:
    print("DATABASE Connected!")

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers.retriever import router as retriever_router
from app.routers.chat import router as chat_router
from postgres.client import close_pool, ensure_schema, init_pool

# from app.routers.llm import router as llm_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_pool()
    ensure_schema()
    yield
    close_pool()


app = FastAPI(
    title="RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(
    retriever_router,
    prefix="/retriever",
    tags=["Retriever"],
)

app.include_router(
    chat_router,
    prefix="/chat",
    tags=["Chat"],
)

# app.include_router(
#     llm_router,
#     prefix="/llm",
#     tags=["LLM"],
# )


@app.get("/")
def root():
    return {"message": "RAG API running"}