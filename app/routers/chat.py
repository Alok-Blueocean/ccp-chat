import uuid

import pandas as pd
from datasets import Dataset
from fastapi import APIRouter, BackgroundTasks, HTTPException
from langfuse.decorators import langfuse_context, observe
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness
from ragas.run_config import RunConfig

from app.core.configs import get_settings
from app.core.langfuse_client import get_langfuse
from app.factories.retrieval_factory import get_retrieval_pipeline
from app.models.schemas import ChatRequest, ChatResponse, ReferenceItem
from app.services.chat_context import build_numbered_chat_context, ordered_parent_slots
from app.services.chat_memory import chat_memory_store
from app.services.llm_service import llm_service

router = APIRouter()


def _score_ragas_background(
    trace_id: str,
    query: str,
    answer: str,
    contexts: list[str],
) -> None:
    print(f"[RAGAS] starting background scoring for trace_id={trace_id}")
    print(f"[RAGAS] contexts count={len(contexts)}, answer_len={len(answer)}")

    try:
        lf = get_langfuse()
        if lf is None:
            print("[RAGAS] skipped — Langfuse not configured (no keys)")
            return

        settings = get_settings()

        dataset = Dataset.from_list([
            {
                "user_input": query,
                "response": answer,
                "retrieved_contexts": contexts,
            }
        ])

        llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=settings.openai_api_key,
                max_tokens=512,
            )
        )
        embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=settings.openai_api_key,
            )
        )

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(max_workers=1, timeout=60, max_retries=1),
        )

        df = result.to_pandas()
        print(f"[RAGAS] result columns: {list(df.columns)}")
        print(f"[RAGAS] scores: {df.iloc[0].to_dict()}")

        for col in ["faithfulness", "answer_relevancy"]:
            if col not in df.columns:
                print(f"[RAGAS] column '{col}' missing from results")
                continue
            val = df.iloc[0][col]
            if pd.notna(val):
                lf.score(trace_id=trace_id, name=col, value=float(val))
                print(f"[RAGAS] posted {col}={val:.4f} to trace {trace_id}")
            else:
                print(f"[RAGAS] {col} is NaN — not posted")

        lf.flush()
        print("[RAGAS] done")

    except Exception as e:
        print(f"[RAGAS] ERROR: {type(e).__name__}: {e}")


@router.post("", response_model=ChatResponse)
@observe(name="chat")
def chat(request: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    if request.session_id:
        try:
            session_uuid = uuid.UUID(request.session_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="session_id must be a valid UUID string.",
            ) from exc
    else:
        session_uuid = uuid.UUID("12345676-1234-5678-1234-567812349843")
        print(session_uuid)
    print("Session ID")
    print(session_uuid)

    langfuse_context.update_current_trace(
        name="chat",
        session_id=str(session_uuid),
        input=request.query,
    )

    pipeline = get_retrieval_pipeline()
    nodes = pipeline.retrieve(
        query=request.query,
        top_k=request.top_k,
    )
    if not nodes:
        raise HTTPException(
            status_code=404,
            detail="No retrieval results for this query.",
        )

    slots = ordered_parent_slots(nodes)
    parent_ids = [pid for pid, _ in slots if pid]
    parent_payloads = pipeline.retriever.qdrant.retrieve_parents_by_ids(parent_ids)

    retrieved_contexts = []
    for parent_id, nws in slots:
        payload = parent_payloads.get(parent_id) if parent_id else None
        text = (payload.get("text") or "") if payload else (nws.node.text or "")
        text = text.strip()[:1500]
        if text:
            retrieved_contexts.append(text)

    context, ref_dicts = build_numbered_chat_context(nodes, parent_payloads)
    if not context:
        raise HTTPException(
            status_code=500,
            detail="Could not build context from retrieval results.",
        )

    history = chat_memory_store.fetch_recent_messages(session_uuid)
    print("History +++++++++++++=================")
    print(history)

    answer = llm_service.generate_chat_answer(
        request.query,
        context,
        history=history or None,
    )

    langfuse_context.update_current_trace(output=answer)

    trace_id = langfuse_context.get_current_trace_id()
    print(f"[CHAT] trace_id={trace_id}")
    if trace_id:
        background_tasks.add_task(
            _score_ragas_background,
            trace_id=trace_id,
            query=request.query,
            answer=answer,
            contexts=retrieved_contexts,
        )

    chat_memory_store.append_turn(session_uuid, request.query, answer)
    references = [ReferenceItem(**d) for d in ref_dicts]
    return ChatResponse(
        answer=answer,
        references=references,
        session_id=str(session_uuid),
    )
