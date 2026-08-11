import uuid

import pandas as pd
from datasets import Dataset
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
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
from app.guardrails.input_guard import scan_input
from app.guardrails.output_guard import scan_output
from app.guardrails.pii_guard import PiiContext
from app.guardrails.rate_limiter import chat_rate_limit
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
    try:
        lf = get_langfuse()
        if lf is None:
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
                max_tokens=2048,
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
        for col in ["faithfulness", "answer_relevancy"]:
            if col not in df.columns:
                continue
            val = df.iloc[0][col]
            if pd.notna(val):
                lf.score(trace_id=trace_id, name=col, value=float(val))

        lf.flush()

    except Exception as e:
        import logging
        logging.getLogger(__name__).error("[RAGAS] %s: %s", type(e).__name__, e)


@router.post("", response_model=ChatResponse)
@observe(name="chat")
def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    _rl: None = Depends(chat_rate_limit),
) -> ChatResponse:
    # --- session ---
    if request.session_id:
        try:
            session_uuid = uuid.UUID(request.session_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="session_id must be a valid UUID string.",
            ) from exc
    else:
        session_uuid = uuid.uuid4()
        
    # --- input safety scan ---
    safe_query = scan_input(request.query)

    langfuse_context.update_current_trace(
        name="chat",
        session_id=str(session_uuid),
        input=safe_query,
    )

    # --- retrieval (use original sanitized query for best recall) ---
    pipeline = get_retrieval_pipeline()
    nodes = pipeline.retrieve(query=safe_query, top_k=request.top_k)
    if not nodes:
        raise HTTPException(status_code=404, detail="No retrieval results for this query.")

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
        raise HTTPException(status_code=500, detail="Could not build context from retrieval results.")

    history = chat_memory_store.fetch_recent_messages(session_uuid)

    # --- PII: mask query before it reaches the LLM ---
    pii = PiiContext.create()
    llm_query = pii.anonymize(safe_query)

    answer = llm_service.generate_chat_answer(llm_query, context, history=history or None)

    # --- output safety scan ---
    safe_answer = answer#scan_output(llm_query, answer)

    # --- PII: restore original entities in the response ---
    final_answer = pii.deanonymize(llm_query, safe_answer)

    langfuse_context.update_current_trace(output=final_answer)

    trace_id = langfuse_context.get_current_trace_id()
    if trace_id:
        background_tasks.add_task(
            _score_ragas_background,
            trace_id=trace_id,
            query=safe_query,
            answer=final_answer,
            contexts=retrieved_contexts,
        )

    chat_memory_store.append_turn(session_uuid, safe_query, final_answer)
    references = [ReferenceItem(**d) for d in ref_dicts]
    return ChatResponse(answer=final_answer, references=references, session_id=str(session_uuid))
