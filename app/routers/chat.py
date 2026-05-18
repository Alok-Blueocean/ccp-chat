import uuid

from fastapi import APIRouter, HTTPException

from app.factories.retrieval_factory import get_retrieval_pipeline
from app.models.schemas import ChatRequest, ChatResponse, ReferenceItem
from app.services.chat_context import build_numbered_chat_context, ordered_parent_slots
from app.services.chat_memory import chat_memory_store
from app.services.llm_service import llm_service

router = APIRouter()


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if request.session_id:
        try:
            session_uuid = uuid.UUID(request.session_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="session_id must be a valid UUID string.",
            ) from exc
    else:
        session_uuid = uuid.UUID("12345678-1234-5678-1234-567812349843")#uuid.uuid4()
        print(session_uuid)
    print("Session ID")
    print(session_uuid)
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
    chat_memory_store.append_turn(session_uuid, request.query, answer)
    references = [ReferenceItem(**d) for d in ref_dicts]
    return ChatResponse(
        answer=answer,
        references=references,
        session_id=str(session_uuid),
    )
