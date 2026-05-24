from fastapi import APIRouter, Depends
from langfuse import Langfuse

from app.core.langfuse_client import get_langfuse
from app.factories.retrieval_factory import create_retrieval_pipeline
from app.guardrails.input_guard import scan_input
from app.guardrails.rate_limiter import retriever_rate_limit

router = APIRouter()


@router.post("/search")
def search(
    query: str,
    top_k: int,
    lf: Langfuse = Depends(get_langfuse),
    pipeline=Depends(create_retrieval_pipeline),
    _rl=Depends(retriever_rate_limit),
):
    safe_query = scan_input(query)

    trace = lf.trace(name="retrieval", input={"query": safe_query, "top_k": top_k})
    span = trace.span(name="vector_search", input={"query": safe_query, "top_k": top_k})

    results = pipeline.retrieve(query=safe_query, top_k=top_k)

    output = [
        {
            "score": float(node.score),
            "node_id": node.node.node_id,
            "text": node.node.text,
            "parent_id": node.node.metadata.get("parent_id"),
            "title": node.node.metadata.get("title"),
            "url": node.node.metadata.get("url"),
        }
        for node in results
    ]

    span.end(output={"retrieved_count": len(output), "results": output})
    trace.update(output={"results": output})

    return {"results": output}
