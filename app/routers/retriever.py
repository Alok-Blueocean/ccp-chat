from fastapi import APIRouter, Depends
from langfuse import Langfuse

from app.core.langfuse_client import get_langfuse
from app.factories.retrieval_factory import create_retrieval_pipeline
print("r1")
router = APIRouter()
print("r1")
# pipeline = get_retrieval_pipeline()
print("r2")

@router.post("/search")
def search(query: str, top_k: int, lf: Langfuse = Depends(get_langfuse),
                                   pipeline = Depends(create_retrieval_pipeline)):

    trace = lf.trace(
        name="retrieval",
        input={"query": query, "top_k": top_k},
    )

    span = trace.span(
        name="vector_search",
        input={"query": query, "top_k": top_k},
    )

    results = pipeline.retrieve(query=query, top_k=top_k)

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
