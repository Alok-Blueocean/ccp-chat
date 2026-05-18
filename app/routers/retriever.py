from fastapi import APIRouter

from app.factories.retrieval_factory import (
    get_retrieval_pipeline,
)

router = APIRouter()

pipeline = get_retrieval_pipeline()


@router.post("/search")
def search(query: str, top_k: int):

    results = pipeline.retrieve(
        query=query,
        top_k=top_k
    )

    return {
        "results": [
            {
                "score": float(node.score),
                "node_id": node.node.node_id,  # The UUID from Qdrant/VectorDB
                "text": node.node.text,
                # Accessing the parent_id we stored in metadata
                "parent_id": node.node.metadata.get("parent_id"),
                # Optional: include the title or URL if needed for the frontend
                "title": node.node.metadata.get("title"),
                "url": node.node.metadata.get("url")
            }
            for node in results
        ]
    }