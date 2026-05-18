from llama_index.core.schema import (
    TextNode,
    NodeWithScore,
)

from app.services.retrievers.base import BaseRetriever
from app.services.logger import get_logger

logger = get_logger(__name__)


class QdrantHybridRetriever(BaseRetriever):

    def __init__(self, qdrant_client_manager):

        self.qdrant = qdrant_client_manager

    def retrieve(self, query: str, top_k: int):

        logger.info("Running hybrid retrieval")

        results = self.qdrant.search(
            query=query,
            limit=top_k,
        )

        nodes = []

        for point in results:
            # Extract metadata safely from payload
            payload = point.payload or {}
            
            text_node = TextNode(
                id_=point.id,  # Explicitly set the Node ID from the vector DB ID
                text=payload.get("text", ""),
                metadata={
                    "id": payload.get("id"),           # The internal ID from your JSON
                    "parent_id": payload.get("parent_id"), # This is what you need for hierarchy
                    "title": payload.get("title"),
                    "url": payload.get("url"),
                    "source": payload.get("source"),
                    "audio_links": payload.get("audio_links"),
                    "video_links": payload.get("video_links"),
                },
            )

            node = NodeWithScore(
                node=text_node,
                score=point.score,
            )

            nodes.append(node)

        logger.info(f"Retrieved {len(nodes)} nodes")

        return nodes