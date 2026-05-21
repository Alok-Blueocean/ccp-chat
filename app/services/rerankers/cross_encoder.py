from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import QueryBundle
from langfuse.decorators import langfuse_context, observe

from app.services.rerankers.base import BaseReranker
from app.services.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker(BaseReranker):

    def __init__(self, model: str, top_n: int):

        self.reranker = SentenceTransformerRerank(
            model=model,
            top_n=top_n,
        )

    @observe(name="reranker")
    def rerank(self, query, nodes):

        logger.info("Running cross encoder reranking")

        def _node_summary(n, rank):
            return {
                "rank": rank,
                "node_id": n.node.node_id,
                "score": round(n.score, 4),
                "title": n.node.metadata.get("title"),
                "text_preview": (n.node.text or "")[:120],
            }

        before_order = [_node_summary(n, i + 1) for i, n in enumerate(nodes)]

        query_bundle = QueryBundle(query)

        reranked_nodes = self.reranker.postprocess_nodes(
            nodes,
            query_bundle=query_bundle,
        )

        logger.info(f"Reranked {len(reranked_nodes)} nodes")

        langfuse_context.update_current_observation(
            output={
                "input_count": len(nodes),
                "output_count": len(reranked_nodes),
                "before_rerank": before_order,
                "after_rerank": [_node_summary(n, i + 1) for i, n in enumerate(reranked_nodes)],
            }
        )

        return reranked_nodes