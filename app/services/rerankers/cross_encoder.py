from llama_index.core.postprocessor import (
    SentenceTransformerRerank,
)

from llama_index.core.schema import QueryBundle

from app.services.rerankers.base import BaseReranker
from app.services.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker(BaseReranker):

    def __init__(self, model: str, top_n: int):

        self.reranker = SentenceTransformerRerank(
            model=model,
            top_n=top_n,
        )

    def rerank(self, query, nodes):

        logger.info("Running cross encoder reranking")

        query_bundle = QueryBundle(query)

        reranked_nodes = (
            self.reranker.postprocess_nodes(
                nodes,
                query_bundle=query_bundle,
            )
        )

        logger.info(
            f"Reranked {len(reranked_nodes)} nodes"
        )

        return reranked_nodes