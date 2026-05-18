from llama_index.core.indices.query.query_transform import (
    HyDEQueryTransform,
)

from llama_index.core.schema import QueryBundle

from app.services.logger import get_logger

logger = get_logger(__name__)


class HydeTransform:

    def __init__(self):

        self.hyde = HyDEQueryTransform(
            include_original=True
        )

    def transform(self, query: str):

        logger.info("Generating HyDE document")

        query_bundle = QueryBundle(query)

        hyde_query = self.hyde.run(query_bundle)

        return hyde_query.embedding_strs[0]