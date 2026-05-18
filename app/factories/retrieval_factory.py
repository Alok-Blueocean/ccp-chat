from functools import lru_cache

from app.core.configs import get_settings

from app.services.retrieval_pipeline import (
    RetrievalPipeline,
)

from app.services.transforms.multiquery import (
    MultiQueryTransform,
)

from app.services.transforms.hyde import (
    HydeTransform,
)

from app.services.fusion.rrf import (
    RRFFusion,
)

from app.services.rerankers.cross_encoder import (
    CrossEncoderReranker,
)

from app.services.retrievers.qdrant_hybrid import (
    QdrantHybridRetriever,
)

from indexing_pipeline.index_qdrant import (
    QdrantClintManager,
)

settings = get_settings()


@lru_cache
def get_retrieval_pipeline() -> RetrievalPipeline:

    qdrant_manager = QdrantClintManager()

    retriever = QdrantHybridRetriever(
        qdrant_manager
    )

    multiquery = MultiQueryTransform(
        model=settings.openai_model
    )

    hyde = HydeTransform()

    fusion = RRFFusion()

    reranker = None
    # CrossEncoderReranker(
    #     model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    #     top_n=5,
    # )

    pipeline = RetrievalPipeline(
        retriever=retriever,
        multiquery_transform=multiquery,
        hyde_transform=hyde,
        fusion=fusion,
        reranker=reranker,
    )

    return pipeline


def create_retrieval_pipeline() -> RetrievalPipeline:
    return get_retrieval_pipeline()
