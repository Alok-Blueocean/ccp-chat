from llama_index.vector_stores.qdrant import (
    QdrantVectorStore
)
from llama_index.core.indices.query.query_transform import (
        HyDEQueryTransform
    )
from llama_index.core.schema import QueryBundle

from llama_index.core import VectorStoreIndex

from indexing_pipeline.index_qdrant import QdrantClintManager

qdrant = QdrantClintManager()

vector_store = QdrantVectorStore(
    client=qdrant.client,
    collection_name=qdrant.CHILD_COLLECTION,
    enable_hybrid=True,
)

index = VectorStoreIndex.from_vector_store(
    vector_store
)

retriever = index.as_retriever(
    similarity_top_k=5,
    vector_store_query_mode="hybrid"
)

hyde = HyDEQueryTransform(
        include_original=True
    )

def get_nodes(query:str):
    print(retriever.retrieve(query))

    

    

   

    query_bundle = QueryBundle(
        query
    )

    print(query_bundle)

    hyde_query = hyde.run(query_bundle)

    print(hyde_query)

    nodes = retriever.retrieve(
        hyde_query
    )
    return nodes

