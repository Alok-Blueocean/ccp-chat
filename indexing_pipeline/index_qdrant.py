from typing import List, Any

from app.core.configs import get_settings
from app.services.logger import get_logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
    PointStruct,
    Prefetch,
    Fusion,
    FusionQuery
)

from fastembed import SparseTextEmbedding
from openai import OpenAI

# llama_index NodeRelationship.PARENT enum value
_NODE_PARENT = "4"

from indexing_pipeline.schema import LectureDocument
from indexing_pipeline.chunking import HierarchicalChunker
# from indexing_pipeline import CONFIG
import os
settings = get_settings()
logger = get_logger(__name__)

os.environ['OPENAI_API_KEY'] = settings.openai_api_key
class QdrantClintManager:
    QDRANT_URL = settings.qdrant_url
    QDRANT_API_KEY = settings.qdrant_api_key
    QDRANT_BASE_COLLECTION = settings.qdrant_collection_name
    PARENT_COLLECTION = QDRANT_BASE_COLLECTION + "_parents"
    CHILD_COLLECTION = QDRANT_BASE_COLLECTION + "_children"

    UPSERT_BATCH_SIZE = 2

    def __init__(self):
        self.client = QdrantClient(
            url=self.QDRANT_URL,
            api_key=self.QDRANT_API_KEY,
            timeout=60,
        )
        self.openai_client = OpenAI()
        self.sparse_model = SparseTextEmbedding(
            model_name="prithivida/Splade_PP_en_v1"
        )
        self.chunker = HierarchicalChunker()

    # =========================
    # COLLECTIONS
    # =========================
    def _create_chunk_collection(self, collection_name: str):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "title_dense": VectorParams(size=1536, distance=Distance.COSINE),
                "text_dense": VectorParams(size=1536, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "title_sparse": SparseVectorParams(),
                "text_sparse": SparseVectorParams(),
            }
        )

    def ensure_collections(self):
        if not self.client.collection_exists(self.PARENT_COLLECTION):
            self._create_chunk_collection(self.PARENT_COLLECTION)
        if not self.client.collection_exists(self.CHILD_COLLECTION):
            self._create_chunk_collection(self.CHILD_COLLECTION)

    # =========================
    # EMBEDDINGS
    # =========================
    def get_dense(self, text: str) -> list:
        return self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding

    def get_dense_batch(self, texts: list[str]) -> list[list]:
        safe = [t if t.strip() else " " for t in texts]
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=safe
        )
        return [e.embedding for e in sorted(response.data, key=lambda x: x.index)]

    def get_sparse(self, text: str) -> dict:
        embedding = list(self.sparse_model.embed([text]))[0]
        return {"indices": embedding.indices.tolist(), "values": embedding.values.tolist()}

    def get_sparse_batch(self, texts: list[str]) -> list[dict]:
        safe = [t if t.strip() else " " for t in texts]
        return [
            {"indices": e.indices.tolist(), "values": e.values.tolist()}
            for e in self.sparse_model.embed(safe)
        ]

    # =========================
    # HELPERS
    # =========================
    def _upsert_batched(self, collection_name: str, points: list):
        for i in range(0, len(points), self.UPSERT_BATCH_SIZE):
            self.client.upsert(
                collection_name=collection_name,
                points=points[i : i + self.UPSERT_BATCH_SIZE],
            )

    # =========================
    # PUSH PARENT NODES
    # =========================
    def push_parent_nodes(self, parent_nodes: list):
        titles = [node.metadata.get("title", "") for node in parent_nodes]
        texts = [node.text for node in parent_nodes]

        title_dense = self.get_dense_batch(titles)
        text_dense = self.get_dense_batch(texts)
        title_sparse = self.get_sparse_batch(titles)
        text_sparse = self.get_sparse_batch(texts)

        points = [
            PointStruct(
                id=node.node_id,
                vector={
                    "title_dense": title_dense[i],
                    "text_dense": text_dense[i],
                    "title_sparse": title_sparse[i],
                    "text_sparse": text_sparse[i],
                },
                payload={"node_id": node.node_id, "text": node.text, **node.metadata},
            )
            for i, node in enumerate(parent_nodes)
        ]
        self._upsert_batched(self.PARENT_COLLECTION, points)

    # =========================
    # PUSH CHILD NODES
    # =========================
    def push_child_nodes(self, child_nodes: list):
        titles = [node.metadata.get("title", "") for node in child_nodes]
        texts = [node.text for node in child_nodes]

        title_dense = self.get_dense_batch(titles)
        text_dense = self.get_dense_batch(texts)
        title_sparse = self.get_sparse_batch(titles)
        text_sparse = self.get_sparse_batch(texts)

        points = [
            PointStruct(
                id=node.node_id,
                vector={
                    "title_dense": title_dense[i],
                    "text_dense": text_dense[i],
                    "title_sparse": title_sparse[i],
                    "text_sparse": text_sparse[i],
                },
                payload={
                    "node_id": node.node_id,
                    "text": node.text,
                    "parent_id": ref.node_id if (ref := node.relationships.get(_NODE_PARENT)) else None,
                    **node.metadata,
                },
            )
            for i, node in enumerate(child_nodes)
        ]
        self._upsert_batched(self.CHILD_COLLECTION, points)

    # =========================
    # INDEX FULL DOCUMENT
    # =========================
    def index_document(self, doc: LectureDocument):
        child_nodes, parent_nodes = self.chunker.process_document(doc)
        self.push_parent_nodes(parent_nodes)
        self.push_child_nodes(child_nodes)

    def index_documents(self, documents: List[LectureDocument]):
        for doc in documents:
            self.index_document(doc)

    # =========================
    # HYBRID SEARCH (children)
    # =========================
    def search(self, query: str, limit: int = 10):
        dense = self.get_dense(query)
        sparse = self.get_sparse(query)

        results = self.client.query_points(
            collection_name=self.CHILD_COLLECTION,
            prefetch=[
                Prefetch(query=dense, using="title_dense", limit=limit * 2),
                Prefetch(query=dense, using="text_dense", limit=limit * 2),
                Prefetch(query=sparse, using="title_sparse", limit=limit * 2),
                Prefetch(query=sparse, using="text_sparse", limit=limit * 2),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit
        )
        return results.points

    def retrieve_parents_by_ids(self, parent_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return payload dict keyed by str(parent id) for points in PARENT_COLLECTION."""
        unique: list[str] = []
        seen: set[str] = set()
        for pid in parent_ids:
            if not pid or pid in seen:
                continue
            seen.add(pid)
            unique.append(pid)
        if not unique:
            return {}

        out: dict[str, dict[str, Any]] = {}
        batch_size = 64
        for i in range(0, len(unique), batch_size):
            batch = unique[i : i + batch_size]
            records = self.client.retrieve(
                collection_name=self.PARENT_COLLECTION,
                ids=batch,
                with_payload=True,
            )
            for rec in records:
                if not rec.payload:
                    continue
                key = str(rec.id)
                out[key] = rec.payload

        missing = [pid for pid in unique if pid not in out]
        if missing:
            logger.warning(
                "Parent points not found in Qdrant (count=%s): sample=%s",
                len(missing),
                missing[:5],
            )
        return out


# =========================
# USAGE
# =========================
if __name__ == "__main__":
    qdrant_manager = QdrantClintManager()
    qdrant_manager.ensure_collections()
    print(qdrant_manager.search("is hating others normal?"))
    # raw_data = {'id':"59fb40f7e3556cf2a61aa632d2418721",'url': 'https://www.thespiritualscientist.com/how-can-humility-go-along-with-self-respect/', 'title': 'Humily and self respect', 'transcript': 'Answer:Let’s look at it this way.  Is there a difference between humility and humiliation?  Yes, there is, definitely.\n\nWe say we should cultivate humility, but none of us want to be humiliated.  Let’s put it another way, we all want to cultivate humility.  So, should we start insulting and humiliating each other in our community?  Then we will all become humble.  You are proud and I will insult you, I will humiliate you and you humiliate me and in this way by exchanging humiliation we will all get humility.  No!\n\nDoes humiliation make one humble?  Not necessarily.  Humiliation can make one feel offended, it can make one feel enraged.  Sometimes it may make one humble, but not necessarily.  So, clearly, there is a difference between humiliation and humility, and certainly although we talk about, say, we should be humble, but we also say respect each other. That’s the injunction in our devotee community and that’s normal human conduct also.  So, we could put it that humiliation is false ego frustrated.  Humility is false ego rejected.\n\nHumiliation is false ego frustrated.  I want to be respected, but instead of being respected, I was disrespected, I was mocked, I was derided, I can’t bear this.  So, humiliation is false ego frustrated.  I want to be respected, but I was not.\n\nBut humility is false ego rejected.  That means I don’t crave for respect from others.  I don’t depend on respect from others.  That doesn’t mean that I don’t care at all.  I mean it’s not so easily possible.  We are human beings and we will notice how people are dealing with us.  We can’t artificially become stone-like and that’s not exactly humility.  But humility and humiliation are… Humiliation is I want something and I don’t get that respect, that’s humiliation.  But humility is I don’t want it that much.\n\nSo, we could look at humility from the perspective of what we expect, what we demand from the world, what we need from the world.  So that’s one aspect of humility.\n\nAnother aspect is, if I’m not demanding, how am I looking at myself?  So now we have great saintly people saying that, Bhaktivinoda Thakura says “Amara Jeevana Sada Pape Rata,” that my life is full of sin and there is no good that is seen in me.  Now, he has songs like that, but then he is writing books, he is sharing Bhakti wisdom confidently, countering misconceptions.  So, when he is saying there are no good qualities in me, I am sinful, he is looking at it from a very elevated perspective.  So, he is thinking of how pure Krishna’s devotees are, how pure Krishna is.  As compared to them, what am I?  Krishna loves me so much, what am I doing to reciprocate with him?  I am doing nothing.\n\nSo, when we are at a different level of consciousness, we are not really perceiving how much Krishna loves us.  Often, we may actually be feeling the opposite.  Why doesn’t Krishna care for me?  There are so many things wrong in my life, why is Krishna not helping me? Does Krishna even care for me?  So, when we are not at that level, we can’t artificially imitate that.\n\nOur frame of reference presently is mostly our human society and how people are interacting with each other and people are interacting with us in human society.  But for great souls like Bhaktivinoda Thakura or Krishnadas Kaviraj Goswami, their frame of reference is not human society.  Their frame of reference is Krishna and community of exalted devotees of Krishna.  As compared to that, they feel, what about me, I am nobody. But in human society, they are functioning assertively, they are functioning very purposefully and even strongly when necessary for Krishna’s service.  So, how do we look at ourselves?  One aspect of humility is that it’s not that we look down at ourselves, we think I am worthless, I am useless.  Well, we see that I am a part of Krishna and in that sense there is intrinsic worth for me.  I have a whole seminar on “can we love ourselves”.  Self-love can seem very self-indulgent, but actually it’s not.\n\nIf we understand we are parts of Krishna, how can we love Krishna without loving his parts? The part over which we have most control, the part for which we are most responsible is we ourselves.  Now we don’t love Krishna in the sense of becoming self-indulgent and I am great.  No, but love in the sense of that we respect, we care for ourselves, we respect ourselves, we want the best for ourselves.  So, that is definitely there.  Humility doesn’t mean we look down at ourselves, rather we are not looking at ourselves constantly and not looking at how people are looking at us.  We are thinking of Krishna and we are thinking of how we can serve Krishna.\n\nWe are thinking of what is my service, what is my responsibility, how can I do it the best.  In that sense, self-respect and humility go together because we don’t need other people’s respect because we have that intrinsic self-respect.', 'soundcloud_links': ['https://soundcloud.com/chaitanya-charan/how-can-humility-go-along-with-self-respect'], 'youtube_links': []}

    # doc = LectureDocument(**raw_data)

    # qdrant_manager.index_document(doc)