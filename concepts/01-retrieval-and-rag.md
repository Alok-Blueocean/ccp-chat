# Retrieval & RAG Concepts

Deep dive into every retrieval/RAG mechanism in this repo, grounded in the actual indexing (`indexing_pipeline/`) and serving (`app/services/`) code. See [README.md](README.md) for the full architecture diagram and how this file fits with the others.

---

## Hierarchical parent-child chunking

**What it is**
The indexing pipeline splits every source article into two coexisting granularities of the same text: a 1024-token "parent" chunk and one or more 256-token "child" (leaf) chunks nested inside it. Children are the unit that gets embedded and searched; parents are the unit that eventually gets sent to the LLM as context. This solves a real tension in RAG: small chunks give precise, high-signal embeddings (good retrieval) but don't carry enough surrounding text to answer well (bad generation); large chunks carry plenty of context (good generation) but their embeddings are diluted averages of many ideas (bad retrieval). Splitting the two roles across two node sizes gets both properties at once instead of picking one. The relationship between a child and its parent is preserved via LlamaIndex's node relationship graph (`NodeRelationship.PARENT`) and is later read back out as `parent_id` when the child is pushed to Qdrant.

**How it works**
- `HierarchicalNodeParser.from_defaults(chunk_sizes=[1024, 256], chunk_overlap=20)` parses one `Document` into a tree of nodes at both sizes, with 20-token overlap at each level to avoid clean sentence cuts at chunk boundaries.
- `get_leaf_nodes(all_nodes)` extracts just the 256-token nodes — these are the children.
- Everything in `all_nodes` that is *not* a leaf is, by construction, a parent (`[n for n in all_nodes if n not in child_nodes]`).
- Each `LectureDocument` (id, url, title, transcript, audio/video links) is converted into a single LlamaIndex `Document` with the transcript as text and the rest as metadata, then parsed — so metadata like `title`/`url` rides along on every resulting node.
- Downstream (`index_qdrant.py`), each child node's parent reference is read via `node.relationships.get(_NODE_PARENT)` and stored as a plain `parent_id` string in the child's Qdrant payload — the tree structure is flattened into a foreign-key-style pointer for cheap lookup later.

**Example**
```python
# File: indexing_pipeline/chunking.py
class HierarchicalChunker:
    def __init__(self):
        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes = [1024, 256],
            chunk_overlap = 20
        )

    def _parse(self, document):
        all_nodes = self.node_parser.get_nodes_from_documents([document])
        child_nodes = get_leaf_nodes(all_nodes)
        # Identify Parents (nodes that are not leaves)
        parent_nodes = [n for n in all_nodes if n not in child_nodes]
        return child_nodes, parent_nodes

    def process_document(self, document: LectureDocument):
        created_document = self._create_document(document)
        child_nodes, parent_nodes = self._parse(created_document)
        return child_nodes, parent_nodes
```

**Where in this repo**
`indexing_pipeline/chunking.py` (`HierarchicalChunker._parse`, `process_document`); consumed by `indexing_pipeline/index_qdrant.py::index_document`, which pushes `parent_nodes` and `child_nodes` to two separate Qdrant collections. The `parent_id` written onto each child's payload is read back at answer time in `app/services/chat_context.py::ordered_parent_slots`.

**Interview angle**
Q: Why not just pick one chunk size and tune it?
A: Because retrieval and generation want opposite properties from chunk size — retrieval wants small, topically pure chunks so the embedding isn't diluted; generation wants enough surrounding text that the LLM isn't working from a fragment. Hierarchical parent-child chunking is a structural fix rather than a tuning compromise: search on the 256-token children, but once a child is selected as relevant, swap in its 1024-token parent before building the LLM context. You get precise retrieval and coherent generation from the same underlying document.

---

## Dual dense + sparse embeddings

**What it is**
Every chunk (parent and child alike) is embedded four separate ways and stored as four named vectors on one Qdrant point: `title_dense` and `text_dense` (OpenAI `text-embedding-3-small`, 1536 dimensions) plus `title_sparse` and `text_sparse` (SPLADE, via the `fastembed` library's `prithivida/Splade_PP_en_v1` model). Dense vectors capture semantic/conceptual similarity — "letting go of anger" can match a chunk about "releasing resentment" even with zero shared words. Sparse vectors are learned bag-of-weighted-terms representations that behave like a smarter TF-IDF/BM25: they excel at exact terminology and proper-noun matches ("Bhaktivinoda Thakura", "false ego") that dense embeddings can blur or miss entirely. Storing both — separately for the title and for the body text — means a single Qdrant collection can be searched on four independent signals simultaneously, which is what makes the subsequent hybrid search meaningful rather than just an average of two similar dense scores.

**How it works**
- The Qdrant collection schema declares two dense vector fields (`VectorParams(size=1536, distance=Distance.COSINE)`) and two sparse vector fields (`SparseVectorParams()`), one pair each for title and text.
- `get_dense` / `get_dense_batch` call OpenAI's embeddings API (`text-embedding-3-small`) for a single string or a batch, preserving order via `sorted(response.data, key=lambda x: x.index)`.
- `get_sparse` / `get_sparse_batch` run the local SPLADE model through `fastembed.SparseTextEmbedding`, producing `{indices, values}` pairs — a sparse vector is just the nonzero token-weight positions, not a dense 1536-length array.
- Blank strings are guarded against (`t if t.strip() else " "`) since some titles/texts can be empty and both embedding APIs choke on truly empty input.
- At indexing time all four embeddings are computed in batches for every node and written into one `PointStruct.vector` dict keyed by field name — one point, four vectors.

**Example**
```python
# File: indexing_pipeline/index_qdrant.py
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

def get_dense(self, text: str) -> list:
    return self.openai_client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding

def get_sparse(self, text: str) -> dict:
    embedding = list(self.sparse_model.embed([text]))[0]
    return {"indices": embedding.indices.tolist(), "values": embedding.values.tolist()}
```

**Where in this repo**
`indexing_pipeline/index_qdrant.py` (`QdrantClintManager._create_chunk_collection`, `get_dense`/`get_dense_batch`, `get_sparse`/`get_sparse_batch`, `push_parent_nodes`, `push_child_nodes`). Both `PARENT_COLLECTION` and `CHILD_COLLECTION` use this exact same four-vector schema.

**Interview angle**
Q: Why store title and text as separate vector fields instead of concatenating title+text into one embedding?
A: Concatenation forces one vector to represent two different kinds of signal (a short label vs. a long body) and whichever is longer dominates the embedding. Keeping them as separate named vectors lets the retrieval step prefetch and fuse across all four fields independently, so a query that matches strongly on title wording (e.g. an exact article title match) isn't drowned out by weaker body-text similarity, and vice versa.

---

## Server-side RRF fusion (Qdrant)

**What it is**
A single query against Qdrant's `query_points` API prefetches candidates from all four named vector fields (`title_dense`, `text_dense`, `title_sparse`, `text_sparse`) in one request, then asks Qdrant itself to fuse those four ranked lists using Reciprocal Rank Fusion (`Fusion.RRF`) before returning results. This is "server-side" fusion in the sense that the ranking math for combining dense and sparse signal types happens inside the vector database, not in application code — the app gets back one already-fused, already-ranked list of Qdrant points. It is a distinct fusion step from the "application-level RRF fusion" described below: this one fuses across *vector field types* for a single query string; the other fuses across *query variants* (multi-query paraphrases) after each has already gone through this exact hybrid search.

**How it works**
- The query text is embedded twice: `get_dense(query)` for the dense side, `get_sparse(query)` for the sparse side (both fields reuse these same two embeddings — the dense query vector is compared against both `title_dense` and `text_dense`, likewise for sparse).
- Four `Prefetch` clauses each request `limit * 2` candidates from one named vector field — over-fetching gives RRF more raw material to fuse from.
- A single top-level `query=FusionQuery(fusion=Fusion.RRF)` tells Qdrant to combine the four prefetch result sets by reciprocal rank and return only `limit` final points — one network round trip, no manual score normalization across dense cosine similarity and sparse dot-product scores (which live on different scales and would be unsafe to average directly).
- `QdrantHybridRetriever.retrieve()` wraps this: it calls `qdrant.search()`, then converts each raw Qdrant `point` into a LlamaIndex `TextNode` + `NodeWithScore`, copying over payload fields (`text`, `parent_id`, `title`, `url`, `source`, audio/video links) as node metadata so the rest of the pipeline can work with LlamaIndex's node abstraction instead of raw Qdrant points.

**Example**
```python
# File: indexing_pipeline/index_qdrant.py
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
```

**Where in this repo**
`indexing_pipeline/index_qdrant.py::QdrantClintManager.search()`; wrapped by `app/services/retrievers/qdrant_hybrid.py::QdrantHybridRetriever.retrieve()`, which turns Qdrant points into `NodeWithScore` objects for the rest of the pipeline.

**Interview angle**
Q: Why RRF instead of a weighted average of dense and sparse scores?
A: Dense cosine similarity and sparse dot-product scores aren't on comparable scales, so averaging or weighting them directly requires ad-hoc normalization that's brittle across queries. RRF sidesteps the problem entirely by discarding raw scores and fusing on *rank position* (`1 / (k + rank))`) — a chunk that's #1 in the sparse list and #1 in the dense list gets a strong combined score regardless of what the underlying similarity numbers were. It's also why doing this fusion inside Qdrant is convenient: the DB already has the ranked lists in hand from the prefetch step.

---

## Application-level RRF fusion

**What it is**
A second, independent RRF pass that runs in Python after all query variants (from multi-query expansion) have each already gone through the server-side hybrid search above. Where the Qdrant-side RRF fuses *vector field types* for one query string, this pass fuses *entire ranked node lists coming from different query paraphrases* into one combined ranking. It's the mechanism that actually makes multi-query expansion pay off: without it, you'd just have several disconnected top-k lists with no principled way to merge them into a single ranking to hand to the reranker.

**How it works**
- `RetrievalPipeline` calls `self.fusion.fuse(all_results)`, passing a list of node lists — one list per query variant, each already ranked by the hybrid retriever.
- `RRFFusion.fuse()` iterates every node list; for each node at rank `r` (0-indexed) in a list, it adds `1 / (k + r + 1)` to that node's running score in a `scores` dict keyed by `node_id` (default `k=60`, the same constant popularized by the original RRF paper).
- A node that appears near the top of *multiple* query variants' results accumulates score contributions from each appearance, so consistently-relevant nodes rise to the top of the fused ranking even if no single query variant ranked them #1.
- The final list is `sorted(scores.items(), key=lambda x: x[1], reverse=True)`, mapped back to the original `NodeWithScore` objects via a `node_map`.
- Note this fusion step runs on `all_results` (the full per-query lists, pre-dedup) rather than on the already-deduplicated set — dedup in the pipeline is really about the fallback path (no fusion configured) and for logging; the fusion path does its own implicit consolidation because the same `node_id` naturally reappears across lists and accumulates score.

**Example**
```python
# File: app/services/fusion/rrf.py
class RRFFusion(BaseFusion):
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, retrieval_results):
        scores = defaultdict(float)
        node_map = {}
        for node_list in retrieval_results:
            for rank, node in enumerate(node_list):
                node_id = node.node.node_id
                node_map[node_id] = node
                scores[node_id] += 1 / (self.k + rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        final_nodes = [node_map[node_id] for node_id, _ in ranked]
        return final_nodes
```

**Where in this repo**
`app/services/fusion/rrf.py::RRFFusion.fuse()`; invoked from `app/services/retrieval_pipeline.py::RetrievalPipeline.retrieve()` step 4, wired in via `app/factories/retrieval_factory.py::get_retrieval_pipeline()` (`fusion = RRFFusion()`).

**Interview angle**
Q: You have RRF in two places — isn't that redundant?
A: No, they fuse different axes. The Qdrant-side RRF merges dense vs. sparse, title vs. text signal *for a single query string*. The app-level RRF merges the independent hybrid-search result sets produced by *different paraphrases of the same underlying question* (multi-query expansion). Removing either one changes behavior: dropping the DB-level RRF would mean choosing just one vector field per search; dropping the app-level RRF would mean falling back to a plain best-score sort across query variants, which doesn't reward a node for being consistently retrieved across multiple phrasings.

---

## Multi-query expansion

**What it is**
Before retrieval runs, an LLM call rewrites the user's query into three additional semantically-equivalent but differently-worded queries, and the original query is kept too. Each of the resulting (deduplicated) queries is retrieved independently, and the results are merged downstream by fusion. This exists to counter vocabulary mismatch: the user's phrasing and the corpus's phrasing of the same idea can diverge (a user might ask "why do bad things happen to good people" while the source articles discuss it as "reconciling suffering with karma"), and a single embedding of the literal query can miss chunks that a slightly different phrasing would have surfaced.

**How it works**
- `MultiQueryTransform.transform(query)` sends the user's query to an LLM (via `litellm.completion`, model configured through settings) with a strict system prompt instructing it to return exactly 3 paraphrases as JSON, preserving meaning, avoiding repeated keywords, staying "spiritually meaningful" (domain-tuned).
- The response is requested as `response_format={"type": "json_object"}` and parsed as `{"queries": ["q1", "q2", "q3"]}`; a parse failure logs an exception and falls back to an empty list rather than crashing.
- The original query is always appended (`queries.append(query)`) and the list is deduplicated while preserving order (`list(dict.fromkeys(queries))`) — so if the LLM's paraphrase collapses back to the original wording, it isn't retrieved twice.
- Each resulting query string is later fed through the full retrieval step (`HyDE` → hybrid search) independently inside `RetrievalPipeline.retrieve()`'s loop.

**Example**
```python
# File: app/services/transforms/multiquery.py
MULTIQUERY_PROMPT = """
You are an expert spiritual retrieval query generator.
Generate exactly 3 semantically similar queries.
Rules:
- preserve meaning
- use different wording
- avoid repeated keywords
...
Return valid JSON:
{ "queries": ["q1", "q2", "q3"] }
"""

class MultiQueryTransform(BaseQueryTransform):
    @observe(name="multiquery_transform")
    def transform(self, query: str):
        response = completion(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": MULTIQUERY_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        queries = data.get("queries", [])
        queries.append(query)
        return list(dict.fromkeys(queries))
```

**Where in this repo**
`app/services/transforms/multiquery.py::MultiQueryTransform.transform()`; called from `app/services/retrieval_pipeline.py::RetrievalPipeline.retrieve()` step 1 (`self.multiquery_transform.transform(query)`); constructed in `app/factories/retrieval_factory.py` with `model=settings.openai_model`.

**Interview angle**
Q: What's the cost/latency tradeoff of multi-query expansion?
A: It adds one full LLM round trip before retrieval even starts, and then multiplies retrieval work by the number of query variants (here, up to 4 — 3 generated + original) since each is independently embedded and searched. That's real added latency and API cost per chat turn. It's worth it here because the corpus is a large, loosely-structured spoken-transcript archive where a single fixed phrasing is genuinely likely to miss relevant material — the recall gain from casting a wider net outweighs the extra latency for a chat product that isn't latency-critical to the millisecond.

---

## HyDE (Hypothetical Document Embeddings)

**What it is**
Instead of embedding the user's literal query and searching with that, HyDE first asks an LLM to write a hypothetical *answer* to the query, and it's the embedding of that hypothetical answer that's used to search the vector store. The intuition: questions and answers occupy noticeably different regions of embedding space (a question is short, interrogative, and abstract; an answer is longer, declarative, and full of the same vocabulary as the source documents it's supposed to resemble). By generating something that already "sounds like" a passage from the corpus before embedding it, HyDE closes that question-answer gap and tends to retrieve more relevant passages than embedding the bare question would.

**How it works**
- `HydeTransform` wraps LlamaIndex's built-in `HyDEQueryTransform(include_original=True)`.
- `transform(query)` wraps the query string in a `QueryBundle` and runs it through the transform, which internally makes an LLM call to generate a hypothetical document answering the query.
- `include_original=True` means the transform keeps the original query text alongside the generated hypothetical document inside `embedding_strs` — the code here takes `embedding_strs[0]`, i.e. the generated hypothetical document, as the string that actually gets embedded and searched.
- In the retrieval pipeline, this happens *per query variant*: each of the (up to 4) queries from multi-query expansion is independently run through HyDE before being handed to the hybrid retriever, so HyDE and multi-query compose rather than replace each other.

**Example**
```python
# File: app/services/transforms/hyde.py
class HydeTransform:
    def __init__(self):
        self.hyde = HyDEQueryTransform(include_original=True)

    @observe(name="hyde_transform")
    def transform(self, query: str):
        query_bundle = QueryBundle(query)
        hyde_query = self.hyde.run(query_bundle)
        return hyde_query.embedding_strs[0]
```

**Where in this repo**
`app/services/transforms/hyde.py::HydeTransform.transform()`; called inside the per-query loop in `app/services/retrieval_pipeline.py::RetrievalPipeline.retrieve()` step 2 (`retrieval_query = self.hyde_transform.transform(q)`), before that query string is passed to the hybrid retriever.

**Interview angle**
Q: Doesn't generating a hypothetical answer risk retrieving based on a *wrong* or hallucinated answer?
A: Yes, in principle — if the LLM's hypothetical document confidently states something false, the embedding search will chase that fabrication rather than the truth. In practice it works because the search is over *embedding similarity*, not fact-checking: even a somewhat-wrong hypothetical answer usually uses the right vocabulary and framing for the topic, which is enough to land near the genuinely relevant chunks. The real answer is generated afterward from the retrieved (real) sources, not from the hypothetical document itself, so the hallucination risk is contained to a retrieval-quality question rather than leaking into the final answer.

---

## Deduplication (best-score-wins)

**What it is**
After every query variant (original + multi-query paraphrases, each possibly HyDE-transformed) has been independently retrieved, the same underlying chunk frequently shows up in more than one of those result lists — a paraphrase and the original query often surface overlapping evidence. Before doing anything else with the combined results, the pipeline collapses these duplicates down to one entry per unique `node_id`, keeping whichever occurrence had the highest similarity score. This matters most for keeping the reranking step's limited budget spent on distinct candidates rather than wasting rerank compute rescoring the same chunk five times because five query variants all found it.

**How it works**
- `unique_nodes_dict` is a plain dict keyed by `node_id`.
- For every node in every per-query result list: if the `node_id` hasn't been seen yet, or the current occurrence's score is higher than the one already stored, the dict entry is overwritten.
- The result is `unique_nodes_dict.values()` — one `NodeWithScore` per distinct chunk, carrying its best observed score across all query variants.
- This deduplicated count is logged and reported (`after_dedup`) in the Langfuse trace, and is also what the pipeline falls back to (sorted by score, truncated to `top_k`) if no fusion strategy is configured at all.
- Note that when fusion *is* configured (the default), the actual final ranking comes from RRF fusion over the raw `all_results` lists, not from this deduplicated set directly — dedup here primarily protects the no-fusion fallback path and gives an accurate "how much overlap did multi-query actually produce" signal for observability.

**Example**
```python
# File: app/services/retrieval_pipeline.py
unique_nodes_dict = {}
for node_list in all_results:
    for node_with_score in node_list:
        node_id = node_with_score.node.node_id
        # If new, OR this score beats the one we stored, keep this one
        if (node_id not in unique_nodes_dict
                or node_with_score.score > unique_nodes_dict[node_id].score):
            unique_nodes_dict[node_id] = node_with_score

unique_results = list(unique_nodes_dict.values())
total_raw = sum(len(l) for l in all_results)
logger.info(f"Deduplicated {total_raw} total nodes to {len(unique_results)} unique nodes.")
```

**Where in this repo**
`app/services/retrieval_pipeline.py::RetrievalPipeline.retrieve()`, step 3 (labeled `DEDUPLICATION` in the code); its output feeds the no-fusion fallback branch of step 4 and is reported in the Langfuse observation as `after_dedup`.

**Interview angle**
Q: Why dedupe by "best score wins" instead of, say, averaging scores or counting how many query variants retrieved it?
A: Best-score-wins is the simplest choice that avoids penalizing a chunk for being retrieved by fewer query variants when it was retrieved *strongly* by at least one. Averaging would drag down a genuinely excellent match that only one paraphrase happened to surface well. Counting occurrences is closer to what the separate RRF fusion step already does more rigorously (via rank-based accumulation) — so this dedup step is intentionally a cheap, simple guard rather than a second scoring model.

---

## Cross-encoder reranking

**What it is**
The fused candidate list — built from fast, approximate similarity search — is rescored by a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) that jointly encodes the (query, chunk) pair through one transformer forward pass, rather than comparing two independently-computed embeddings. Bi-encoders (what dense/sparse retrieval used above) are fast because a chunk's vector is precomputed once at index time and compared to a query vector with a cheap dot product, but that independence is also their weakness — the model never actually looks at the query and the chunk together. A cross-encoder is far more accurate at judging true relevance because it does look at both together, but it's too slow to run over an entire collection; the standard pattern (used here) is to let the cheap bi-encoder retrieval narrow the field, then spend the cross-encoder's cost only on that small shortlist.

**How it works**
- `CrossEncoderReranker` wraps LlamaIndex's `SentenceTransformerRerank(model=..., top_n=5)`.
- `rerank(query, nodes)` wraps the raw query string in a `QueryBundle` and calls `self.reranker.postprocess_nodes(nodes, query_bundle=query_bundle)`, which runs the cross-encoder over every (query, node.text) pair in the input list and returns the top `top_n` re-sorted by the cross-encoder's own relevance score.
- Before and after snapshots (`_node_summary`: rank, node_id, score, title, text preview) are captured and pushed into the Langfuse observation via `langfuse_context.update_current_observation`, so a full "before rerank" vs. "after rerank" ordering is visible per trace — one of the more useful debugging views in Langfuse for this pipeline.
- This is deliberately the *last* step in the pipeline (after dedup and fusion), since it's the most compute-expensive per-candidate operation — it should only ever see a small, already-deduplicated, already-fused candidate set.

**Example**
```python
# File: app/services/rerankers/cross_encoder.py
class CrossEncoderReranker(BaseReranker):
    def __init__(self, model: str, top_n: int):
        self.reranker = SentenceTransformerRerank(model=model, top_n=top_n)

    @observe(name="reranker")
    def rerank(self, query, nodes):
        query_bundle = QueryBundle(query)
        reranked_nodes = self.reranker.postprocess_nodes(
            nodes, query_bundle=query_bundle,
        )
        logger.info(f"Reranked {len(reranked_nodes)} nodes")
        return reranked_nodes
```

**Where in this repo**
`app/services/rerankers/cross_encoder.py::CrossEncoderReranker.rerank()`; constructed with `model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=5` in `app/factories/retrieval_factory.py`; invoked as step 5, the final step, in `app/services/retrieval_pipeline.py::RetrievalPipeline.retrieve()`.

**Interview angle**
Q: Why not just rerank *everything* and skip the earlier retrieval/fusion machinery?
A: Cross-encoders don't scale to full-collection search — running a transformer forward pass over every (query, chunk) pair in a corpus of thousands of chunks, for every single user query, is far too slow and expensive to be viable online. The two-stage design (cheap bi-encoder hybrid search to get a shortlist of tens of candidates, then an expensive cross-encoder only on that shortlist) is the standard way retrieval systems get near cross-encoder-quality relevance judgments without cross-encoder-scale cost — it's a recall/precision funnel, not redundancy.

---

## Retrieval pipeline orchestration

**What it is**
`RetrievalPipeline.retrieve()` is the single entry point that wires every concept above into one fixed sequence: multi-query expansion → (HyDE per query) → hybrid retrieval per query → deduplication → fusion → cross-encoder rerank. The stage order isn't arbitrary — each stage is deliberately placed to feed a smaller, higher-quality candidate set into the next, most expensive stage. The whole call is wrapped with a Langfuse `@observe` decorator, and it pushes a structured summary (`queries_generated`, `raw_retrieved`, `after_dedup`, `final_returned`, `reranked`) onto the trace, so a single Langfuse trace tells you exactly how many candidates survived each stage for any given production query.

**How it works**
- Step 1 — `queries = self.multiquery_transform.transform(query)` if a multi-query transform is configured, else just `[query]`.
- Step 2 — for each query variant, optionally run it through `self.hyde_transform.transform(q)` to get a hypothetical-document string, then call `self.retriever.retrieve(retrieval_query, top_k)` (the Qdrant hybrid retriever, itself already RRF-fused server-side). Each variant's results are appended to `all_results`, a list of lists.
- Step 3 — deduplicate across all variants' results, keeping the best score per `node_id` (see above).
- Step 4 — if a fusion strategy is configured, `self.fusion.fuse(all_results)` produces the merged ranking; otherwise fall back to sorting the deduplicated set by score and truncating to `top_k`.
- Step 5 — if a reranker is configured and there are candidates, `self.reranker.rerank(query, merged_nodes)` gives the final ordering (note: reranking always uses the *original* `query`, not any HyDE/paraphrase variant).
- All components (`retriever`, `multiquery_transform`, `hyde_transform`, `fusion`, `reranker`) are optional constructor arguments — any of them can be `None` and the pipeline degrades gracefully (e.g. no multi-query means just retrieving the raw query once).
- The whole pipeline object is built once by `app/factories/retrieval_factory.py::get_retrieval_pipeline()`, decorated with `@lru_cache(maxsize=1)` — one singleton pipeline for the process, not rebuilt per request.

**Example — worked trace for one query**
```text
query: "why do I feel disconnected from Krishna"
  ↓ multiquery_transform.transform()
4 unique queries (3 LLM paraphrases + original, dedup via dict.fromkeys)
  ↓ per query: hyde_transform.transform() → hybrid retriever.retrieve(top_k=10)
4 result lists, up to 10 nodes each  →  ~34 raw nodes total (some lists shorter)
  ↓ dedup (best score wins per node_id)
~19 unique nodes   (logged: "Deduplicated 34 total nodes to 19 unique nodes.")
  ↓ app-level RRF fusion over the 4 raw ranked lists
19 nodes, re-ranked by accumulated reciprocal rank
  ↓ cross-encoder rerank (top_n=5), scored against the ORIGINAL query text
5 final NodeWithScore objects returned to app/routers/chat.py
```

```python
# File: app/services/retrieval_pipeline.py
@observe(name="retrieval_pipeline")
def retrieve(self, query: str, top_k: int):
    queries = [query]
    if self.multiquery_transform:
        queries = self.multiquery_transform.transform(query)

    all_results = []
    for q in queries:
        retrieval_query = q
        if self.hyde_transform:
            retrieval_query = self.hyde_transform.transform(q)
        nodes = self.retriever.retrieve(retrieval_query, top_k)
        all_results.append(nodes)
    ...
    if self.fusion:
        merged_nodes = self.fusion.fuse(all_results)
    if self.reranker and merged_nodes:
        merged_nodes = self.reranker.rerank(query, merged_nodes)
    return merged_nodes
```

**Where in this repo**
`app/services/retrieval_pipeline.py::RetrievalPipeline.retrieve()`; assembled in `app/factories/retrieval_factory.py::get_retrieval_pipeline()`; called from the API layer in `app/routers/chat.py` (`pipeline.retrieve(query=safe_query, top_k=request.top_k)`) and `app/routers/retriever.py`.

**Interview angle**
Q: Walk me through the exact stage order and why rerank is last, not first.
A: Multi-query first, to widen the query set before spending any retrieval budget. HyDE next, per query, to improve what gets embedded before it hits the vector store. Then hybrid retrieval per variant — this is already server-side RRF-fused across dense/sparse signals. Then dedup, to collapse the overlap that multi-query inevitably produces. Then app-level RRF fusion, to merge the deduplicated-but-still-per-variant rankings into one list. Reranking is last and only ever touches that small, already-fused, already-deduplicated candidate set, because a cross-encoder forward pass per (query, chunk) pair is the most expensive operation in the whole pipeline — running it first, over unfiltered raw candidates, would be needlessly costly for no quality benefit, since the earlier stages already did the cheap work of narrowing the field.

---

## Parent-slot resolution

**What it is**
By the time reranking finishes, the pipeline is holding a short list of *child* nodes (256-token chunks) — good for having been precisely retrieved, but too small individually to be great LLM context. Because chunking is hierarchical, several of these top child nodes can legitimately share the same parent (e.g. two different paragraphs from the same 1024-token block both ranked highly). `ordered_parent_slots` walks the reranked child nodes in order, and for each one resolves its `parent_id`, keeping only the *first* occurrence of any given parent — so if the 2nd and 5th ranked children both belong to parent X, only one context slot is created for X, at the position where it first appeared (rank 2 in this example), rather than showing the LLM the same parent text twice.

**How it works**
- For each node, `parent_id` is read from `node.metadata["parent_id"]` (set back at indexing time in `push_child_nodes`), coerced to `str` or `None`.
- The dedup key is the `parent_id` if present, or a synthetic `f"__child__:{node_id}"` if the node genuinely has no parent (a defensive fallback — every child should have one, since chunking always produces a parent for each leaf).
- A `seen` set enforces first-occurrence-wins ordering: once a parent_id has produced a slot, later children pointing to the same parent are skipped.
- The output is a list of `(parent_id, NodeWithScore)` tuples in rank order — this list's length is the number of *distinct* context blocks that will actually be shown to the LLM, which can be smaller than the number of reranked child nodes.
- The caller (`app/routers/chat.py`) then batches all non-null `parent_id`s and fetches their full parent text in one shot via `pipeline.retriever.qdrant.retrieve_parents_by_ids(parent_ids)`, which does a direct Qdrant `retrieve()` by point ID against the parent collection (not a similarity search — an exact ID lookup).

**Example**
```python
# File: app/services/chat_context.py
def ordered_parent_slots(
    nodes: list[NodeWithScore],
) -> list[tuple[str | None, NodeWithScore]]:
    """First occurrence wins. Duplicate parent_ids share one slot."""
    seen: set[str] = set()
    slots: list[tuple[str | None, NodeWithScore]] = []
    for nws in nodes:
        meta = nws.node.metadata or {}
        raw_pid = meta.get("parent_id")
        pid = str(raw_pid) if raw_pid else None
        key = pid if pid else f"__child__:{nws.node.node_id}"
        if key in seen:
            continue
        seen.add(key)
        slots.append((pid, nws))
    return slots
```

**Where in this repo**
`app/services/chat_context.py::ordered_parent_slots()`; consumed by `app/services/chat_context.py::build_numbered_chat_context()` and directly by `app/routers/chat.py`, which also uses it to gather the `parent_ids` list passed to `indexing_pipeline/index_qdrant.py::QdrantClintManager.retrieve_parents_by_ids()`.

**Interview angle**
Q: This is where parent-child chunking actually pays off — how, concretely?
A: Retrieval and reranking operated entirely on the small, precise child chunks — that's what made the similarity search and cross-encoder scoring accurate. But nobody wants the LLM reading five disconnected 256-token fragments. `ordered_parent_slots` is the hand-off point: it takes the *ranking decision* made on children and translates it into which *parents* to actually show the model, deduplicating so a parent that "won" via two of its children isn't shown twice. Generation ends up reading full, coherent 1024-token blocks even though the ranking that selected them was done at a much finer grain.

---

## Numbered, cited context construction

**What it is**
`build_numbered_chat_context` turns the resolved parent slots into the actual text block the LLM sees, formatted as a sequence of `[1]`, `[2]`, `[3]`... labeled sections (each with title, URL, body text, and audio/video links if present), plus a parallel list of reference metadata dicts used later to render citations/links in the API response. This single mechanism does double duty: it's how grounding actually gets delivered to the model (the model is told, via the system prompt, to answer *only* from these numbered blocks), and the numbering itself acts as a lightweight hallucination deterrent — the model is instructed to cite a bracket number after any claim, which means it has to point at a specific block rather than asserting things unmoored from any source.

**How it works**
- Iterates `ordered_parent_slots(nodes)` with an incrementing 1-based index.
- For each slot, if a parent payload was successfully fetched from Qdrant, prefers the parent's `title`/`url`/`text`/`audio_links`/`video_links`/`source`, falling back to the child node's own metadata/text if no parent payload exists (defensive path for the synthetic no-parent case).
- Builds each block as plain text lines: `[{idx}] Title: ...`, optional `URL: ...`, `Text:` followed by the body, then explicit `Audio links: ...` / `(none)` and `Video links: ...` / `(none)` lines (always present, even when empty, so the LLM has a consistent format to parse) and an optional `Source:` line.
- Blocks are joined with a `\n\n---\n\n` separator into one big context string.
- In parallel, builds a `references` list of dicts (`index`, `title`, `url`, `node_id`, `audio_links`, `video_links`, `source`) — this is what the API layer likely surfaces back to the client alongside the answer text, separate from the raw text blob sent to the LLM.
- Returns `("", [])` if there were no slots at all, which `app/routers/chat.py` treats as a hard failure (HTTP 500 — "Could not build context from retrieval results").

**Example**
```python
# File: app/services/chat_context.py
for idx, (parent_id, nws) in enumerate(slots, start=1):
    ...
    lines = [f"[{idx}] Title: {title}"]
    if url:
        lines.append(f"URL: {url}")
    lines.append("Text:")
    lines.append(text)
    lines.append("Audio links: " + "; ".join(audio_links) if audio_links else "Audio links: (none)")
    lines.append("Video links: " + "; ".join(video_links) if video_links else "Video links: (none)")
    if source:
        lines.append(f"Source: {source}")

    blocks.append("\n".join(lines))
    references.append({
        "index": idx, "title": title or None, "url": url,
        "node_id": node_id, "audio_links": audio_links,
        "video_links": video_links, "source": source,
    })
```

**Where in this repo**
`app/services/chat_context.py::build_numbered_chat_context()`; called from `app/routers/chat.py` right after `ordered_parent_slots`/`retrieve_parents_by_ids`, producing `(context, ref_dicts)` — `context` goes to `llm_service.generate_chat_answer()`, `ref_dicts` presumably back into the `ChatResponse` for the client. See [05-app-and-api-design.md](05-app-and-api-design.md) for the rest of that request lifecycle.

**Interview angle**
Q: How does this design actually reduce hallucination, mechanically, rather than just by asking nicely?
A: It's a two-part contract enforced structurally, not just by prompt wording. Structurally, every piece of context the model receives is pre-numbered and self-contained — there's no ambiguity about what block `[3]` refers to. The system prompt (see the next section) then makes citing a bracket number the expected behavior after any claim. Together, this means a hallucinated claim isn't just "ungrounded" in some abstract sense — it's a claim with no bracket number attached, or a bracket number pointing at a block that doesn't actually support it, both of which are much easier to spot (by a human reviewer, or an automated eval like RAGAS faithfulness) than free-form hallucination would be.

---

## Grounded system prompt

**What it is**
`CHAT_SYSTEM` is the fixed system prompt sent with every chat completion request. It is the prompt-level half of this repo's hallucination mitigation strategy (the other half being the numbered-citation context format above): it explicitly instructs the model to answer only from the numbered sources, to cite inline, to say so when the sources don't answer the question, and never to invent URLs or facts. It also carries persona/tone instructions specific to this product (friendly, encouraging, addressing the user's emotional concern, matching the user's language) — so it's doing both grounding *and* product-voice work in one place. Generation is run at `temperature=0.2`, low enough to favor consistent, source-hugging phrasing over creative variation.

**How it works**
- `LLMService.generate_chat_answer(query, context, history)` builds the message list as: system prompt, then any prior turns (`history`), then a final user message that wraps the retrieved context and the question together: `f"Sources:\n\n{context}\n\n---\n\nQuestion: {query}"`.
- The `context` string here is exactly the numbered block text produced by `build_numbered_chat_context` — the model never sees raw retrieval scores or node IDs, just the formatted `[1]`, `[2]`... sections.
- Conversation `history` (fetched from Postgres via `chat_memory_store`, see [04-data-pipeline-and-infra.md](04-data-pipeline-and-infra.md)) is inserted as normal prior turns so the model can resolve follow-up/pronoun references, while the grounding rules in the system prompt still apply to the *current* answer.
- The call is wrapped with a Langfuse `@observe(as_type="generation")` decorator, so the exact prompt, completion, and model are captured per trace — useful for spotting drift between what the prompt asks for and what the model actually did.
- A separate, much simpler `generate_answer()` method exists without the citation/grounding rules or history — used for non-chat-context invocations elsewhere in the app.

**Example**
```python
# File: app/services/llm_service.py
CHAT_SYSTEM = """You are a helpful chat assistant answering from articles of chaitanya charan das.

Rules:
- Use only information supported by the provided sources, labeled [1], [2], etc.
- After statements that rely on a source, add the bracket citation with the matching number, e.g. [1].
- When audio or video URLs listed for a source are relevant, include them as markdown links.
- Do not invent URLs or facts beyond the sources.
- If the sources do not answer the question, say so briefly.
- Always answer in the same language as the question.
"""

@observe(name="llm_generate", as_type="generation")
def generate_chat_answer(self, query, context, history=None):
    messages = [{"role": "system", "content": CHAT_SYSTEM}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"Sources:\n\n{context}\n\n---\n\nQuestion: {query}"})
    completion = self._client.chat.completions.create(
        model=self._model, messages=messages, temperature=0.2,
    )
    return (completion.choices[0].message.content or "").strip()
```

**Where in this repo**
`app/services/llm_service.py::CHAT_SYSTEM`, `LLMService.generate_chat_answer()`; called from `app/routers/chat.py` as the last generative step before output guardrail scanning and PII de-anonymization (see [02-guardrails-and-safety.md](02-guardrails-and-safety.md)).

**Interview angle**
Q: If a chat answer looks ungrounded or hallucinated, where's the first place you'd look?
A: The prompt-level grounding rule is the cheapest, always-on layer, so it's the first thing to check when answers drift — is `CHAT_SYSTEM` still being sent, is `context` actually non-empty and well-formed (not silently truncated), and is the model actually being given the numbered blocks it's told to cite? Only after confirming the prompt and context are intact would I look upstream at retrieval quality (did the pipeline actually surface relevant sources for this query) — a perfect prompt can't ground an answer in sources that were never retrieved in the first place.
