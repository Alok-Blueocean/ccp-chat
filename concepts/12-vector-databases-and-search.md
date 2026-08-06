# Vector Databases & Search Concepts

General vector search theory. This repo's actual Qdrant usage (dual collections, 4 named vectors, server-side RRF) is in [01-retrieval-and-rag.md](01-retrieval-and-rag.md) and [04-data-pipeline-and-infra.md](04-data-pipeline-and-infra.md) — this file is the underlying "how does ANN search even work" layer.

## Approximate Nearest Neighbor (ANN) Search

**What it is**
ANN search finds the *k* vectors in a collection that are closest to a query vector, without guaranteeing it found the mathematically exact top-*k*. It trades a small, tunable amount of accuracy (recall) for a massive improvement in query speed and memory footprint. Exact nearest-neighbor search — computing the distance from the query to every single vector and sorting — is O(n) per query and simply does not scale past roughly a few hundred thousand to a million vectors at interactive latency. Every production vector database (Qdrant, Pinecone, Weaviate, Milvus, pgvector's HNSW mode) is built around an ANN index, not exact search. "Approximate" here doesn't mean unreliable — with well-tuned parameters, ANN indexes routinely hit 95-99%+ recall against exact search while being 10-100x+ faster.

**How it works**
- An index-building phase organizes vectors into a data structure (graph, clusters, trees) that lets search skip most of the collection.
- A query-time phase walks that structure, visiting only a small candidate set instead of every vector.
- Recall (how close the approximate top-*k* is to the true top-*k*) is controlled by index/query parameters — more candidates visited = higher recall, higher latency.
- Recall is measured empirically, typically as "recall@k" against a brute-force ground truth on a sample of queries — it's a dial you tune per use case, not a fixed guarantee the database gives you.
- Common families: graph-based (HNSW), cluster-based (IVF), tree-based (older, e.g. KD-trees — rarely used at scale today), hashing-based (LSH — largely superseded by graph methods for text embeddings).

**Example**
A collection of 5 million document chunks, each a 1536-dim OpenAI embedding. Exact search means 5 million dot products per query — maybe 200-500ms of raw compute even vectorized, before considering multiple concurrent users. An HNSW index over the same collection touches perhaps a few thousand vectors per query (a few hundred microseconds to low milliseconds of graph traversal), returning a top-10 that overlaps ~98% with the exact top-10 for a well-tuned index. That gap — 500ms exact vs. 2ms approximate for a 98%-as-good answer — is the entire reason ANN indexes exist.

**Interview angle**
Q: Is vector search always approximate, or can you get exact results if you want them?
A: At any real scale (millions of vectors), production systems always use an ANN index — exact brute-force search is a linear scan that becomes too slow to serve interactively. Recall is a tunable parameter (via `ef`, `nprobe`, etc. depending on the index), not a fixed property of "using a vector database." You can push recall close to 100% by cranking those knobs, but you pay for it in latency and sometimes memory — there's no free exact-and-fast option once you're past a small collection.

## HNSW (Hierarchical Navigable Small World)

**What it is**
HNSW is a graph-based ANN index: every vector becomes a node, and nodes are connected to their approximate nearest neighbors by edges, forming a proximity graph. The "hierarchical" part is what makes it fast — rather than one giant graph, HNSW builds several layers, where the top layer has very few nodes and very long-range connections (for fast coarse navigation), and each layer below has progressively more nodes and shorter connections, down to the bottom layer which contains every vector. It's the dominant ANN algorithm for text-embedding search today and is the index Qdrant, Weaviate, and pgvector's `hnsw` index type all use.

**How it works**
- **Layered structure**: layer assignment is probabilistic — most nodes only exist at layer 0 (the base layer, containing all vectors); progressively fewer nodes also exist at layer 1, layer 2, etc., roughly like a skip list. This gives sparse "highways" at the top and dense local connectivity at the bottom.
- **Greedy search descent**: a query starts at a fixed entry point in the topmost layer. At each layer, it greedily walks to whichever neighbor is closest to the query vector, until no neighbor improves the distance — then it drops down one layer and repeats, using the same starting point as where it got stuck.
- Once at layer 0, the search keeps expanding through a *dynamic candidate list* of size `ef` (search-time beam width) rather than stopping at the first local optimum, to avoid missing genuinely close vectors just past the current best.
- **`m`**: max number of edges per node (per layer). Higher `m` → denser graph → better recall, but more memory and slower index builds.
- **`ef_construction`**: the beam width used *while building* the graph — higher values produce a higher-quality graph (better recall later) at the cost of slower indexing.
- **`ef` (search)**: the beam width used *at query time* — higher values visit more candidates before returning, trading latency for recall. This is the knob you tune live without rebuilding the index.
- Insertions are incremental — new vectors can be added to a live HNSW graph without a full rebuild, which is why it fits databases that ingest continuously.

**Example**
Say `m=16`, `ef_construction=128`, `ef=64`, over 1M vectors. A query enters at the single top-layer entry point, say at layer 4, and in a couple of hops finds itself near the right neighborhood by layer 2 — this coarse phase costs almost nothing because layer 4 might have only ~50 nodes total. It then drops to layer 0 and expands a candidate list of up to 64 (`ef=64`) nodes among the query's local neighborhood (each node has up to 16 edges), computing real distances only for those ~64-300 vectors rather than 1,000,000. If recall on evaluation queries comes back at 92% and that's too low, bumping `ef` to 128 (more candidates explored at layer 0, no reindex needed) typically pushes recall toward 97-98% at roughly 2x the query latency.

**Interview angle**
Q: What are the key tunables in HNSW, and what do they trade off?
A: `m` controls edges per node (graph density — set at build time, affects memory and recall), `ef_construction` controls index build quality (set at build time, affects indexing speed and recall ceiling), and `ef` controls the search-time candidate list size (adjustable per-query without rebuilding, trades latency for recall). The practical operating pattern is: pick `m` and `ef_construction` once based on your recall/memory budget, then tune `ef` per query or per use case since it's cheap to change.

## IVF (Inverted File Index)

**What it is**
IVF is a cluster-based ANN approach: vectors are partitioned into a fixed number of clusters using something like k-means, each cluster gets a centroid, and at query time the search only looks inside the handful of clusters whose centroids are closest to the query — never touching the vectors in clusters that are far away. It's the classic index type behind FAISS's `IndexIVFFlat`/`IndexIVFPQ`, and it's generally faster and cheaper to build than HNSW, at the cost of typically lower recall for the same query latency (since it's a coarser, one-shot partitioning rather than a fine-grained navigable graph).

**How it works**
- **Training/clustering phase**: run k-means (or similar) over a representative sample of the vectors to produce `nlist` centroids.
- **Assignment**: every vector in the full collection is assigned to its nearest centroid and stored in that centroid's "inverted list" (bucket).
- **Query time**: compute the distance from the query vector to all `nlist` centroids (cheap — there are far fewer centroids than vectors), pick the `nprobe` closest centroids, and only exhaustively scan the vectors inside those `nprobe` buckets.
- `nlist` (number of clusters) and `nprobe` (how many clusters to search per query) are the main tunables — more `nprobe` means more of the collection gets scanned, raising recall and latency together.
- Often paired with Product Quantization (IVF-PQ) to compress the vectors stored inside each bucket, since IVF's flat variant still stores full vectors.
- Because clustering is data-dependent, IVF indexes need to be (re)trained on a representative sample — it doesn't degrade as gracefully as HNSW when the data distribution shifts significantly after training.

**Example**
1 million vectors, `nlist=100` clusters (so roughly 10,000 vectors per cluster on average). A query computes 100 centroid distances (trivial), then with `nprobe=8` scans only the 8 nearest clusters — about 80,000 vectors instead of 1,000,000, a 12.5x reduction in distance computations. Raising `nprobe` to 20 scans ~200,000 vectors: better recall (less chance the true nearest neighbor was sitting in a nearby-but-unprobed cluster), but 2.5x more work than `nprobe=8`.

**Interview angle**
Q: HNSW vs. IVF — when would you pick one over the other?
A: HNSW generally wins on recall-per-millisecond for in-memory workloads and supports efficient incremental inserts, which is why it's the default in most modern vector databases (Qdrant, Weaviate). IVF (especially IVF-PQ) builds faster, uses less memory per vector when combined with quantization, and is a natural fit for FAISS-style batch-built, less frequently updated indexes over very large static or slowly-changing collections. If asked "what else is out there besides HNSW," IVF is the answer that shows you know the field has more than one shape of tradeoff.

## Similarity Metrics

**What it is**
Similarity metrics define what "closest" means when comparing vectors. The three you'll encounter constantly are cosine similarity (angle between vectors, ignoring magnitude), dot product (angle *and* magnitude combined), and Euclidean/L2 distance (straight-line distance in vector space). Which one is "correct" isn't a free choice — it has to match how the embedding model was trained, because the model's training objective determines what geometric relationship in embedding space actually corresponds to semantic similarity. Using the wrong metric doesn't throw an error; it silently returns worse results.

**How it works**
- **Cosine similarity**: `cos(A,B) = (A·B) / (|A| × |B|)` — normalizes out vector length, so it only measures direction. Bounded in [-1, 1] for real vectors, and in [0, 1] for typical embedding models where vectors point in "similar" directions.
- **Dot product**: `A·B = Σ(Aᵢ × Bᵢ)` — no normalization, so both angle and magnitude affect the result. If vectors are already unit-normalized (common in many embedding pipelines), dot product and cosine similarity produce identical *rankings*.
- **Euclidean (L2) distance**: `√Σ(Aᵢ - Bᵢ)²` — straight-line distance; smaller means more similar (it's a distance, not a similarity, so ranking is ascending, not descending).
- The embedding model's training loss (e.g., contrastive loss with cosine similarity, or a dot-product-based loss) determines which metric is geometrically meaningful for that model's output space — most modern text embedding models (OpenAI's `text-embedding-3-*`, Cohere embed, sentence-transformers) are trained/evaluated with cosine similarity in mind.

**Example**
Take three tiny 3-dimensional vectors: `A = [1, 2, 3]`, `B = [2, 1, 3]`, `C = [2, 4, 6]` (note: `C` is just `A` scaled by 2 — same direction, different magnitude).

*A vs. B* (different direction, similar magnitude — `|A| = |B| = √14 ≈ 3.742`):
- Dot product: `(1×2)+(2×1)+(3×3) = 2+2+9 = 13`
- Cosine: `13 / (3.742 × 3.742) = 13/14 ≈ 0.929`
- Euclidean: `√((1-2)² + (2-1)² + (3-3)²) = √2 ≈ 1.414`

*A vs. C* (identical direction, `C` twice the length — `|C| = 2√14 ≈ 7.483`):
- Dot product: `(1×2)+(2×4)+(3×6) = 2+8+18 = 28` — much larger than A·B, purely because `C` is longer, not because it's "more similar" in direction.
- Cosine: `28 / (3.742 × 7.483) = 28/28 = 1.0` — perfectly identical direction, correctly reported as maximally similar regardless of magnitude.
- Euclidean: `√((1-2)²+(2-4)²+(3-6)²) = √(1+4+9) = √14 ≈ 3.742` — reports `C` as *farther* from `A` than `B` was, purely due to magnitude, even though `C` points in exactly the same direction as `A`.

This is the concrete illustration of why metric choice matters: by cosine, `C` is the best possible match to `A` (angle 0); by dot product, `B` scores higher than a naive glance would suggest is fair since dot product conflates "same direction" with "large magnitude"; by Euclidean, `C` looks like a worse match than `B` despite pointing the exact same way.

**Interview angle**
Q: Why does it matter which distance/similarity metric a vector index uses?
A: The embedding model's training procedure defines a geometric notion of "similar" — usually angular (cosine) similarity for modern text embedding models — and the index's distance metric needs to match that, or "closest" in the index stops corresponding to "most semantically similar" in reality. It's a silent failure mode: mismatched metrics don't error, they just quietly return worse retrieval results, which makes it an easy thing to get wrong in a pipeline and never notice until relevance quality is investigated.

## Dense vs. Sparse Retrieval

**What it is**
Dense retrieval represents text as a fixed-length, low-dimensional (relative to vocabulary size) vector of continuous numbers learned by an embedding model — every dimension is a learned, uninterpretable feature, and similarity is semantic (captures meaning, paraphrase, synonyms). Sparse retrieval represents text as a vector nearly as long as the vocabulary, where almost every dimension is zero and the nonzero entries correspond to specific terms with weights — either classic term-statistics-based (BM25/TF-IDF: nonzero exactly where a term appears, weighted by term frequency and rarity) or *learned sparse* (SPLADE: a transformer learns which vocabulary terms should be nonzero and how much, including terms that never literally appeared in the text — a learned form of query/document expansion, while still producing a sparse, interpretable vector).

**How it works**
- **Classic sparse (BM25/TF-IDF)**: nonzero weight for a term only if that exact term (or stem) appears in the text; weight increases with term frequency in the document and decreases with how common the term is across the whole corpus (inverse document frequency). Exact lexical match only — "car" will not match "automobile."
- **Learned sparse (SPLADE)**: a masked-language-model-style transformer produces a sparse weight vector over the full vocabulary, trained so that semantically related terms get nonzero weight even if absent from the original text (term expansion), while staying sparse via an explicit sparsity-inducing training objective (typically an L1-style regularization term).
- **Dense embeddings**: a transformer encodes text into a single dense vector (e.g. 1536 dimensions for `text-embedding-3-small`), trained via contrastive learning so that semantically similar texts land close together in vector space regardless of exact wording overlap.
- Dense excels at paraphrase/synonym/semantic-intent matches ("cheap flights" ≈ "affordable airfare"); sparse (especially classic) excels at exact-match precision for identifiers, codes, rare proper nouns, and acronyms that embeddings tend to blur together.
- Both are typically stored and searched the same way conceptually — nearest-neighbor over a vector space — but sparse vectors use specialized inverted-index-style storage (since most dimensions are zero) rather than a dense ANN graph, because comparing sparse vectors is really just intersecting nonzero term sets weighted by score.

**Example**
Query: "How do I reset my API key?" Against a document containing the phrase "regenerate your access token" and no literal occurrence of "reset," "API," or "key":
- Classic BM25 sparse retrieval scores this document near zero — no shared terms.
- SPLADE (learned sparse) can assign nonzero weight to "regenerate," "token," "access," "credential," "key" etc. in both the query's and document's sparse vectors (because it learned that these terms co-occur/are substitutable in training data), producing meaningful overlap and a nonzero score despite no exact lexical match.
- Dense embedding retrieval scores this document highly because "reset my API key" and "regenerate your access token" land close together in embedding space semantically, independent of exact wording.
This is exactly why SPLADE is described as "learned sparse" rather than classic sparse — it inherits sparse retrieval's exact-match strengths (interpretable, good at rare/technical terms it *has* seen) while gaining some of dense retrieval's ability to bridge vocabulary mismatch, without ever producing a dense, uninterpretable vector.

**Interview angle**
Q: Isn't SPLADE just BM25 with extra steps? Why bother with a "learned" sparse method at all?
A: No — classic BM25 can only ever score a nonzero match on literal (or stemmed) term overlap, while SPLADE is a trained model that predicts term *importance and expansion*, assigning weight to related vocabulary that never appeared in the original text. It keeps sparse retrieval's efficiency and interpretability (you can literally inspect which vocabulary terms got weight) while closing part of the vocabulary-mismatch gap that pure lexical matching suffers from. Conflating "sparse" with "BM25" is a common mistake — SPLADE is sparse but not classic keyword matching.

## Hybrid Search

**What it is**
Hybrid search runs two or more retrieval signals against the same query — most commonly a dense (semantic) vector search and a sparse (lexical/learned-sparse) search, though it can also mean fusing multiple dense signals (e.g. a title-vector search and a body-vector search) — and then merges the separate ranked result lists into one final ranking. The motivation is that dense and sparse retrieval have complementary failure modes: dense misses exact identifiers/rare terms, sparse misses paraphrases/synonyms, so combining them covers more query types than either alone.

**How it works**
- Each retrieval method runs independently and returns its own top-N ranked list (e.g. top-50 from dense, top-50 from sparse) — the underlying similarity scores are usually on completely different, incomparable scales (cosine ∈ [0,1] vs. an unbounded BM25-like score).
- **Reciprocal Rank Fusion (RRF)** fuses purely on *rank position*, not raw score, which sidesteps the score-normalization problem entirely: `RRF_score(doc) = Σ_lists 1 / (k + rank_in_that_list)`, summed over every list the document appears in (a document absent from a list simply contributes 0 from that list). `k` is a smoothing constant (commonly 60) that dampens the impact of small rank differences, especially near rank 1.
- **Weighted score combination** is the alternative: normalize each list's scores (e.g. min-max or z-score normalization) then combine as `α × dense_score + (1-α) × sparse_score`. This needs careful, often per-query-distribution normalization to be meaningful, which is exactly the headache RRF avoids.
- After fusion, the merged list is typically the candidate set fed into an optional reranking stage (see below) before being returned or passed to an LLM.

**Example**
Two ranked lists returned for the same query, `k = 60`:

| Rank | Dense list | Sparse list |
|---|---|---|
| 1 | D1 | D3 |
| 2 | D2 | D1 |
| 3 | D3 | D5 |
| 4 | D4 | D2 |
| 5 | D5 | D4 |

RRF score = `1/(60 + rank_dense) + 1/(60 + rank_sparse)` (using 0 if absent from a list):

- **D1**: dense rank 1, sparse rank 2 → `1/61 + 1/62 = 0.016393 + 0.016129 = 0.032523`
- **D3**: dense rank 3, sparse rank 1 → `1/63 + 1/61 = 0.015873 + 0.016393 = 0.032266`
- **D2**: dense rank 2, sparse rank 4 → `1/62 + 1/64 = 0.016129 + 0.015625 = 0.031754`

Fused order: **D1 (0.032523) > D3 (0.032266) > D2 (0.031754)** > D5 > D4.

Notice D1 wins the fusion even though D3 was ranked #1 in the sparse list — D1's consistent strength across *both* lists (1st and 2nd) edges out D3's single first-place finish paired with a mediocre 3rd place elsewhere. That's the core behavior RRF is prized for: it rewards documents both retrievers agree are good, rather than let one list's single top pick dominate.

**Interview angle**
Q: Why use RRF instead of just averaging normalized scores from each retriever?
A: RRF fuses on rank position rather than raw score, so it never needs to solve the genuinely hard problem of putting a bounded cosine similarity and an unbounded BM25-style score on the same numeric scale. It's simple, has one tunable constant (`k`), and empirically performs competitively with more complex score-normalization schemes in most hybrid-search setups — which is why it's the default fusion method in Qdrant, Elasticsearch, and most hybrid-search reference implementations.

## Quantization

**What it is**
Quantization compresses vector representations to reduce memory and storage footprint, at some cost to precision (and therefore recall). The three common flavors are scalar quantization (each float32 dimension → a lower-precision type, typically int8), binary quantization (each dimension collapses to a single bit), and product quantization (the vector is split into sub-vectors, each of which is separately clustered into a small codebook and represented by a compact code). Quantization matters because embedding vectors are the dominant memory cost in a vector database at scale — a 1536-dim float32 vector is 6 KB before any index overhead, and that number multiplied by tens of millions of vectors is where infrastructure budgets go.

**How it works**
- **Scalar quantization (float32 → int8)**: find the min/max range of values (per-vector or per-dimension across the collection), linearly map that continuous range onto the 256 discrete values an int8 can represent, store the int8 codes plus the min/max (or scale/offset) needed to approximately reconstruct the original floats.
- **Binary quantization**: each dimension is reduced to a single bit, typically by thresholding against zero (or the dimension's mean) — extremely aggressive compression, generally only usable well when paired with a rescoring pass, since a huge amount of information is discarded.
- **Product quantization (PQ)**: split each vector into `m` sub-vectors, run k-means separately on each sub-vector's dimensions across the whole collection to build a small codebook (e.g. 256 centroids) per sub-vector, then store just the centroid index for each sub-vector segment — this is a much finer-grained, learned compression than uniform scalar quantization.
- **Rescoring (oversampling)**: because quantized vectors lose precision, the standard pattern is to search the compressed/quantized index for a larger candidate set (e.g. top-200) than actually needed, then recompute exact distances using the original full-precision vectors for just that small candidate set, and return the true top-k from that rescoring — recovering most of the recall quantization would otherwise cost.

**Example**
A 4-dimensional embedding-like vector: `[0.15, -0.83, 0.42, 0.07]`. Per-vector min = `-0.83`, max = `0.42`, range = `1.25`.

Scalar-quantize to int8 (symmetric range [-128, 127]) via `q = round((x - min) / range × 255) - 128`:
- `0.15` → `round((0.98/1.25)×255) - 128 = round(199.9) - 128 = 200 - 128 = 72`
- `-0.83` → `round((0/1.25)×255) - 128 = 0 - 128 = -128`
- `0.42` → `round((1.25/1.25)×255) - 128 = 255 - 128 = 127`
- `0.07` → `round((0.90/1.25)×255) - 128 = 184 - 128 = 56`

Stored as int8 codes `[72, -128, 127, 56]` plus the two floats `min=-0.83, range=1.25` needed to dequantize. Dequantizing `72` back: `-0.83 + (72+128)/255 × 1.25 ≈ 0.150` — recovers the original value almost exactly.

Memory math: float32 uses 4 bytes/dimension, int8 uses 1 byte/dimension — a 4x reduction. For a real 1536-dim OpenAI embedding across 10 million vectors: float32 storage is `1536 × 4 × 10,000,000 ≈ 61.4 GB`; int8 scalar quantization brings that to `1536 × 1 × 10,000,000 ≈ 15.4 GB` — roughly 46 GB saved. Binary quantization (1 bit/dimension) would bring the same collection to `1536/8 × 10,000,000 ≈ 1.9 GB` — a ~32x reduction versus float32 — but with meaningfully more recall loss, which is why it's almost always paired with a full-precision rescoring pass over the top candidates.

**Interview angle**
Q: How would you serve a vector search index over 50+ million embeddings without a huge memory bill?
A: Quantize — scalar (int8, ~4x smaller, minimal recall loss) is the safe default; binary (~32x smaller) if the memory pressure is severe, but pair it with a rescoring step that recomputes exact distances on full-precision vectors for a small oversampled candidate set, since binary quantization alone loses too much precision to trust directly. This "search compressed, rerank a shortlist at full precision" pattern is the standard way to get most of the memory savings without eating the full recall cost.

## Payload Filtering

**What it is**
Payload filtering restricts a vector search to only the points whose metadata (payload) satisfies given conditions — e.g. `tenant_id = "acme-corp"` or `doc_type = "policy"` — and the critical implementation detail is *when* that filter is applied relative to the vector search itself. Applied *before or during* the ANN traversal ("pre-filtering" or "filtered search"), the search only ever considers matching points. Applied *after* ("post-filtering"), the vector search runs unfiltered first and matching results are discarded afterward — which can silently starve you of results when the filter is selective.

**How it works**
- Metadata fields intended for filtering need a payload index (e.g. Qdrant's keyword/integer/geo payload indexes) so filter evaluation itself is fast rather than a full scan.
- **Pre-filtering / filter-aware traversal**: modern engines like Qdrant push the filter check *into* the graph traversal — as HNSW walks the graph, it only considers/expands through nodes matching the filter, so the search never wastes its candidate budget on non-matching points.
- **Post-filtering**: run the vector search first (e.g. top-100 by similarity, filter-blind), then discard any of those 100 that fail the metadata condition — if the filter matches only 2% of the collection, most or all of the top-100 similarity results can fail the filter, leaving far fewer than the requested `limit`, or even zero results, despite plenty of matching points existing elsewhere in the collection.
- Filter selectivity is the deciding factor: a broad filter (matches 40% of data) barely matters whether it's pre- or post-applied; a narrow filter (matches 0.1% of data) makes post-filtering nearly useless.

**Example**
A Qdrant-style filtered search restricted to one tenant's policy documents:

```json
{
  "vector": [0.02, -0.15, ...],
  "filter": {
    "must": [
      { "key": "tenant_id", "match": { "value": "acme-corp" } },
      { "key": "doc_type",  "match": { "value": "policy" } }
    ]
  },
  "limit": 10
}
```

If `tenant_id = "acme-corp"` represents only 0.5% of a 10-million-point collection (~50,000 matching points) and the search naively did similarity search first over all 10M points and then filtered: the unfiltered top-100 nearest neighbors would very plausibly contain zero points belonging to `acme-corp`, since 99.5% of the collection can't match — the query would return an empty or near-empty result set despite 50,000 perfectly good candidates existing. A filter-aware pre-filtered search instead confines the entire graph traversal to that 50,000-point subset from the start, correctly finding the true top-10 within it.

**Interview angle**
Q: Why does it matter whether filtering happens before or after the vector search?
A: With a selective filter, post-filtering (search first, discard non-matches after) can return far fewer results than requested — or none — because the unfiltered similarity search may never surface any matching points in its candidate window. Pre-filtering (or filter-aware traversal, as in Qdrant's HNSW implementation) confines the search itself to the matching subset, so it correctly finds the true nearest matches within that subset. This distinction — knowing filter placement is an implementation detail with real correctness consequences, not just a performance nuance — is what separates "used a vector DB" from "understands how it's implemented."

## On-Disk / Memory-Mapped Vectors

**What it is**
On-disk (memory-mapped/mmap) vector storage keeps the vector data on disk rather than fully resident in RAM, using the operating system's virtual memory system to page data in on demand as it's accessed. This trades some query latency for the ability to serve collections far larger than available RAM — instead of "how much RAM do I need to hold my whole collection," the question becomes "how much RAM do I need for a good hot-page cache," which is a much smaller number for most real access patterns.

**How it works**
- The vector file lives on disk; the OS maps it into the process's virtual address space via `mmap`, so reading a vector looks like a normal memory access but may trigger a page fault that pulls the relevant disk block into the OS page cache transparently.
- Frequently accessed ("hot") vectors end up cached in RAM by the OS's normal page cache behavior, so repeated queries touching the same regions get near-RAM-speed access after the first read.
- Cold reads pay real disk I/O latency — this is why SSD (and ideally NVMe) storage is effectively mandatory for on-disk vector search; HDD random-read latency (measured in milliseconds per seek) makes on-disk mode impractical, while SSD random reads (tens of microseconds) keep it viable.
- An explicit LRU (or similar) cache layered on top of the raw OS page cache lets the application prioritize which vectors/index structures stay warm, rather than relying purely on OS heuristics — useful when access patterns are skewed and predictable (e.g. a small set of frequently-queried documents).
- Index structures (like the HNSW graph itself) can also be memory-mapped, not just raw vectors — Qdrant, for instance, supports mmap for both vector storage and index segments independently.

**Example**
A collection of 200 million vectors at 1536 dimensions in float32 would need `200,000,000 × 1536 × 4 bytes ≈ 1.14 TB` of RAM to hold entirely in memory — often more RAM than a single machine reasonably has, or a very expensive machine if it does. With on-disk/mmap storage on NVMe SSDs, the same collection can be served from a machine with, say, 64 GB of RAM: the OS page cache naturally keeps the most-recently/most-frequently accessed portions of the 1.14 TB file warm in that 64 GB, and queries touching cold regions pay an extra few hundred microseconds to low milliseconds of SSD read latency per page fault instead of failing to fit in memory at all.

**Interview angle**
Q: Your vector collection has grown too large to fit in RAM — what do you do?
A: Move to on-disk/memory-mapped vector storage on SSD (not spinning disk — random-read latency there kills the approach), and lean on the OS page cache (optionally backed by an explicit LRU cache) to keep hot vectors fast while cold ones pay a disk-read penalty. This converts a hard "won't fit in RAM" wall into a soft latency/cost tradeoff, which is generally the right shape of tradeoff to have in a production system.

## Reranking (Cross-Encoder Re-scoring)

**What it is**
Reranking is a second-pass scoring step applied to a small shortlist of already-retrieved candidates (from vector search, hybrid fusion, or both), using a more expensive but more accurate relevance model — typically a cross-encoder that jointly processes the query and each candidate document together, rather than encoding them independently as retrieval-stage embeddings do. It's the standard way to recover precision that fast approximate retrieval (and any quantization/compression along the way) leaves on the table, without paying the cross-encoder's cost across the entire collection.

**How it works**
- Retrieval (dense/sparse/hybrid + fusion) returns a shortlist — commonly 20-100 candidates — optimized for recall and speed over the full collection.
- A cross-encoder model takes the query and *each* candidate document as a single joint input (e.g. `[CLS] query [SEP] document [SEP]`) and outputs a relevance score per pair — this captures query-document interaction directly, which independently-encoded embeddings structurally cannot, at the cost of needing one full model forward pass per candidate.
- Because the cross-encoder only runs on the shortlist (tens of candidates), not the full collection (millions), its per-item cost is affordable even though it would be far too slow to run at full-collection scale.
- The reranked order becomes the final result set (or the final context fed to an LLM), typically truncated to a much smaller top-k (e.g. top-5) than the shortlist size.

**Example**
A hybrid search + RRF fusion step returns a top-50 shortlist for a query. A cross-encoder (e.g. a `ms-marco-MiniLM` style reranker, or a hosted reranking API) scores all 50 query-document pairs directly — computationally trivial at 50 items even though the model is far too slow to run against the full multi-million-document collection. The final top-5 handed to the LLM might reorder candidates significantly versus the fused ranking: a document that only ranked #22 in the fused list, because both retrievers under-weighted a subtle phrasing match, can jump to #1 after the cross-encoder directly evaluates the actual query-document relevance rather than relying on separately-encoded vector similarity.

**Interview angle**
Q: If you already have hybrid search with RRF fusion, why add reranking on top?
A: Retrieval-stage similarity (dense, sparse, or their fusion) scores query and document independently and combines the scores after the fact, which is fundamentally a weaker signal than a cross-encoder that looks at the query and document *together* and can pick up on interaction effects neither retrieval signal saw alone. Reranking is affordable specifically because it only runs on a small shortlist, not the full collection — it's the standard "cheap broad retrieval, then expensive narrow re-scoring" two-stage pattern used across modern search and RAG systems.

## Vector DB Landscape

**What it is**
The vector database space spans purpose-built systems (Qdrant, Pinecone, Weaviate, Milvus) and vector *extensions* to existing general-purpose databases (pgvector for Postgres, similar extensions exist for other databases). The real decision isn't "which one has the best benchmark numbers" — it's how much of your stack you want to be a dedicated vector-search system versus how much you'd rather fold into infrastructure you already run and operate.

**How it works / how they differ**
- **Qdrant**: open-source, self-hostable or managed cloud, HNSW-based, strong payload filtering support, named/multi-vector collections (multiple vectors per point) — this repo's choice.
- **Pinecone**: fully managed only (no self-hosted option), historically simple to operate with less infrastructure ownership, but less architectural flexibility/visibility than open-source options.
- **Weaviate**: open-source, HNSW-based, has built-in modules for embedding generation and hybrid search out of the box.
- **Milvus**: open-source, supports multiple index types (HNSW, IVF, and others) with a strong focus on very large-scale deployments and distributed architecture.
- **pgvector**: a Postgres extension, not a separate database — vectors live as a column type alongside your normal relational data, queryable with regular SQL joins. Supports both exact and HNSW-approximate search, but generally trails dedicated vector databases on advanced ANN feature depth (e.g. more limited filtering/quantization ergonomics) and large-scale ANN performance tuning.
- The recurring tradeoff: dedicated vector databases give more ANN performance headroom, richer filtering, and purpose-built scaling; folding vectors into an existing relational database (pgvector) gives you one fewer system to operate, deploy, back up, and keep consistent — at some ceiling on ANN performance and feature depth.

**Example**
A team already running Postgres for their application's relational data (users, orders, chat history) adds a `pgvector` column to store document embeddings alongside existing tables, and does `SELECT * FROM docs WHERE tenant_id = $1 ORDER BY embedding <=> $2 LIMIT 10` — a single SQL query that joins relational filtering and vector search with zero new infrastructure. That's attractive at moderate scale (hundreds of thousands to low millions of vectors) specifically because there's no second database to run. Past that scale, or once requirements grow to need e.g. fine-grained filtered-HNSW performance, multiple named vectors per point, or built-in quantization tuning, teams often migrate to a dedicated vector database like Qdrant purpose-built for that workload.

**Interview angle**
Q: This repo already runs Postgres for chat memory — why use Qdrant instead of just pgvector?
A: pgvector is the right call when you want one fewer system to operate and your scale/feature needs are modest — it turns vector search into a SQL query against data you already have relationally. Once you need things like fine-grained filtered ANN search at scale, multiple named vectors per point (e.g. separate title/body/dense/sparse vectors on the same document, as this repo uses), or index-level quantization controls, a dedicated engine like Qdrant gives more headroom and better ergonomics for exactly those needs, at the cost of running a second system. It's a real production tradeoff — "one less system to operate" is a legitimate reason to pick pgvector, not just a fallback for teams who didn't know better.
