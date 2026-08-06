# CCP Project — Interview Q&A

---

## RAG Pipeline

**Q: Your RAG system returns wrong answers sometimes. How do you debug whether it's a retrieval problem or an LLM problem?**

Split the investigation into two stages. First check retrieval independently — log the raw chunks returned for the failing query and manually verify if the right content is present. If the right chunk is not retrieved, the problem is in the vector search (wrong embedding model, poor chunking, index not updated). If the right chunk *is* retrieved but the answer is still wrong, the problem is in the LLM (bad prompt, hallucination, context too long, model not following instructions). In the CCP project, `lexical_context_recall` and `reference_context_hit` metrics in `evaluate_testset.py` are specifically designed to isolate retrieval quality before the LLM is even involved.

---

**Q: You changed the chunk size from 500 to 1000 tokens. How do you measure if it actually improved quality?**

Run the RAGAS evaluation pipeline before and after the change using the same test set (`qdrant_ragas_testset.csv`). Compare `context_recall` (did we retrieve all needed content), `context_precision` (did we retrieve only relevant content), and `faithfulness` (did the answer stay grounded). Larger chunks improve recall but hurt precision — you retrieve more content but more of it is noise. The right chunk size depends on your document structure and query type. In CCP, the parent-child chunking strategy handles this — small child chunks for precise retrieval, large parent chunks sent to the LLM for full context.

---

**Q: A user asks a multi-hop question that requires combining information from 3 different documents. How does your system handle that, and where does it fail?**

Standard vector search retrieves based on semantic similarity to the query, which works well for single-document questions. Multi-hop questions fail because the query embedding is close to document A, but the answer requires connecting A → B → C. The system will retrieve chunks from A and miss B and C entirely. Mitigation strategies include query decomposition (break the multi-hop question into sub-questions, retrieve separately, combine), graph-based retrieval (build a knowledge graph and traverse edges), or re-ranking (retrieve more chunks, then use a cross-encoder to rerank). In CCP, multi-hop questions are a known weak point — the `synthesizer_name` metadata in the RAGAS test set helps identify if failing cases are systematically multi-hop.

---

**Q: Why did you use parent-child chunking instead of simple chunking? What problem does it solve?**

Simple chunking stores and retrieves fixed-size chunks. If the chunk boundary cuts through a relevant passage, the retrieved text is incomplete or loses context. Parent-child chunking stores small child chunks for retrieval (precise semantic matching) but returns the full parent chunk (larger context) to the LLM. This means retrieval stays accurate (small chunks match queries precisely) while the LLM gets enough surrounding context to generate a grounded answer. Without it, you face a tradeoff: small chunks = good retrieval, bad generation; large chunks = bad retrieval, good generation. Parent-child solves both simultaneously.

---

**Q: Your vector search returns semantically similar chunks but they don't contain the answer. What do you do?**

This is a semantic gap problem — the query and the answer use different vocabulary even though they mean the same thing. Options: (1) **Hybrid search** — combine dense vector search with BM25 keyword search and merge results; keyword search catches exact terminology that embeddings miss. (2) **HyDE (Hypothetical Document Embeddings)** — use the LLM to generate a hypothetical answer, embed that, and search with it; the hypothetical answer uses the same vocabulary as the actual answer. (3) **Query expansion** — generate multiple paraphrases of the query and search with all of them. (4) **Reranking** — retrieve top 20 with vector search, then use a cross-encoder to rerank and keep top 5.

---

## Evaluation

**Q: How did you create your evaluation dataset? Why not just write test cases manually?**

The CCP evaluation dataset is synthetically generated using RAGAS `TestsetGenerator`. It fetches real document chunks from Qdrant, re-splits them, and uses an LLM (GPT-4o-mini) to generate realistic questions a user might ask, along with ground truth answers derived from the actual content. Manual writing doesn't scale — you need 100+ cases to get statistically meaningful scores, and manually writing 100 domain-specific Q&A pairs for a spiritual philosophy corpus is impractical. The tradeoff is that synthetic ground truth quality depends on the generator LLM. For the 5 CI cases in `test_llm_quality.py`, they are hand-written because those need to be 100% trusted as a regression gate.

---

**Q: RAGAS gave you a mean faithfulness of 0.78. Is that good or bad? How do you decide?**

A score in isolation means nothing. You need: (1) a baseline — what was it before the last change? If it was 0.85 and dropped to 0.78, that's a regression worth investigating. (2) A floor — below 0.5 is generally a sign the LLM is making up content not in the retrieved context. (3) Distribution context — mean 0.78 with std 0.05 is very different from mean 0.78 with std 0.25; the second hides cases scoring below 0.3. The outlier detection in CCP catches these — the mean looked acceptable but 6 individual cases had faithfulness below 0.35. For a domain like spiritual philosophy where factual precision matters, 0.78 is borderline; for a casual chatbot it might be fine.

---

**Q: You found 6 low outliers in context_recall. Walk me through how you'd investigate and fix them.**

First, open `_metrics_per_case.csv` and identify the 6 failing queries. Look for patterns — are they all long questions, multi-hop, or about a specific topic that may be underrepresented in the index? Then manually run retrieval for one failing query and inspect the raw chunks returned. If the right content exists in Qdrant but wasn't retrieved — embedding model or `top_k` issue; increase `top_k` or switch to hybrid search. If the right content doesn't exist in Qdrant — indexing gap; check the source documents and re-run the indexing pipeline. If content exists and was retrieved but RAGAS still scores it low — the reference context from the test set may be stale or the chunk boundary changed after re-indexing.

---

**Q: What's the difference between context_precision and context_recall? Give an example where one is high and the other is low.**

- **Context precision**: of the chunks you retrieved, what fraction were actually relevant? High precision = clean retrieval, no noise.
- **Context recall**: of all the content needed to answer the question, what fraction did you retrieve? High recall = you didn't miss anything important.

Example where precision is high, recall is low: Query is "What are all the side effects of this drug?" Your retriever returns 3 chunks, all highly relevant — but the answer requires 5 pieces of information and you only retrieved 3. Everything you got was useful, but you missed some.

Example where recall is high, precision is low: You retrieve 10 chunks, all 5 needed pieces of information are in there — but 5 other chunks are irrelevant noise. The LLM now has to work through noisy context, increasing the chance of a wrong or hallucinated answer.

---

**Q: Your synthetic test set was generated from the same corpus your retriever indexes. Is that a problem?**

Yes, partially. It means the test set only covers topics that exist in your corpus — you won't catch failures on out-of-domain queries. More importantly, if your chunking strategy or indexing changes, the reference contexts in the test set become stale (they reflect the old chunk boundaries). This means context_recall scores can drop not because retrieval got worse, but because the reference contexts no longer match your current index. The right mitigation is to regenerate the test set after major indexing changes, and to supplement with real user queries from production logs as ground truth when available.

---

## Deployment & Architecture

**Q: Why is the app deployed as a Lambda container instead of a regular server?**

The CCP bot is a Telegram webhook handler — traffic is completely bursty (messages arrive unpredictably, then nothing for hours). A persistent server wastes money during idle periods. Lambda scales to zero automatically and you pay only per invocation. The container approach (rather than a zip deploy) is used because the app has heavy native dependencies — Tesseract OCR, OpenCV, numpy — which exceed Lambda's 250MB zip limit. A container image has no size limit. The tradeoff is cold start latency (first request after idle period is slower) and the 15-minute execution limit, though the async invocation pattern works around the latter.

---

**Q: Telegram has a 29-second response timeout. How did you work around it?**

API Gateway (which receives the Telegram webhook) has a hard 29-second timeout. LLM calls with retrieval can take 10–30+ seconds. If the Lambda doesn't respond in time, Telegram marks the webhook as failed and retries, causing duplicate processing. The solution in CCP is a two-Lambda pattern: the first Lambda (HTTP path via Mangum) receives the Telegram update, immediately fires a second Lambda with `InvocationType=Event` (async — no wait for response), and returns 200 to Telegram within milliseconds. The second Lambda (async path) runs the actual retrieval + LLM work with no time pressure. This decouples acknowledgement from processing.

---

**Q: What does Mangum do, and why do you need it?**

FastAPI is an ASGI framework — it expects HTTP requests in ASGI format. AWS Lambda receives events as JSON dicts from API Gateway, which is a completely different interface. Mangum is an adapter that translates API Gateway JSON events into ASGI-compatible requests, passes them to FastAPI, and translates the FastAPI response back into the Lambda response format. Without Mangum, you'd have to manually parse the API Gateway event and construct the response dict, essentially reimplementing HTTP handling from scratch.

---

## Hallucinations

**Q: How do you detect hallucinations in your RAG system?**

Three layers: (1) **Offline — RAGAS faithfulness metric**: after generating answers from the test set, RAGAS uses an LLM judge to check each claim in the answer against the retrieved context. Claims not supported by the context are flagged. (2) **Offline — DeepEval HallucinationMetric**: `test_llm_quality.py` has explicit hallucination probe cases — a fabricated answer is fed to the metric and expected to fail, verifying the detection works. (3) **Online — output guardrail**: `llm-guard` output scanners can flag responses that contain content inconsistent with the context at serving time. The most reliable signal in practice is faithfulness score below 0.5 combined with a high answer_relevancy — the model answered confidently but invented the content.

---

**Q: How do you prevent hallucinations at the prompt level?**

Several techniques used together: (1) **Explicit grounding instruction** in the system prompt — "Answer only using the provided context. If the context does not contain enough information, say so explicitly." (2) **Numbered context** — in CCP, `build_numbered_chat_context()` builds a numbered list of source chunks, making it easy for the LLM to cite and stay anchored. (3) **Refusal instruction** — tell the model to say "I don't know" rather than guess; LLMs hallucinate partly because they're trained to always produce a response. (4) **Temperature 0** — deterministic output reduces creative fabrication. (5) **Short context windows** — don't flood the LLM with 20 chunks; 3–5 high-quality chunks reduces the chance the model ignores them and draws from training memory instead.

---

**Q: What's the difference between hallucination and faithfulness in RAGAS?**

They measure similar things but from different angles. **Faithfulness** (RAGAS) measures what fraction of claims in the generated answer are supported by the retrieved context — it's a precision measure on the answer's claims. **Hallucination** (DeepEval) measures what fraction of the context is contradicted by the answer — it's about contradiction specifically, not just unsupported claims. A model can have low faithfulness (making claims not in the context) without high hallucination (not contradicting the context) — for example, the model adds extra true facts from its training data that aren't in the retrieved chunks. That's unfaithful but not hallucination. Both matter: faithfulness for grounding, hallucination for factual correctness.

---

**Q: Your system is hallucinating specific facts — dates, numbers, names. How do you fix it?**

Numerical and named entity hallucinations are the hardest because the LLM has strong priors from training data. Steps: (1) **Check if the fact exists in your corpus** — if it's not indexed, the model will always fall back to training memory. Index the missing content. (2) **Increase context specificity** — retrieve chunks that contain the exact fact; use metadata filters (document type, date range) to narrow retrieval. (3) **Add fact-checking instruction** — "Do not state specific dates, numbers, or names unless they appear verbatim in the context." (4) **Post-processing** — extract named entities and numbers from the answer and verify each appears in the retrieved context; flag or reject answers where they don't match. (5) **Use a stronger model** — smaller models hallucinate specific facts more than larger ones; GPT-4o vs GPT-4o-mini makes a measurable difference on factual precision.

---

**Q: How do you test for hallucinations in CI?**

In CCP's `test_llm_quality.py`, two complementary tests run: (1) A faithful answer (one that stays within the context) is tested against `HallucinationMetric(threshold=0.5)` and expected to pass. (2) A fabricated answer (one with specific claims not in the context, like a made-up year) is tested against the same metric and expected to **fail** — the test catches this with `pytest.fail` if the metric incorrectly passes it. Testing both directions is important: if you only test that faithful answers pass, a broken metric (one that always returns "no hallucination") would make all your tests green while catching nothing.

---

## Not Performing Over Time

**Q: Your system worked well at launch but quality degraded over 3 months. Why might that happen?**

Several root causes: (1) **Index drift** — new documents were added but old ones weren't updated or re-chunked; the index no longer reflects the current state of the knowledge base. (2) **Distribution shift** — users are asking different types of questions now than at launch; the test set doesn't cover new query patterns. (3) **Model changes** — OpenAI or the embedding provider silently updated their models, shifting embedding space; old vectors are now slightly misaligned with new query embeddings. (4) **Prompt rot** — the system prompt was tweaked multiple times for edge cases and is now inconsistent. (5) **Data quality** — new documents added to the corpus are lower quality (poor formatting, different language, corrupted encoding) and the chunking pipeline didn't filter them out. In CCP, the `is_english()` and `clean_text()` filters in `test_case_generation.py` exist partly to prevent this last issue.

---

**Q: How do you monitor RAG quality in production over time?**

Three mechanisms: (1) **Periodic batch eval** — run `evaluate_testset.py` on a schedule (weekly or after each deploy) and track mean scores over time. Store results in a database or S3 and plot trends. A drop of more than 5% in any metric triggers investigation. (2) **User feedback signals** — thumbs up/down, follow-up clarification questions ("that's not right"), or session abandonment are implicit quality signals. Log them and correlate with retrieval metadata. (3) **LLM-as-judge on live traffic** — sample 5–10% of real queries in production, run faithfulness and answer_relevancy scoring asynchronously (not in the request path), and track rolling averages. Langfuse (referenced in `evaluate_langfuse.py`) is the tool used in CCP for this — it traces LLM calls and allows offline evaluation of production traffic.

---

**Q: What is index drift and how do you handle it?**

Index drift happens when the documents in your source (database, file system, SharePoint) change but your vector index is not updated — the index becomes stale. Answers start referencing outdated information, or the system can't answer questions about recently added content. Handling it: (1) **Event-driven re-indexing** — whenever a document is updated or added, trigger a pipeline run for just that document; delete old vectors by `document_id` and insert new ones. (2) **Periodic full re-index** — for smaller corpora, re-index everything nightly or weekly. (3) **Versioned vectors** — store a `last_updated` timestamp in each vector's payload; periodically scan for vectors whose source document has a newer modification time and re-index those. In CCP, the Airflow DAGs in the `dags/` directory orchestrate scheduled indexing runs.

---

**Q: Your evaluation scores drop after adding 10,000 new documents. What do you investigate?**

Systematically: (1) **Retrieval noise** — more documents means more candidates; vectors that were previously top-5 may now be pushed out by new documents that are semantically similar but less relevant. Check `context_precision` specifically. (2) **Encoding issues** — new documents may have different character encodings, languages, or formatting; run the same `clean_text()` and `is_english()` filters and check how many new chunks were filtered vs passed. (3) **Duplicate content** — new documents may overlap significantly with existing ones; near-duplicate chunks confuse retrieval by splitting relevance score across multiple similar results. (4) **Collection size limit** — if using HNSW indexing in Qdrant, very large collections may need tuned `ef_construction` and `m` parameters to maintain recall. (5) **Test set staleness** — if the reference contexts in your test set came from the old index, they won't match the new chunk boundaries.

---

## Handling Huge Vector Store

**Q: Your Qdrant collection has 10 million vectors. How do you keep retrieval fast?**

Several levers: (1) **HNSW tuning** — Qdrant uses HNSW (Hierarchical Navigable Small World) graph for ANN search. Increase `m` (graph connectivity) and `ef` (search beam width) for better recall, but this trades off against memory and latency. For 10M vectors, start with `m=16, ef=128` and benchmark. (2) **Quantization** — scalar quantization (int8) or binary quantization reduces vector memory footprint by 4–32x with minimal recall loss; Qdrant supports this natively. (3) **Payload indexing + filtering** — if queries can be pre-filtered by metadata (document type, date, category), add payload indexes on those fields; Qdrant applies filters before vector search, dramatically reducing the search space. (4) **Sharding** — distribute the collection across multiple Qdrant nodes; each shard handles a subset of vectors and results are merged. (5) **On-disk vectors** — store vectors on disk with mmap; slower than RAM but enables collections larger than available memory.

---

**Q: How do you handle a vector store that's too large to fit in memory?**

Qdrant supports memmap (memory-mapped) storage — vectors live on disk and are paged into RAM on demand. This allows collections much larger than available RAM, at the cost of higher latency for cold pages. For read-heavy workloads, combine with: (1) an LRU cache for frequently accessed vectors (hot documents that are queried often stay in RAM), (2) SSD storage instead of HDD (random read latency matters hugely for mmap), (3) prefetching — if your query patterns are predictable (certain document categories queried more in the morning), pre-warm the cache. For truly massive scale, consider tiered storage — hot recent documents in a fast in-memory collection, older documents in an on-disk collection — and merge results at query time.

---

**Q: Your vector store has duplicate or near-duplicate chunks. How do you detect and clean them?**

Exact duplicates are easy — hash the chunk text and deduplicate during indexing. Near-duplicates are harder: (1) **MinHash/LSH** — locality-sensitive hashing groups similar texts in the same bucket without comparing all pairs; fast for large corpora. (2) **Embedding cosine similarity** — compute pairwise similarity between chunk embeddings; pairs above 0.97 are near-duplicates. At 10M vectors this is O(n²) — only feasible in batches or with ANN approximation. (3) **Qdrant payload dedup** — store a content hash in the payload at index time; before inserting a new chunk, scroll for existing chunks with the same hash and skip if found. Near-duplicates hurt retrieval because the similarity score gets split across multiple redundant chunks — the retriever returns 3 copies of the same passage instead of 3 different relevant passages, reducing coverage.

---

**Q: How do you do incremental indexing without re-indexing everything?**

Track a `last_indexed_at` timestamp per document in a metadata store (Postgres in CCP). The indexing pipeline queries for documents modified after `last_indexed_at`. For each updated document: (1) delete all existing vectors with that `document_id` from Qdrant using a payload filter, (2) re-chunk and re-embed the new version, (3) insert new vectors, (4) update `last_indexed_at`. This ensures the index always reflects the current document state without touching the 99% of documents that haven't changed. The risk is a window where old vectors are deleted but new ones aren't yet inserted — mitigate by inserting first, then deleting, using the document version as a tag to distinguish old from new vectors during the transition.

---

**Q: How do you handle embedding model upgrades without re-indexing 10 million vectors?**

You can't mix vectors from different embedding models in the same collection — their dimensions and semantic spaces are incompatible. Options: (1) **Blue-green index** — build the new index in a separate Qdrant collection in parallel while the old one serves traffic; switch the application pointer once the new index is complete and validated. (2) **Dual-write** — during the migration window, write to both old and new collections, query from old; once new is fully populated, switch queries to new. (3) **Canary** — route 5% of traffic to the new index, measure quality metrics, gradually increase. The cost is embedding 10M chunks again — for OpenAI `text-embedding-3-small`, this is roughly $0.02 per million tokens; estimate your total token count and budget accordingly.

---

## The Sharpest Questions (Most Likely to Be Asked)

**Q: Retrieval problem or LLM problem?**
Always isolate by logging raw retrieved chunks for the failing query first. If the right content is there, it's the LLM. If it's not, it's retrieval. Never blame the LLM until you've verified retrieval.

**Q: Synthetic dataset from the same corpus — is that a problem?**
Yes. It means you only test what you've indexed, not what users actually ask. Supplement with real production queries. Also, the reference contexts become stale after re-indexing.

**Q: CI passes but production quality drops.**
Your CI uses mocked LLM calls or a fixed golden dataset. Production has real users, real queries, and live model versions. The gap is covered by periodic batch eval on real traffic + LLM-as-judge monitoring (Langfuse).

**Q: Mean faithfulness 0.78 — good or bad?**
Meaningless without a baseline, a floor, and the distribution. Always look at the outliers and the trend, not the single number.

**Q: System worked at launch but degraded over 3 months.**
Index drift + distribution shift + silent model updates are the three most common causes. Monitor with periodic batch eval and production traffic sampling.
