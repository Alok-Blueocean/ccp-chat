# CCP Concept Map — At-a-Glance Reference

Fast-scan reference for every technical concept in this repo, one glance per row.
For deep "walk me through your reasoning" answers, see [`../INTERVIEW_QA.md`](../INTERVIEW_QA.md) — that file argues a point; this directory just names the pieces and points at the code.

## What this project is

A RAG chatbot over a spiritual-philosophy article corpus (chaitanya charan das). Two halves:

- **Serving path** — FastAPI app (`app/`) that takes a query, retrieves grounded context from Qdrant, and generates a cited answer.
- **Indexing path** — pipelines (`indexing_pipeline/`, `dags/`) that scrape/ingest source articles and push them into Qdrant + Algolia.

## Architecture at a glance

```
                              ┌─────────────────────────────────────────────────────┐
                              │                    SERVING PATH                     │
                              │                                                      │
 Client ──POST /chat──▶ rate_limiter ──▶ input_guard ──▶ RetrievalPipeline          │
                              │  (Redis)      (llm-guard:      │                    │
                              │                token/inj/tox)  ▼                    │
                              │                          multiquery (LLM, 3 variants)│
                              │                                │                    │
                              │                          ┌─────┴─────┐              │
                              │                          ▼           ▼              │
                              │                    HyDE transform  (skip)           │
                              │                          │           │              │
                              │                          ▼           ▼              │
                              │                   QdrantHybridRetriever              │
                              │                (dense+sparse, RRF fused in Qdrant)   │
                              │                          │                          │
                              │                     dedupe (best score wins)        │
                              │                          │                          │
                              │                    app-level RRF fusion             │
                              │                          │                          │
                              │                  cross-encoder rerank (top 5)        │
                              │                          │                          │
                              │              resolve parent chunks (full context)   │
                              │                          │                          │
                              │            PII anonymize ──▶ LLM (gpt-4o) ──▶ ...   │
                              │                          │                          │
                              │        output_guard ──▶ PII deanonymize ──▶ answer  │
                              │                          │                          │
                              │           Postgres (session history) ◀──────────────┤
                              │           Langfuse trace + async RAGAS score        │
                              └─────────────────────────────────────────────────────┘

                              ┌─────────────────────────────────────────────────────┐
                              │                    INDEXING PATH                    │
                              │                                                      │
  CSV of articles ──▶ [batch DAG: Airflow .expand()]  ──┐                          │
  CSV of articles ──▶ Redis Stream ──▶ [streaming DAG, cron */2min,                │
                                        consumer group + XACK]  ──┤                 │
                                                                    ▼                │
                                                          extract_page_content       │
                                                                    │                │
                                                     HierarchicalChunker             │
                                                     (parent 1024 / child 256 tok)   │
                                                                    │                │
                                                        ┌───────────┴───────────┐    │
                                                        ▼                       ▼    │
                                                   Algolia index          Qdrant (parent + child
                                                   (keyword search)        collections, dense+sparse)│
                              └─────────────────────────────────────────────────────┘
```

## Concept files

Two layers: files 01–05 are **grounded in this repo's actual code** (every row names a real file/function). Files 06+ are **general framework/industry knowledge** — not things this repo uses, but concepts an AI engineering interview will assume you know regardless of what any one project does. Where relevant, the general files cross-reference the project-specific ones (and note where this repo deliberately does something *different*, e.g. LlamaIndex instead of LangChain).

**This project, concretely:**

| File | Covers |
|---|---|
| [01-retrieval-and-rag.md](01-retrieval-and-rag.md) | Chunking strategy, hybrid search, fusion, HyDE, multi-query, reranking |
| [02-guardrails-and-safety.md](02-guardrails-and-safety.md) | Input/output scanning, PII anonymization, rate limiting |
| [03-evaluation-and-observability.md](03-evaluation-and-observability.md) | RAGAS, DeepEval, lexical metrics, Langfuse tracing |
| [04-data-pipeline-and-infra.md](04-data-pipeline-and-infra.md) | Airflow DAGs, Redis Streams, dual vector collections, Postgres memory |
| [05-app-and-api-design.md](05-app-and-api-design.md) | FastAPI lifespan, singleton factories, request lifecycle, citations |

**General AI/LLM engineering knowledge (framework-level, not tied to this repo):**

| File | Covers |
|---|---|
| [06-fastapi-concepts.md](06-fastapi-concepts.md) | ASGI, DI, middleware, background tasks, lifespan, testing |
| [07-ai-agents-concepts.md](07-ai-agents-concepts.md) | ReAct, tool calling, agentic loops, memory, multi-agent, failure modes |
| [08-guardrails-and-llm-safety-concepts.md](08-guardrails-and-llm-safety-concepts.md) | Prompt injection, jailbreaking, moderation, framework landscape |
| [09-langchain-concepts.md](09-langchain-concepts.md) | LCEL/Runnables, chains, retrievers, memory, tools, tracing |
| [10-langgraph-concepts.md](10-langgraph-concepts.md) | StateGraph, nodes/edges, checkpointing, human-in-the-loop |
| [11-llm-fundamentals-and-prompting.md](11-llm-fundamentals-and-prompting.md) | Tokens, context windows, sampling, embeddings, fine-tuning vs RAG |
| [12-vector-databases-and-search.md](12-vector-databases-and-search.md) | ANN (HNSW/IVF), similarity metrics, quantization, filtering |
| [13-mcp-and-tool-calling.md](13-mcp-and-tool-calling.md) | MCP host/client/server, tools vs resources, transports, security surface |

## How to use this before an interview

1. Skim the architecture diagram until you can redraw it from memory.
2. Read each concept table top to bottom — for every row, ask yourself "could I explain *why* this exists, not just what it does?"
3. For files 01–05: if a row doesn't click, open the file path listed and read the real code — every entry there points at something concrete, not a paraphrase.
4. For files 06+: these are framework knowledge you're expected to have independent of this project — but notice where this repo makes a *different* choice (LlamaIndex over LangChain, a fixed pipeline over an agent, llm-guard over NeMo Guardrails) and be ready to justify it.
