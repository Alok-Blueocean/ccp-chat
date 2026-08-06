# App & API Design Concepts

How the FastAPI process itself is put together: what happens once at boot, what gets rebuilt (or not) on every request, how the two HTTP surfaces divide responsibility, how a stateless server fakes statefulness across turns, and the exact order of operations a `/chat` call goes through end to end.

## FastAPI Lifespan Startup

**What it is**
The app never handles a request cold. Everything expensive and one-time — opening the Postgres connection pool, making sure the `chat_messages` table exists, and loading every enabled guardrail's ML model into memory — happens once, inside FastAPI's `lifespan` context manager, before `yield` hands control to the server. Anything after `yield` runs on shutdown. This turns "first request is slow" into "startup is slow, every request after that is fast," which is the right trade for a service that will field thousands of requests per process lifetime.

**How it works**
- `lifespan()` is an `@asynccontextmanager` registered on `FastAPI(lifespan=lifespan)`.
- Before `yield`: `init_pool()` opens a `psycopg_pool.ConnectionPool` (min 1 / max 10 connections) against `DATABASE_URL`; `ensure_schema()` reads `postgres/schema.sql` and executes every statement so the schema exists idempotently; `_prewarm_guards()` loads whichever of the input guard, PII guard, and output guard are enabled via settings flags.
- Each guard's loader (`_load_scanners`, `prewarm`) is itself wrapped in `functools.lru_cache`, so pre-warming and the first real request both hit the same cached model — the prewarm call's only job is to pay that cost before traffic arrives instead of during the first user's request.
- Guards that are toggled off in config (`guardrail_input=false`, etc.) skip model loading entirely — no wasted memory or startup time for disabled features.
- After `yield`: `close_pool()` releases the Postgres connections on shutdown.

**Example**
```python
# File: main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_pool()
    ensure_schema()
    _prewarm_guards()
    yield
    close_pool()

app = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)
```

**Where in this repo**
`main.py::lifespan`, `main.py::_prewarm_guards`; pool lifecycle in `postgres/client.py::init_pool/ensure_schema/close_pool`; per-guard loaders in `app/guardrails/input_guard.py`, `app/guardrails/pii_guard.py::prewarm`, `app/guardrails/output_guard.py`.

**Interview angle**
Q: Why not just lazily initialize the DB pool and models on first use?
A: Lazy init would push the entire cost — DB connect handshake plus loading a NER model and two scanner models — onto whichever unlucky user sends the first request, and that cost would be paid again after every restart. `lifespan` makes that cost a deployment-time property (how long the container takes to become "ready") instead of a user-facing latency spike, which also plays nicely with container orchestrators that gate traffic on a readiness check.

## Singleton Pipeline Factory

**What it is**
The full retrieval pipeline — the Qdrant hybrid retriever, the multi-query transform, HyDE, RRF fusion, and the cross-encoder reranker — is an expensive object graph: it holds a Qdrant client and loads a transformer-based reranker model. `get_retrieval_pipeline()` is decorated with `@lru_cache(maxsize=1)`, so no matter how many times or from where it's called, the constructor body runs exactly once per process; every subsequent call returns the same cached `RetrievalPipeline` instance.

**How it works**
- `@lru_cache(maxsize=1)` on a zero-argument function is the idiomatic Python "build once, memoize forever" pattern — the cache key is trivially the empty argument tuple, so there's only ever one cached entry.
- Inside the factory: build a `QdrantClintManager`, wrap it in `QdrantHybridRetriever`, construct `MultiQueryTransform` (uses the configured OpenAI model to generate query variants), `HydeTransform`, `RRFFusion`, and a `CrossEncoderReranker` loading `cross-encoder/ms-marco-MiniLM-L-6-v2` with `top_n=5` — then assemble all of them into one `RetrievalPipeline`.
- `chat.py` calls `get_retrieval_pipeline()` directly as a module-level function call inside the handler.
- `retriever.py` instead wires the *same* function through FastAPI's dependency system via a thin wrapper, `create_retrieval_pipeline()`, passed as `Depends(create_retrieval_pipeline)`. Because that wrapper just calls the `lru_cache`d factory, both routers still end up sharing the one cached pipeline — FastAPI's dependency resolution doesn't defeat the cache, it just adds a layer of indirection for testability (a test can override the dependency without touching `chat.py`'s direct call).
- The reranker's cross-encoder model, the embedding model behind the retriever, and the Qdrant client connection are all now shared, in-memory, thread-safe-by-construction singletons for the life of the process.

**Example**
```python
# File: app/factories/retrieval_factory.py
@lru_cache(maxsize=1)
def get_retrieval_pipeline() -> RetrievalPipeline:
    qdrant_manager = QdrantClintManager()
    retriever = QdrantHybridRetriever(qdrant_manager)
    multiquery = MultiQueryTransform(model=settings.openai_model)
    hyde = HydeTransform()
    fusion = RRFFusion()
    reranker = CrossEncoderReranker(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=5,
    )
    return RetrievalPipeline(
        retriever=retriever, multiquery_transform=multiquery,
        hyde_transform=hyde, fusion=fusion, reranker=reranker,
    )

def create_retrieval_pipeline() -> RetrievalPipeline:
    return get_retrieval_pipeline()
```

**Where in this repo**
`app/factories/retrieval_factory.py::get_retrieval_pipeline` (the singleton) and `::create_retrieval_pipeline` (the `Depends`-friendly wrapper used by `app/routers/retriever.py`); consumed directly in `app/routers/chat.py`.

**Interview angle**
Q: What would go wrong if you built this pipeline fresh on every request instead of caching it?
A: You'd reload the cross-encoder reranker weights and re-establish a Qdrant client connection on every single call, adding real latency (model load, not inference) and memory churn to every request instead of paying it once. `lru_cache(maxsize=1)` is the simplest correct fix — no explicit global variable, no manual "is it initialized yet" check, and it's trivially thread-safe because CPython's GIL serializes the first-call race.

## Two API Surfaces, Two Purposes

**What it is**
The app exposes two routers with deliberately different jobs. `/retriever/search` is a debugging and inspection endpoint: it runs retrieval only, no LLM call, and returns raw scored nodes so you can see exactly what the retrieval pipeline found for a query. `/chat` is the real product endpoint: retrieval plus guardrails plus memory plus generation plus citations, returning a conversational answer. Splitting these means retrieval quality and generation quality can be debugged and iterated on independently, and the cheap path (`/retriever/search`, no OpenAI completion call) never gets conflated with the expensive one.

**How it works**
- `retriever.py`'s `search` handler takes `query`/`top_k` as plain query parameters (not a Pydantic body), scans the input for injection/safety issues via `scan_input`, calls the shared pipeline's `.retrieve()`, and serializes each `NodeWithScore` into a flat dict (`score`, `node_id`, `text`, `parent_id`, `title`, `url`) — no dedup by parent, no LLM call, no persistence.
- It wraps the call in a manually-created Langfuse `trace`/`span` pair (`lf.trace(...)`, `trace.span(...)`) rather than the `@observe` decorator that `chat.py` uses — an inconsistency worth noticing, but both land in the same Langfuse project.
- `chat.py`'s `chat` handler takes a typed `ChatRequest` body, is decorated with `@observe(name="chat")` for automatic tracing, and runs the full pipeline described in "Full Request Lifecycle" below.
- Both routers depend on rate limiting (`chat_rate_limit` / `retriever_rate_limit`) and both call into the exact same cached retrieval pipeline — the only real difference is how much happens *after* retrieval.

**Example**
```python
# File: app/routers/retriever.py
@router.post("/search")
def search(
    query: str,
    top_k: int,
    lf: Langfuse = Depends(get_langfuse),
    pipeline=Depends(create_retrieval_pipeline),
    _rl=Depends(retriever_rate_limit),
):
    safe_query = scan_input(query)
    results = pipeline.retrieve(query=safe_query, top_k=top_k)
    output = [
        {"score": float(n.score), "node_id": n.node.node_id, "text": n.node.text,
         "parent_id": n.node.metadata.get("parent_id"),
         "title": n.node.metadata.get("title"), "url": n.node.metadata.get("url")}
        for n in results
    ]
    return {"results": output}
```

**Where in this repo**
`app/routers/retriever.py::search` mounted at `/retriever` in `main.py`; `app/routers/chat.py::chat` mounted at `/chat`.

**Interview angle**
Q: Why would you want a retrieval-only endpoint in production at all?
A: Because a bad chat answer has at least two independent failure points — the retriever found the wrong chunks, or the LLM said something wrong given the right chunks — and without a way to isolate the first from the second, every bug report turns into re-running the whole expensive pipeline with an LLM call just to rule out retrieval. `/retriever/search` lets you answer "did we even find the right source?" in one cheap, deterministic call.

## Session-Based Multi-Turn Memory

**What it is**
The FastAPI process itself holds no per-user state between requests — there's no in-memory dict of "conversation objects." Continuity across turns is entirely reconstructed from Postgres on every request, keyed by a client-held `session_id` UUID. A new conversation omits `session_id` and the server mints one; every response — first turn or twentieth — echoes the `session_id` back so the client can pass it on the next call.

**How it works**
- `ChatRequest.session_id` is `Optional[str]`; when absent, `chat()` generates `uuid.uuid4()`. When present, it's parsed with `uuid.UUID(...)` and a malformed value raises `HTTPException(422)` — you cannot smuggle in an arbitrary string as a session key.
- The resolved `session_uuid` is used both to look up prior history (`chat_memory_store.fetch_recent_messages`) and to persist the new turn (`chat_memory_store.append_turn`) at the end of the same request.
- `ChatResponse.session_id` is always populated (`str(session_uuid)`), so the contract is: "send back whatever session_id I gave you, or none at all for a new chat."
- Because everything lives in Postgres rather than process memory, any replica behind a load balancer can serve any turn of any conversation — horizontal scaling doesn't fragment sessions.

**Example**
```python
# File: app/routers/chat.py
if request.session_id:
    try:
        session_uuid = uuid.UUID(request.session_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="session_id must be a valid UUID string.",
        ) from exc
else:
    session_uuid = uuid.uuid4()
```

**Where in this repo**
`app/models/schemas.py::ChatRequest.session_id` / `ChatResponse.session_id`; session resolution in `app/routers/chat.py::chat`; storage in `app/services/chat_memory.py::ChatMemoryStore`.

**Interview angle**
Q: Why hand session-id generation to the server instead of letting the client pick a UUID itself?
A: Two reasons: it means new clients need zero knowledge of UUID generation to start a conversation (omit the field, get one back), and the server retains full control over the ID format and validation — a client-generated ID would still need server-side validation anyway, and centralizing generation removes a whole class of "malformed or colliding session_id" bugs before they exist.

## Bounded History Window

**What it is**
When building the prompt for a turn, the server doesn't load a session's entire conversation history — it fetches only the most recent 40 stored messages (20 user/assistant turn pairs, since each turn writes one user row and one assistant row). This is a deliberate cap, not an accident of a `LIMIT` clause someone forgot to raise: unbounded history would eventually make every request in a long conversation slower and more expensive as the prompt grows, and could exceed the model's context window outright.

**How it works**
- `_DEFAULT_MESSAGE_LIMIT = 40` is a module-level constant in `chat_memory.py`, passed as the default `limit` argument to `fetch_recent_messages`.
- The SQL orders by `created_at DESC, id DESC LIMIT 40` — i.e. it fetches the *newest* 40 rows first (so the `LIMIT` actually bounds the right end of the conversation), then the Python code reverses the list (`list(reversed(rows))`) to restore oldest-first order, because LLM chat APIs expect messages in chronological order.
- The tie-break on `id DESC` alongside `created_at DESC` matters: if two messages land in the same timestamp (common with fast successive inserts), ordering by timestamp alone is ambiguous — the numeric `id` gives a stable secondary sort.
- Rows come back as `(role, content)` tuples and are mapped into OpenAI-style dicts (`{"role": ..., "content": ...}`), ready to splice directly into the `messages` list sent to the chat completion call.
- If the Postgres pool was never initialized (no `DATABASE_URL`), `fetch_recent_messages` returns `[]` rather than raising — chat still works, just without memory.

**Example**
```python
# File: app/services/chat_memory.py
_DEFAULT_MESSAGE_LIMIT = 40  # 20 user/assistant turns

def fetch_recent_messages(self, session_id: UUID, limit: int = _DEFAULT_MESSAGE_LIMIT):
    cur.execute(
        """
        SELECT role, content FROM chat_messages
        WHERE session_id = %s
        ORDER BY created_at DESC, id DESC
        LIMIT %s
        """,
        (str(session_id), limit),
    )
    rows = list(reversed(cur.fetchall()))
    return [{"role": r[0], "content": r[1]} for r in rows]
```

**Where in this repo**
`app/services/chat_memory.py::ChatMemoryStore.fetch_recent_messages`, constant `_DEFAULT_MESSAGE_LIMIT`.

**Interview angle**
Q: What happens to a conversation once it passes 20 turns — does the model just forget everything before that?
A: Yes, effectively — only the most recent 20 turns are ever sent as history, so anything earlier silently drops out of the model's context on every subsequent call; there's no summarization or compression step recovering it. That's an explicit trade-off in this codebase: a fixed, cheap, predictable context cost versus perfect long-conversation recall. A production upgrade path would be summarizing older turns into a running digest instead of hard-truncating them.

## Full Request Lifecycle

**What it is**
The `/chat` endpoint is the one place every subsystem in this app touches in a single call: rate limiting, input safety, retrieval, context assembly, memory, PII masking, generation, output safety, persistence, and background evaluation all execute in one deterministic order inside `chat()`. Being able to narrate this chain precisely — and explain *why* each step sits where it does — is the single best signal of actually understanding the system rather than having skimmed it.

**How it works** (in exact order, per `app/routers/chat.py::chat`)
1. **Rate limit** — `Depends(chat_rate_limit)` runs before the handler body even starts; a sliding-window check against Redis, `429` if the caller's IP has exceeded the configured chat limit in the current window.
2. **Resolve session** — parse `request.session_id` into a UUID, or mint a new `uuid4()` if absent; malformed input raises `422`.
3. **Input safety scan** — `scan_input(request.query)` runs the input guardrail (prompt-injection / toxicity / ban-substrings scanners) and returns a sanitized `safe_query`.
4. **Start the Langfuse trace** — `langfuse_context.update_current_trace(name="chat", session_id=..., input=safe_query)` attaches session and input to the `@observe`-created trace for this request.
5. **Retrieve** — `pipeline.retrieve(query=safe_query, top_k=request.top_k)` runs the full retrieval pipeline (multiquery + HyDE + hybrid search + RRF fusion + rerank) on the *sanitized but not yet PII-masked* query, on the theory that masking before retrieval would hurt recall against an index built on real text. No results → `404`.
6. **Dedup to parent slots** — `ordered_parent_slots(nodes)` collapses multiple child chunks that share a `parent_id` into one slot each, preserving first-occurrence order.
7. **Fetch parent payloads** — `pipeline.retriever.qdrant.retrieve_parents_by_ids(parent_ids)` pulls full parent-document text/metadata for those slots directly from Qdrant.
8. **Build retrieved_contexts for eval** — truncate each parent's text to 1500 chars, used later only for the background RAGAS scoring call, not for the prompt itself.
9. **Build the numbered chat context** — `build_numbered_chat_context(nodes, parent_payloads)` produces the `[1]..[n]` labeled context blocks and parallel reference metadata. Empty context → `500`.
10. **Fetch history** — `chat_memory_store.fetch_recent_messages(session_uuid)`, bounded to the last 40 messages.
11. **PII-mask the query** — `PiiContext.create()` builds a per-request `Anonymize`/`Deanonymize` pair sharing one `Vault`; `pii.anonymize(safe_query)` replaces detected entities (names, emails, etc.) with fake stand-ins *right before* it's sent to the LLM — deliberately the last possible moment, so every step before this (retrieval, tracing input, logging) still sees real user text, and only the third-party LLM API sees masked text.
12. **Generate** — `llm_service.generate_chat_answer(llm_query, context, history=history or None)` calls the OpenAI chat completion with system prompt + history + sources + question.
13. **Output safety scan** — `scan_output(llm_query, answer)` runs the output guardrail (e.g. relevance/no-harmful-content checks) against the masked query and raw answer.
14. **De-anonymize** — `pii.deanonymize(llm_query, safe_answer)` swaps the fake entities in the *answer* back to the real ones the user provided, using the same request-scoped `Vault` — so a user who mentions their own name gets it back in the reply, but it never round-tripped through OpenAI in the clear.
15. **Close out the trace** — `langfuse_context.update_current_trace(output=final_answer)`.
16. **Schedule background scoring** — if a trace ID exists, `background_tasks.add_task(_score_ragas_background, ...)` schedules RAGAS `faithfulness`/`answer_relevancy` scoring to run *after* the HTTP response is already sent, keyed to the same Langfuse trace.
17. **Persist the turn** — `chat_memory_store.append_turn(session_uuid, safe_query, final_answer)` writes both the user and assistant messages to Postgres in one INSERT.
18. **Respond** — return `ChatResponse(answer=final_answer, references=[...], session_id=str(session_uuid))`.

**Example**
```python
# File: app/routers/chat.py (abridged skeleton of the ordered chain)
def chat(request: ChatRequest, background_tasks: BackgroundTasks,
         _rl: None = Depends(chat_rate_limit)) -> ChatResponse:
    session_uuid = ...                              # 2
    safe_query = scan_input(request.query)           # 3
    nodes = get_retrieval_pipeline().retrieve(...)    # 5
    context, ref_dicts = build_numbered_chat_context(nodes, parent_payloads)  # 9
    history = chat_memory_store.fetch_recent_messages(session_uuid)          # 10
    pii = PiiContext.create()
    llm_query = pii.anonymize(safe_query)             # 11
    answer = llm_service.generate_chat_answer(llm_query, context, history=history)  # 12
    safe_answer = scan_output(llm_query, answer)       # 13
    final_answer = pii.deanonymize(llm_query, safe_answer)  # 14
    background_tasks.add_task(_score_ragas_background, ...)  # 16
    chat_memory_store.append_turn(session_uuid, safe_query, final_answer)  # 17
    return ChatResponse(answer=final_answer, references=[...], session_id=str(session_uuid))
```

**Where in this repo**
`app/routers/chat.py::chat` (the whole chain lives in this one function); supporting calls into `app/guardrails/input_guard.py`, `app/factories/retrieval_factory.py`, `app/services/chat_context.py`, `app/services/chat_memory.py`, `app/guardrails/pii_guard.py`, `app/services/llm_service.py`, `app/guardrails/output_guard.py`.

**Interview angle**
Q: Why does PII anonymization happen right before the LLM call rather than at the same time as the input safety scan in step 3?
A: The input scan and PII masking solve different problems and need different inputs. Input scanning (step 3) is about rejecting or cleaning adversarial/unsafe *queries* early, and retrieval (step 5) needs the real, unmasked text for best recall against the vector index — masking a name to a fake one before embedding it would hurt retrieval quality for no benefit, since Qdrant and the reranker are trusted internal components. PII masking only needs to happen in front of the one component that's an external third party receiving raw text over the network — the LLM API — so it's deferred to the last possible step before that call, and reversed immediately after, on the same request-scoped vault, before anything else (output scan, persistence, response) sees the answer.

## Citation-Aware Context Assembly

**What it is**
Retrieval returns a flat list of scored chunks, often with several chunks from the same parent document. Before that goes anywhere near the LLM, `build_numbered_chat_context` collapses duplicates down to one slot per parent, numbers the surviving slots `[1]..[n]`, and attaches each slot's title, URL, and audio/video links. The system prompt then instructs the model to cite `[1]`, `[2]`, etc. after any claim it draws from a source, and the same numbering is returned to the client as structured `ReferenceItem`s — so a citation in the answer text and an entry in `references` refer to the exact same source by construction, not by coincidence.

**How it works**
- `ordered_parent_slots(nodes)` walks the retrieved nodes in their existing (already reranked) order, and for each one computes a dedup key: the node's `parent_id` metadata if present, otherwise a synthetic `__child__:<node_id>` key. The first node to claim a key wins the slot; later nodes sharing that key are dropped. This preserves rank order while guaranteeing one citation number per underlying document, even if the retriever surfaced three chunks from the same article.
- For each surviving slot, `build_numbered_chat_context` prefers the full parent payload (fetched separately from Qdrant by parent ID) over the child node's own metadata/text — parents carry the canonical `title`/`url`/`audio_links`/`video_links`, children only carry what was indexed on the chunk.
- Each block is formatted as plain text: `[i] Title: ...`, optional `URL: ...`, `Text: ...`, then explicit `Audio links: ...`/`Video links: ...` lines (literally `(none)` when empty, so the LLM sees an explicit absence rather than a missing field it might hallucinate around), and an optional `Source: ...` line.
- Blocks are joined with `\n\n---\n\n` into one string passed as the `context` argument to `generate_chat_answer`; the parallel `references` list of dicts is converted into `ReferenceItem` Pydantic models and returned verbatim in `ChatResponse.references`.
- The system prompt (`CHAT_SYSTEM` in `llm_service.py`) explicitly instructs: cite the bracket number after any claim, surface audio/video links as markdown links when relevant, and never invent URLs or facts beyond the sources — the numbering scheme is what makes that instruction actionable rather than aspirational.

**Example**
```python
# File: app/services/chat_context.py
def ordered_parent_slots(nodes):
    """First occurrence wins. Duplicate parent_ids share one slot."""
    seen, slots = set(), []
    for nws in nodes:
        pid = str(nws.node.metadata.get("parent_id") or "") or None
        key = pid if pid else f"__child__:{nws.node.node_id}"
        if key in seen:
            continue
        seen.add(key)
        slots.append((pid, nws))
    return slots

# lines = [f"[{idx}] Title: {title}", f"URL: {url}", "Text:", text,
#          "Audio links: " + "; ".join(audio_links) or "Audio links: (none)", ...]
```

**Where in this repo**
`app/services/chat_context.py::ordered_parent_slots` and `::build_numbered_chat_context`; consumed in `app/routers/chat.py::chat`; system-prompt citation instructions in `app/services/llm_service.py::CHAT_SYSTEM`; output shape in `app/models/schemas.py::ReferenceItem`.

**Interview angle**
Q: Why dedup by parent document instead of just numbering every retrieved chunk 1-through-k?
A: Because the reranker commonly surfaces multiple chunks from the same source article when it's highly relevant, and numbering each of those chunks separately would produce citations like `[2]` and `[5]` that point to the same underlying document under two different numbers — confusing for the user and actively harmful for faithfulness evaluation, which needs a clean claim-to-source mapping. Deduping to one slot per parent keeps the citation numbers meaningful: each number is one checkable source, not one arbitrary chunk boundary.

## Fail-Fast Error Handling via HTTPException

**What it is**
Rather than returning `200` with an ambiguous or empty body when something goes wrong internally, every failure mode in the request path is mapped to a specific HTTP status code the moment it's detected, using FastAPI's `HTTPException`. A malformed session id, an empty retrieval result, a context-assembly failure, and an over-quota client each get their own distinct, semantically correct status — the caller doesn't have to sniff response content to figure out what happened.

**How it works**
- `429 Too Many Requests` — raised by the rate limiter dependency (`chat_rate_limit`/`retriever_rate_limit`) *before* the handler body runs at all, with a `Retry-After` header telling the client exactly how long to back off.
- `422 Unprocessable Entity` — raised in `chat()` when a client-supplied `session_id` isn't a parseable UUID; this is a request-shape problem, not a server failure, so it maps to the same status FastAPI already uses for Pydantic validation errors.
- `404 Not Found` — raised when `pipeline.retrieve(...)` comes back with zero nodes for the query; there's genuinely nothing to answer with, so this is treated as "resource not found" rather than masked as a low-confidence `200`.
- `500 Internal Server Error` — raised if `build_numbered_chat_context` produces an empty context string despite having nodes (a data/metadata problem, not a client problem) — distinguishing this from the `404` case is deliberate: one is "no results," the other is "we got results but couldn't turn them into something usable."
- Because these are raised as exceptions rather than returned as values, they short-circuit the rest of the pipeline automatically — a `404` on retrieval means the code never reaches PII masking, generation, or persistence, so no wasted LLM cost is spent on a request that's already known to be answerless.

**Example**
```python
# File: app/routers/chat.py
nodes = pipeline.retrieve(query=safe_query, top_k=request.top_k)
if not nodes:
    raise HTTPException(status_code=404, detail="No retrieval results for this query.")
...
context, ref_dicts = build_numbered_chat_context(nodes, parent_payloads)
if not context:
    raise HTTPException(status_code=500, detail="Could not build context from retrieval results.")
```
```python
# File: app/guardrails/rate_limiter.py
if count > limit:
    raise HTTPException(
        status_code=429,
        detail=f"Rate limit exceeded: max {limit} requests per {window}s.",
        headers={"Retry-After": str(window)},
    )
```

**Where in this repo**
`app/routers/chat.py::chat` (422/404/500); `app/guardrails/rate_limiter.py::_check` (429).

**Interview angle**
Q: Why bother distinguishing 404 from 500 here — couldn't both just be "something went wrong, try again"?
A: They imply different remediation. A `404` ("no retrieval results") tells the caller the query itself may need rephrasing or the corpus genuinely has no coverage — retrying the identical request won't help. A `500` after nodes *were* found but context assembly still failed signals an internal bug (bad metadata, a broken payload fetch) that retrying might actually recover from if it was transient, and that should page an engineer if it keeps happening. Collapsing both into one generic error code would throw away exactly the signal a caller or an on-call engineer needs to decide what to do next.

---
Related: [06-fastapi-concepts.md](06-fastapi-concepts.md) for general FastAPI mechanics (lifespan, `Depends`, background tasks, ASGI) independent of this repo's specific usage.
