# Data Pipeline & Infra Concepts

How raw articles get from a CSV/URL into two searchable stores, how that indexing can run either as a one-shot batch or as a continuously-polled stream, and the supporting infra (Postgres chat memory, typed config) that keeps the FastAPI app itself boring and predictable.

## Batch Indexing DAG

**What it is**

`dags/dags.py` is the "backfill" path: read a CSV export of article URLs/titles, scrape each page, build a validated document, and push it to both search backends. It's written as a single Airflow DAG using dynamic task mapping (`.expand()`) so that one dynamic set of rows fans out into N parallel task instances instead of a Python `for` loop. Practically important: **the entire file is currently wrapped in a top-level triple-quoted string** (`"""` on line 1, closing `"""` on line 87) — every line, including `dag = indexing_pipeline()`, is dead code sitting inside a module docstring. Airflow will import the module but never register a live DAG from it. That's either a deliberate "disable this DAG without deleting it" move or a leftover from debugging; either way it means the streaming pipeline (see below) is the one actually running.

**How it works**

- `read_csv` task reads a fixed local path, filters rows by index (`>= 1490`), hard-caps to the first 10 for testing, and stashes the row list as JSON at `/tmp/ccp_rows.json` — then returns just the list of *indices* `[0..N)`.
- `extract_page.expand(row_index=row_indices)` maps one task instance per index; each instance re-reads the same `/tmp` JSON and scrapes its one row via `extract_page_content`.
- `create_document.expand(raw=raw_docs)` converts each scraped dict into a validated `LectureDocument` (via `AlgoliaIndexManager.create_document`) and passes it downstream as a plain dict (`model_dump(by_alias=True)`) — Airflow's XCom serialization can't carry Pydantic objects directly.
- `push_algolia.expand(...)` and `index_qdrant.expand(...)` both depend on `doc_dicts` but not on each other — they run as independent parallel branches, not a strict "Algolia then Qdrant" chain.
- No dedup/state tracking across runs: re-running the DAG re-scrapes and re-pushes everything the input rows point to.

**Example**

```python
# File: dags/dags.py (entire module body is inside a docstring — inert)
@task
def read_csv() -> list[int]:
    path = "/mnt/c/Users/ARL/Videos/CCP/Posts-Export-2026-January-17-0453.csv"
    data = pd.read_csv(path)
    rows = [
        {"url": row["Permalink"], "title": row["Title"]}
        for index, row in data.iterrows()
        if index >= 1490
    ]
    limited_rows = rows[:10]
    with open("/tmp/ccp_rows.json", "w") as f:
        json.dump(limited_rows, f)
    return list(range(len(limited_rows)))
```

**Where in this repo**

`dags/dags.py` — `indexing_pipeline()` DAG (`read_csv` → `extract_page` → `create_document` → `push_algolia` / `index_qdrant`).

**Interview angle**

Q: Walk me through the batch indexing DAG and how it differs from the streaming one.
A: It's a straight linear pipeline with Airflow dynamic task mapping fanning a CSV of rows out into parallel per-row tasks for scrape → build-document → dual-push. It's simple and deterministic, good for one-off backfills, but has no cursor/offset tracking — re-running it reprocesses everything unless you slice the input file yourself. Worth flagging live: as currently committed, the whole file is nested inside a triple-quoted string, so it isn't actually a registered DAG right now — a good example of "dead code that looks live" to catch in review.

## Streaming Indexing via Redis Streams (Producer / Consumer)

**What it is**

Instead of one long batch job, indexing is split into two independently-scheduled Airflow DAGs that talk to each other only through a Redis Stream. `dags/producer_dags.py` is a manually-triggered DAG that reads new CSV rows and `XADD`s them onto a stream. `dags/consumer_dags.py` is a DAG on a 2-minute cron schedule that reads from that stream via a **consumer group** (`XREADGROUP`), scrapes and indexes each message, and only `XACK`s it after a fully successful dual-push. This decouples "what needs indexing" from "index it," and gives at-least-once delivery semantics for free from Redis itself rather than hand-rolled retry logic.

**How it works**

- Producer: `r.xadd(RAW_STREAM, {"url": ..., "title": ...})` for each of the last 20 rows in the CSV — a pure append, no consumer-group involvement at all.
- Consumer `fetch_batch`: calls `xgroup_create(name=RAW_STREAM, groupname="airflow_group", id="0", mkstream=True)` once (wrapped in `try/except ResponseError` since Redis errors if the group already exists — used here as a poor-man's "create if not exists").
- It then reads with `id="0"` first — this asks for **this consumer's own pending, previously-delivered-but-unacked** messages (i.e., recovery from a crashed prior run) — and only if that returns nothing does it fall through to `id=">"`, which means "new messages never delivered to anyone in this group," blocking up to 2 seconds (`block=2000`) for up to 5 messages (`count=5`).
- `extract_document`: if scraping throws, the task **XACKs immediately** and returns `None` — a permanently bad URL is dropped from the stream rather than retried forever.
- `index_document` is decorated `@task(trigger_rule="all_done", max_active_tis_per_dag=1, retries=0)`. `trigger_rule="all_done"` means it still runs even though some sibling mapped instances upstream may have failed/been skipped (so one bad row in the batch doesn't stall the rest); `retries=0` combined with **not** acking on exception means a genuine indexing failure leaves the message pending in the consumer group — the *next* DAG run's `fetch_batch` will pick it back up via the `id="0"` pending-read path.
- `XACK` happens only after **both** `algolia.push_document` and `qdrant.index_document` succeed — so the ack is the actual "fully processed" signal, not "task ran."

**Example**

```python
# File: dags/consumer_dags.py — pending-first, then new
messages = r.xreadgroup(
    groupname=group, consumername=consumer,
    streams={RAW_STREAM: "0"}, count=5,
)
if not messages or not messages[0][1]:
    messages = r.xreadgroup(
        groupname=group, consumername=consumer,
        streams={RAW_STREAM: ">"}, count=5, block=2000,
    )
...
@task(trigger_rule="all_done", max_active_tis_per_dag=1, retries=0)
def index_document(doc):
    if doc is None:
        return
    algolia.push_document(lecture_doc)
    qdrant.index_document(lecture_doc)
    get_redis().xack(RAW_STREAM, "airflow_group", doc["redis_id"])  # ACK only after success
```

> Security note: both `dags/producer_dags.py` and `dags/consumer_dags.py` hardcode a live Upstash Redis connection string with embedded credentials — `REDIS_URL = "rediss://<redacted>"` — directly in source. That's a real secret-in-source-control smell worth flagging in any review of this codebase; it should be pulled from `app/core/configs.py`'s `Settings.redis_url` instead, the way the rest of the app does it.

**Where in this repo**

`dags/producer_dags.py` (`pipeline()` → `publish_to_redis`), `dags/consumer_dags.py` (`pipeline()` → `fetch_batch` → `extract_document` → `index_document`).

**Interview angle**

Q: How do you get at-least-once (not at-most-once, not exactly-once) delivery out of Redis Streams here, and what's the failure mode if a crash happens mid-index?
A: Consumer groups track per-consumer delivery via `XREADGROUP`, and a message only leaves the "pending" state on explicit `XACK`. Acking happens as the *last* line of `index_document`, after both sinks succeed — so a crash between "delivered" and "acked" leaves the message pending, and the next run's pending-read (`id="0"`) redelivers it instead of losing it. The tradeoff is duplicate processing on retry: because `retries=0` and there's no idempotency key checked before reprocessing, a message that fails after Algolia succeeds but before Qdrant does will re-push to Algolia (harmless, since `save_object` upserts by `objectID`) and redo the whole scrape+embed+index cycle.

## Redis Streams as a Durable Work Queue

**What it is**

Beyond the producer/consumer mechanics, it's worth calling out *why* Redis Streams specifically: the design goal is decoupling "discovering new content" from "processing content" so each half can run on its own schedule, scale independently, and fail without taking the other down. The producer DAG is triggered manually/on-demand when new URLs need to be queued; the consumer DAG runs on a fixed cron regardless of whether the producer ever ran that day. The stream itself is the durable buffer between them — it survives both DAGs' runs, Airflow scheduler restarts, and (per Upstash) is backed by persistent storage rather than an in-memory-only cache.

**How it works**

- `RAW_STREAM = "raw_pages"` is the append-only log; each `XADD` gets an auto-generated, strictly increasing entry ID.
- The `airflow_group` consumer group maintains its own read cursor and pending-entries list (PEL) *per group*, independent of the raw stream — multiple groups could read the same stream differently, though this repo only uses one.
- Because it's a log rather than a plain list/pub-sub channel, messages aren't lost if no consumer is currently running — they simply accumulate until the next scheduled consumer run drains them (bounded by `count=5` per run, so a large backlog drains over several 2-minute ticks, not all at once).
- No dead-letter handling beyond the drop-on-extract-failure path in `extract_document` — there's no `XCLAIM`/`XAUTOCLAIM` logic to reassign entries stuck pending under a dead consumer name, which matters less here since `consumer` is a fixed string (`"airflow_consumer"`) reused by every run rather than one per worker.

**Example**

```python
# File: dags/consumer_dags.py
def get_redis():
    return redis.Redis.from_url(
        REDIS_URL,           # rediss://<redacted>  (see security note above)
        decode_responses=True,
        socket_connect_timeout=10,
        socket_timeout=10,
    )

RAW_STREAM = "raw_pages"
```

**Where in this repo**

`dags/producer_dags.py` / `dags/consumer_dags.py` — both operate on the same `RAW_STREAM = "raw_pages"` key on the same Upstash Redis instance.

**Interview angle**

Q: Why Redis Streams here instead of, say, SQS or Kafka?
A: It's the same producer/consumer decoupling pattern those systems provide — durable, ordered, ack-based delivery — just implemented on infrastructure the project already has cheap access to (a small hosted Redis instance) instead of standing up a dedicated broker. For this workload's volume (tens of documents at a time, one 2-minute-cron consumer), Streams' consumer groups are more than sufficient; the tradeoff is you inherit Redis's weaker durability/ops story (no built-in dead-letter queue, no partition-based horizontal scaling) rather than SQS/Kafka's, in exchange for simplicity.

## Dual Qdrant Collections (Parent / Child)

**What it is**

Every document is chunked hierarchically (1024-token parents, 256-token child leaves — see `concepts/01-retrieval-and-rag.md`), and instead of storing parent and child nodes in the same collection distinguished by a payload flag, `QdrantClintManager` maintains **two physically separate collections**: `<QDRANT_COLLECTION_NAME>_parents` and `<QDRANT_COLLECTION_NAME>_children`. Both are created with an identical vector schema, but they're queried very differently: children are what hybrid search runs against; parents are only ever fetched by ID once search has picked which children matched.

**How it works**

- `ensure_collections()` checks `client.collection_exists(...)` for each of `PARENT_COLLECTION` and `CHILD_COLLECTION` independently and creates whichever is missing — idempotent to call on every indexing run.
- `_create_chunk_collection` gives both collections the same 4 named vectors: `title_dense`/`text_dense` (1536-dim, cosine) and `title_sparse`/`text_sparse` (SPLADE sparse vectors) — so a parent point and a child point are structurally interchangeable, only the collection they live in differs.
- `push_child_nodes` additionally stores `parent_id` in the payload, resolved via `node.relationships.get(_NODE_PARENT)` where `_NODE_PARENT = "4"` — a hardcoded string mirroring llama_index's internal `NodeRelationship.PARENT` enum value. That's a fragile magic constant: it works because of how llama_index serializes relationship keys today, but it isn't type-checked against the enum, so a llama_index upgrade that changes that value would silently break parent linkage.
- At query time (`search()`), only `CHILD_COLLECTION` is searched; `retrieve_parents_by_ids()` does a separate `client.retrieve()` by ID against `PARENT_COLLECTION`, batched 64 IDs at a time, and logs a warning for any parent ID that comes back missing.

**Example**

```python
# File: indexing_pipeline/index_qdrant.py
QDRANT_BASE_COLLECTION = settings.qdrant_collection_name
PARENT_COLLECTION = QDRANT_BASE_COLLECTION + "_parents"
CHILD_COLLECTION = QDRANT_BASE_COLLECTION + "_children"

def ensure_collections(self):
    if not self.client.collection_exists(self.PARENT_COLLECTION):
        self._create_chunk_collection(self.PARENT_COLLECTION)
    if not self.client.collection_exists(self.CHILD_COLLECTION):
        self._create_chunk_collection(self.CHILD_COLLECTION)
```

**Where in this repo**

`indexing_pipeline/index_qdrant.py` — `QdrantClintManager.PARENT_COLLECTION` / `CHILD_COLLECTION`, `ensure_collections`, `push_parent_nodes`, `push_child_nodes`, `retrieve_parents_by_ids`.

**Interview angle**

Q: Why two separate collections instead of one collection with a `node_type` payload field?
A: Physical separation keeps the collection that search actually hits (children) small, since it never has to filter out parent-type points at query time — every point in `CHILD_COLLECTION` is a valid hybrid-search candidate by construction, with no wasted index space on parent vectors that would never be searched directly. It's a deliberate space/speed tradeoff, at the cost of running two `ensure_collections`/upsert code paths and having to keep both schemas in sync by hand — a single collection with a filtered payload field would be less duplication but would need a filter on every child search.

## Dual Indexing Sinks (Algolia + Qdrant)

**What it is**

Every successfully processed document is written to *two* independent search backends: Algolia (hosted keyword/instant search) and Qdrant (self-managed dense+sparse vector search). Neither is a fallback for the other — they serve different retrieval needs the app exposes separately (see `concepts/01-retrieval-and-rag.md` and `concepts/12-vector-databases-and-search.md`), and there is no shared transaction between the two writes.

**How it works**

- In the batch DAG (`dags/dags.py`), `push_algolia.expand(...)` and `index_qdrant.expand(...)` both branch off the same `doc_dicts` XCom independently — they run in parallel with no ordering guarantee between them.
- In the streaming consumer DAG, the two pushes happen **sequentially in the same task**: `algolia.push_document(lecture_doc)` then `qdrant.index_document(lecture_doc)`, and the Redis `XACK` only fires after both return without raising.
- There's no two-phase commit or compensating rollback: if Qdrant indexing throws after Algolia already succeeded, the message stays unacked and gets reprocessed — which re-pushes to Algolia too. That's safe *only* because `AlgoliaIndexManager.push_document` calls `save_object`, which is an upsert keyed by `objectID`, so redundant re-pushes overwrite rather than duplicate.
- Both sinks are constructed from the same validated `LectureDocument` (see the "Idempotent document identity" section below) so the two stores never see divergent field sets for the same article.

**Example**

```python
# File: dags/consumer_dags.py — sequential dual push inside one task
algolia = AlgoliaIndexManager()
lecture_doc = algolia.create_document(doc)

qdrant = QdrantClintManager()
qdrant.ensure_collections()

algolia.push_document(lecture_doc)
qdrant.index_document(lecture_doc)

get_redis().xack(RAW_STREAM, "airflow_group", doc["redis_id"])
```

**Where in this repo**

`indexing_pipeline/index_algolia.py` (`AlgoliaIndexManager`), `indexing_pipeline/index_qdrant.py` (`QdrantClintManager`) — invoked together from both `dags/dags.py` and `dags/consumer_dags.py`.

**Interview angle**

Q: What happens if the Algolia push succeeds but the Qdrant push fails?
A: In the streaming path, the ack never fires, so the whole message is retried — including re-pushing to Algolia. That's fine in practice because Algolia's `save_object` is an upsert on `objectID`, so the retry is idempotent there; Qdrant, however, generates fresh node IDs on every chunking pass (see the idempotency section below), so a retried Qdrant write isn't a clean overwrite — it can leave duplicate points from the earlier partial attempt. The system tolerates *some* inconsistency between the two sinks by design; it doesn't guarantee they're ever perfectly in sync.

## Batched Embedding + Upsert

**What it is**

Embedding calls (both OpenAI dense and SPLADE sparse) and Qdrant writes are deliberately done in batches rather than one-node-at-a-time, but the *upsert* batch size (`UPSERT_BATCH_SIZE = 2`) is strikingly small compared to the embedding batch size (which is effectively "all nodes in one document's chunk set" in a single call). This is a real tension in the code: embeddings are computed efficiently in bulk, but the writes back to Qdrant are throttled hard.

**How it works**

- `get_dense_batch`/`get_sparse_batch` take a list of texts and make exactly one OpenAI/`fastembed` call each, returning embeddings in input order (`get_dense_batch` explicitly re-sorts OpenAI's response by `.index` since batch responses aren't guaranteed to return in request order).
- Both batch methods guard against empty strings (`t if t.strip() else " "`) — OpenAI's embeddings endpoint errors on an empty input string, so any chunk with blank text/title gets replaced with a single space rather than crashing the whole batch.
- `push_parent_nodes`/`push_child_nodes` build the full list of `PointStruct`s for a document in memory first, then hand the whole list to `_upsert_batched`, which slices it into chunks of `UPSERT_BATCH_SIZE = 2` and issues one `client.upsert()` call per slice.
- A batch size of 2 trades throughput for reliability/predictability — smaller request bodies are less likely to hit payload-size or timeout limits on a hosted Qdrant cluster, at the cost of many more round trips per document (a document with 40 child nodes means 20 separate upsert calls).

**Example**

```python
# File: indexing_pipeline/index_qdrant.py
UPSERT_BATCH_SIZE = 2

def _upsert_batched(self, collection_name: str, points: list):
    for i in range(0, len(points), self.UPSERT_BATCH_SIZE):
        self.client.upsert(
            collection_name=collection_name,
            points=points[i : i + self.UPSERT_BATCH_SIZE],
        )

def get_dense_batch(self, texts: list[str]) -> list[list]:
    safe = [t if t.strip() else " " for t in texts]
    response = self.openai_client.embeddings.create(
        model="text-embedding-3-small", input=safe
    )
    return [e.embedding for e in sorted(response.data, key=lambda x: x.index)]
```

**Where in this repo**

`indexing_pipeline/index_qdrant.py` — `get_dense_batch`, `get_sparse_batch`, `_upsert_batched`, `push_parent_nodes`, `push_child_nodes`.

**Interview angle**

Q: Why batch the embedding calls but keep the Qdrant upsert batch size so small?
A: They're solving different problems. Batching the embedding calls amortizes network/API overhead across many texts in one request, which OpenAI and `fastembed` both support natively. The tiny upsert batch (2 points) is a separate, more conservative choice about write reliability against the vector DB — smaller writes fail less often and are cheaper to retry individually. In a higher-throughput setting you'd want to raise `UPSERT_BATCH_SIZE` and measure against Qdrant's actual payload limits rather than leaving it at a value that looks tuned for local testing.

## Postgres-Backed Chat Memory

**What it is**

Multi-turn conversation history is persisted in a single Postgres table, `chat_messages`, behind a connection pool that's opened once at FastAPI startup and shared across all requests. Critically, the whole feature is optional by construction: if `DATABASE_URL` isn't configured, the pool is simply never created, and every caller degrades gracefully to "no history" rather than throwing.

**How it works**

- `postgres/client.py::init_pool()` checks `settings.database_url` first; if it's `None`, it logs and returns without creating anything — `_pool` stays `None`.
- If a URL is set, it lazily imports `psycopg_pool.ConnectionPool` (`min_size=1, max_size=10, open=True`) and stores it in a module-level global, so `get_pool()` returns the same pool object everywhere in the process.
- `ensure_schema()` reads `postgres/schema.sql` as raw text, splits it on `;`, and executes each non-empty statement — a naive split that works fine for this file's two simple statements but would break on any statement containing a semicolon inside a string literal or function body.
- Both `init_pool()` and `ensure_schema()` are called from `main.py`'s `lifespan` context manager on startup, and `close_pool()` on shutdown — standard FastAPI resource lifecycle management.
- `ChatMemoryStore.fetch_recent_messages` queries `ORDER BY created_at DESC, id DESC LIMIT %s` (to cheaply grab the most recent N rows) and then reverses the Python list before returning, so callers get messages oldest-first even though the SQL fetched newest-first.
- `append_turn` inserts both the user and assistant message in a single `INSERT ... VALUES (...), (...)` statement — one round trip, one implicit transaction, for the whole turn.

**Example**

```python
# File: postgres/client.py
def init_pool() -> None:
    global _pool
    settings = get_settings()
    if not settings.database_url:
        logger.info("DATABASE_URL not set; chat memory disabled.")
        return
    if _pool is not None:
        return
    from psycopg_pool import ConnectionPool
    _pool = ConnectionPool(
        conninfo=settings.database_url, min_size=1, max_size=10, open=True,
    )
```

```sql
-- File: postgres/schema.sql
CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    role VARCHAR(16) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

**Where in this repo**

`postgres/client.py` (`init_pool`, `close_pool`, `get_pool`, `ensure_schema`), `postgres/schema.sql`, `app/services/chat_memory.py` (`ChatMemoryStore`), wired in `main.py::lifespan`.

**Interview angle**

Q: What happens to chat history if Postgres is misconfigured or unreachable in a given deployment?
A: The app never crashes because of it — `get_pool()` returns `None`, and both `fetch_recent_messages` and `append_turn` check for that and return an empty list / no-op respectively. It's an explicit design choice to make persistence a pure enhancement rather than a hard dependency, which matters for a project where the core RAG chat flow shouldn't be blocked by an optional feature. The tradeoff is silent degradation — nothing surfaces to the caller that history was skipped, so it's worth knowing where to look (`DATABASE_URL` in `.env`) if history unexpectedly isn't showing up.

## Typed, Env-Driven Configuration

**What it is**

`app/core/configs.py` defines a single `pydantic-settings` `Settings` class that loads and validates every credential and feature flag the app needs from `.env`, then exposes it as a process-wide singleton via `functools.lru_cache`. This replaces scattered `os.getenv()` calls with one typed, IDE-autocompletable object, and lets required-vs-optional integrations be expressed directly in the type system (`str` vs `Optional[str]`).

**How it works**

- `model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="allow")` — `case_sensitive=True` means the alias must match the `.env` key's case exactly, and `extra="allow"` means any *other* env vars present are silently accepted rather than raising, which also means a typo'd env var name for one of these settings won't fail loudly — it just falls back to the field's own default/required behavior while the typo'd variable sits unused.
- Required fields (no default) — `openai_api_key`, `qdrant_url`, `qdrant_api_key`, `algolia_*`, etc. — make `Settings()` raise a validation error at import time if they're missing, which is a fast, fail-at-startup check rather than a runtime `KeyError` deep in a request.
- Optional integrations are modeled as `Optional[...] = Field(default=None, ...)`: `database_url`, `langfuse_public_key`/`langfuse_secret_key`, and `redis_url`. Every consumer of these (`postgres/client.py`, the Langfuse tracer, the rate limiter) checks for `None` before wiring the feature in — the type signature *is* the contract for "this integration is optional."
- Naming is slightly inconsistent by convention: most aliases mirror `SCREAMING_SNAKE_CASE` env vars exactly (`OPENAI_API_KEY`), but the Azure Speech fields alias lowercase names (`Field(alias="speech_key")`, `Field(alias="region")``) — a real inconsistency worth noticing if you're hunting for why an env var "isn't picking up."
- `get_settings()` is decorated `@lru_cache`, so `Settings()` is constructed exactly once per process; every call site gets the same cached instance rather than re-parsing `.env` repeatedly.

**Example**

```python
# File: app/core/configs.py
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        case_sensitive=True, extra="allow",
    )

    openai_api_key: str = Field(alias="OPENAI_API_KEY")          # required
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")   # optional
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")         # optional

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

**Where in this repo**

`app/core/configs.py` (`Settings`, `get_settings`) — imported by `indexing_pipeline/index_qdrant.py`, `postgres/client.py`, and most of `app/`.

**Interview angle**

Q: How does this app tell the difference between "this integration is disabled" and "this is misconfigured"?
A: Through the type system — required settings have no default, so a missing required var fails fast at `Settings()` construction (effectively at process boot); optional settings default to `None` and are meant to be checked explicitly by every consumer before use. If you see a file doing `if settings.database_url:` or `if settings.redis_url:`, that's the "optional" contract in action; anything without a default is assumed always present, which is why the app crashes on boot rather than mid-request if, say, `QDRANT_API_KEY` is missing from `.env`.

## Idempotent Document Identity — and Its Limits

**What it is**

This concept isn't in a single named class, but it's a real, load-bearing pattern worth calling out on its own: `AlgoliaIndexManager.create_document` derives a document's ID deterministically — `hashlib.md5(data["url"].encode()).hexdigest()` — rather than generating a random UUID. That makes re-indexing the *same URL* an overwrite in Algolia, not a duplicate. But that guarantee stops at the Algolia boundary: Qdrant point IDs come from `node.node_id`, which llama_index's `HierarchicalNodeParser` assigns freshly (effectively random) on every chunking pass, so reprocessing the same article produces an entirely new set of Qdrant point IDs rather than overwriting the old ones.

**How it works**

- `doc_id = hashlib.md5(data["url"].encode()).hexdigest()` becomes both the Algolia `objectID` and the `LectureDocument.id` field (aliased `objectID`) — same URL in, same ID out, every time.
- `AlgoliaIndexManager.push_document` calls `write_client.save_object(index_name=..., body=record)` — Algolia's `save_object` is an upsert keyed by `objectID`, so a second push for the same URL replaces the first record cleanly.
- `HierarchicalChunker.process_document` → `HierarchicalNodeParser.get_nodes_from_documents(...)` assigns each parent/child node a fresh `node_id_` internally; nothing in `chunking.py` or `index_qdrant.py` seeds that ID from the document's URL/hash.
- Net effect: reprocessing a document (batch DAG re-run, or a message redelivered after a partial-failure retry in the streaming consumer) is safe and idempotent for Algolia, but leaves the *old* Qdrant points for that document orphaned alongside newly-inserted duplicates — there is no delete-by-`url`-payload-filter step anywhere in the indexing code to clean those up first.

**Example**

```python
# File: indexing_pipeline/index_algolia.py
doc_id = hashlib.md5(data["url"].encode()).hexdigest()
document = LectureDocument(
    objectID=doc_id,
    title=data.get("title", "Untitled"),
    transcript=data.get("transcript", ""),
    url=data.get("url"),
    ...
)
```

**Where in this repo**

`indexing_pipeline/index_algolia.py::AlgoliaIndexManager.create_document` (deterministic ID), `indexing_pipeline/chunking.py::HierarchicalChunker` and `indexing_pipeline/index_qdrant.py::push_parent_nodes`/`push_child_nodes` (non-deterministic Qdrant point IDs).

**Interview angle**

Q: Is it safe to just re-run the indexing pipeline on documents that have already been indexed?
A: Only halfway. Algolia is safe to re-run against — the MD5-of-URL `objectID` makes every push an upsert. Qdrant is not: because point IDs come from llama_index's node parser rather than being derived from the document URL, re-indexing the same article creates a fresh, disconnected set of parent/child points without removing the earlier ones, so a naive re-run policy would slowly accumulate duplicate/stale vectors in both Qdrant collections. Fixing that properly would mean either deriving node IDs deterministically from `(url, chunk_index)` or deleting existing points for a URL (via a payload filter) before re-inserting.
