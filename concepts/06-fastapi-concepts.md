# FastAPI Concepts

General FastAPI knowledge, not tied to this repo's specific usage (see [05-app-and-api-design.md](05-app-and-api-design.md) for how these apply here).

## ASGI vs WSGI

**What it is**

WSGI (Web Server Gateway Interface) is the traditional Python web-server interface used by Flask and classic Django: it is strictly synchronous — one thread handles one request from start to finish, and the whole call stack blocks until a response is ready. ASGI (Asynchronous Server Gateway Interface) is its successor, and it's what FastAPI is built on (via Starlette). ASGI supports async/await natively, plus WebSockets, Server-Sent Events, and other long-lived, bidirectional connections that a request/response-only protocol like WSGI can't express. An ASGI server (Uvicorn, Hypercorn, Daphne) runs an event loop that can juggle thousands of concurrent connections in a single process, switching between them whenever one is waiting on I/O rather than dedicating a thread to each. This matters enormously for LLM-backed APIs: a call to an LLM provider or a vector DB can take seconds, and during that wait an ASGI app can keep serving other requests instead of parking a thread.

**How it works**

- A WSGI app is a callable `app(environ, start_response)`; an ASGI app is a callable `app(scope, receive, send)` — three ASGI primitives replace WSGI's two, because ASGI needs to model streaming/bidirectional messages instead of one flat request/response.
- `scope` describes the connection (HTTP, WebSocket, or lifespan), `receive` awaits incoming messages (body chunks, WebSocket frames), `send` awaits outgoing ones.
- One OS thread can run one event loop that multiplexes many connections, as long as the code actually yields control at I/O points (via `await`).
- WSGI concurrency instead comes from spinning up more threads or worker processes (e.g. Gunicorn workers) — concurrency is achieved by parallelism, not cooperative multitasking.
- FastAPI itself is a thin layer of routing/validation/dependency-injection sugar on top of Starlette's ASGI implementation.

**Example**

```python
# A minimal raw ASGI app (no framework) — illustrates what FastAPI sits on top of.
async def app(scope, receive, send):
    assert scope["type"] == "http"
    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [(b"content-type", b"text/plain")],
    })
    await send({
        "type": "http.response.body",
        "body": b"Hello from raw ASGI",
    })

# Run with: uvicorn this_module:app
```

**Common pitfall**

Assuming ASGI automatically makes your code fast or non-blocking. ASGI only gives you the *option* to be concurrent — if your handlers call blocking synchronous code inside `async def` (e.g. a blocking DB driver, `time.sleep`), you lose the benefit entirely and can stall the single event loop for every user, which is strictly worse than WSGI's thread-per-request isolation.

**Interview angle**

Q: Why would you choose FastAPI over Flask for an LLM-backed service?
A: LLM calls, vector DB lookups, and most external API calls are I/O-bound — the CPU is idle while waiting on the network. ASGI lets a single process hold many such requests concurrently on one event loop, because each `await` yields control back to the loop instead of blocking a thread. Flask/WSGI would need a thread (or worker process) per concurrent in-flight request, which scales far worse under high-latency, high-concurrency I/O workloads like calling an LLM API.

## `async def` vs `def` endpoints

**What it is**

FastAPI lets you declare a path operation function as either `async def` or plain `def`, and it treats them very differently under the hood. An `async def` endpoint is awaited directly on the running event loop — it must not block, because blocking freezes every other concurrent request being served by that worker. A plain `def` endpoint is automatically dispatched to a separate threadpool (Starlette's `run_in_threadpool`, backed by `anyio`), so it can call blocking/synchronous code safely without stalling the event loop — at the cost of thread overhead and a bounded pool size. Choosing correctly is one of the most consequential decisions in a FastAPI codebase, because it's easy to write `async def` out of habit and then unknowingly introduce a blocking call inside it.

**How it works**

- FastAPI inspects the function signature; if it's a coroutine function, it's awaited in-loop, otherwise it's submitted to a threadpool executor.
- The default threadpool has a limited number of worker threads (historically 40 in Starlette), so a burst of blocking sync endpoints can exhaust it and start queuing requests.
- Dependencies follow the same rule independently — a route can be `async def` with a sync dependency (goes to threadpool) or vice versa; FastAPI resolves each appropriately.
- CPU-bound work (heavy computation) blocks *either way* — async doesn't parallelize CPU work, it only helps concurrency during I/O waits; genuinely CPU-heavy work belongs in a separate process (e.g. `ProcessPoolExecutor`, a worker queue).
- To call a blocking function from inside `async def` safely, wrap it: `await run_in_threadpool(blocking_fn, *args)` or use `asyncio.to_thread`.

**Example**

```python
from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool
import httpx
import requests  # sync/blocking client

app = FastAPI()

@app.get("/async-good")
async def async_good():
    async with httpx.AsyncClient() as client:
        r = await client.get("https://api.example.com/data")  # non-blocking
    return r.json()

@app.get("/async-bad")
async def async_bad():
    r = requests.get("https://api.example.com/data")  # BLOCKS the event loop!
    return r.json()

@app.get("/async-fixed")
async def async_fixed():
    r = await run_in_threadpool(requests.get, "https://api.example.com/data")
    return r.json()

@app.get("/sync-endpoint")
def sync_endpoint():
    r = requests.get("https://api.example.com/data")  # fine — runs in threadpool automatically
    return r.json()
```

**Common pitfall**

Calling a blocking library (a synchronous DB driver, `requests`, `time.sleep`, heavy `pandas`/CPU work) directly inside an `async def` handler. It doesn't raise an error — it just silently serializes every concurrent request on that worker behind the blocking call, which shows up in production as mysteriously terrible p99 latency under load, not as an obvious bug in dev with one user.

**Interview angle**

Q: If a route doesn't need to await anything, should it be `async def` or `def`?
A: Prefer plain `def` unless you specifically need to `await` inside it. `def` endpoints are dispatched to a threadpool automatically, so any incidental blocking work inside them doesn't touch the event loop. Marking something `async def` is a promise that nothing inside it blocks — if you can't guarantee that (e.g. because you depend on a sync library), `def` is the safer default.

## Pydantic request/response validation

**What it is**

FastAPI uses Pydantic models (type-annotated classes) to describe the *shape* of request bodies, query/path parameters, headers, and responses, and it validates and (de)serializes against those models automatically on every request. This turns your type hints into working runtime validation: if a client sends a malformed body — wrong type, missing required field, out-of-range value — FastAPI returns a structured `422 Unprocessable Entity` with a field-by-field error list, and your handler code never even runs. On the way out, declaring a `response_model` does double duty: it filters the object your function returns down to only the declared fields (so you can safely return an ORM object with extra internal fields and trust that only the public ones leak out), and it feeds the OpenAPI schema.

**How it works**

- Pydantic models are plain classes with type-annotated fields (`BaseModel` subclasses); FastAPI reads these annotations to build parsers and JSON Schema.
- Path/query params, headers, and cookies use `Path(...)`, `Query(...)`, `Header(...)`, `Cookie(...)` to add extra constraints (min/max length, regex, ranges) beyond the bare type.
- Validation failures raise `RequestValidationError` internally, which FastAPI's default exception handler turns into a `422` with a `detail` list of `{loc, msg, type}` entries.
- `response_model` runs the returned object back through a Pydantic model *before* serialization — extra attributes are dropped, missing ones raise a server-side error (a bug in your code, not the client's).
- Pydantic v2 (current) is a Rust-backed core (`pydantic-core`), which made validation dramatically faster than v1; field validators use `@field_validator` and model-level ones use `@model_validator`.
- `model_config = ConfigDict(...)` (v2) controls behaviors like extra-field handling (`extra="forbid"`), aliasing, and ORM-mode-equivalent (`from_attributes=True`).

**Example**

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

app = FastAPI()

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @field_validator("message")
    @classmethod
    def no_blank_message(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be blank")
        return v

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    # note: no `tokens_used` field here even if the handler object has one —
    # response_model filters it out of the outgoing JSON

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    # req is already validated: message is non-empty, temperature in [0, 2]
    return ChatResponse(reply=f"Echo: {req.message}", session_id=req.session_id or "new")
```

**Common pitfall**

Forgetting that `response_model` *filters* the output — returning a dict or ORM object with sensitive/internal fields (e.g. a hashed password, an internal cost/token count) and assuming it's hidden "because I didn't put it in the model" is correct, but the inverse mistake is common too: developers add a field to their internal object and expect it to show up in the API response, then are confused when it's silently dropped because they forgot to add it to the `response_model` class as well.

**Interview angle**

Q: How does declaring types on a FastAPI route give you validation "for free"?
A: The same Pydantic model you'd write anyway for editor autocompletion and type-checking is introspected by FastAPI at request time to build a parser: incoming JSON is parsed against it, and any mismatch (wrong type, missing field, failed constraint) short-circuits into a 422 before your handler body runs. So most "is this input well-formed" bugs never reach your business logic at all — you only have to handle the "well-formed but semantically wrong" cases yourself.

## Dependency Injection (`Depends`)

**What it is**

FastAPI's dependency injection system lets you factor out reusable pieces of request-handling logic — authentication, pagination parameters, a database session, rate limiting, a computed "current user" object — into standalone callables, then declare them as parameters on your route (or on other dependencies) using `Depends(...)`. Unlike DI frameworks in other ecosystems that need config files or container registration, FastAPI's DI is just plain function calls resolved automatically from type hints, and dependencies can themselves depend on other dependencies, forming a directed acyclic graph that FastAPI resolves per-request. This makes DI the primary way FastAPI encourages you to write testable, composable route logic instead of duplicating boilerplate across every handler.

**How it works**

- A dependency is any callable (function or class) that FastAPI can call and inject the return value of; it can itself accept `Depends(...)` parameters, request params, or use `yield` for setup/teardown (a "dependency with cleanup").
- Within a single request, a dependency used more than once (e.g. two routes' dependencies both need `get_db`) is by default only invoked once and its result cached/reused across that request's graph, unless you pass `use_cache=False`.
- Dependencies can be declared at the route level (`Depends(...)` in the signature), the router level (`APIRouter(dependencies=[...])`), or the app level (`FastAPI(dependencies=[...])`) — for logic that should apply to *every* route in that scope without each one explicitly asking for it.
- `yield`-based dependencies run the code before `yield` before the route, then run the code after `yield` after the response is generated (useful for closing a DB session, even if the route raised) — conceptually like a context manager.
- Class-based dependencies (a class with `__call__`) are useful when a dependency needs configuration (e.g. `RateLimiter(max_calls=5)`).

**Example**

```python
from fastapi import Depends, FastAPI, Header, HTTPException

app = FastAPI()

async def get_db():
    db = create_session()
    try:
        yield db  # handed to the route
    finally:
        db.close()  # always runs, even if the route raised

async def get_current_user(
    authorization: str = Header(...),
    db=Depends(get_db),          # dependency depending on another dependency
):
    user = db.get_user_by_token(authorization)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

class RoleChecker:
    def __init__(self, required_role: str):
        self.required_role = required_role

    def __call__(self, user=Depends(get_current_user)):
        if user.role != self.required_role:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user

require_admin = RoleChecker("admin")

@app.get("/admin/stats")
async def admin_stats(user=Depends(require_admin)):
    return {"ok": True, "user": user.id}
```

**Common pitfall**

Treating `Depends` purely as a way to avoid retyping code, and missing that it's also the seam you need for testability — if auth, DB access, or an external API client is buried as a hardcoded call inside the route body instead of a dependency, you can't swap it out in tests without monkeypatching internals. The fix is almost always "push it behind a `Depends`," which is also why DI and dependency overrides (below) are taught together.

**Interview angle**

Q: How is `Depends` different from middleware, and when would you pick one over the other?
A: `Depends` is per-route (or per-router/app if declared at that scope), fully typed, participates in FastAPI's validation/OpenAPI generation, and can be overridden per-test. Middleware runs globally on every request *before* routing and dependency resolution even happens, and only sees the raw ASGI request/response — it has no access to typed route parameters. Use middleware for truly cross-cutting concerns applied uniformly (CORS, request timing, global logging); use a dependency when the logic is specific to certain routes, needs typed access to request data, or needs to be swapped out in unit tests.

## Middleware

**What it is**

Middleware in FastAPI (inherited from Starlette) is code that wraps *every* request and response passing through the app — it runs before the router even decides which endpoint to call, and again on the way back out after the endpoint produces a response. It's the mechanism for concerns that genuinely apply uniformly across the whole API surface: CORS headers, request timing/logging, injecting a request ID, enforcing a global size limit, gzip compression. Because middleware operates at the ASGI layer, it doesn't know about your route-specific typed parameters or dependencies — it only sees the raw request/response objects, which is both its power (applies to literally everything, including 404s) and its limitation (can't do route-aware logic cleanly).

**How it works**

- Middleware is added via `app.add_middleware(MiddlewareClass, **options)` or with the lightweight `@app.middleware("http")` decorator for simple function-based cases.
- Middleware executes in a nested/onion order: the last-added middleware (with `add_middleware`) runs its "before" logic first (closest to the request) — order matters, especially for things like `GZipMiddleware` needing to wrap the actual response body.
- A function-based middleware receives `request` and a `call_next` coroutine; it must `await call_next(request)` to continue the chain and get the downstream response.
- Built-in Starlette middlewares include `CORSMiddleware`, `GZipMiddleware`, `TrustedHostMiddleware`, and `HTTPSRedirectMiddleware`.
- Because middleware runs outside the dependency-injection system, exceptions raised inside route handlers are (by default) already converted to responses by FastAPI's exception handling *before* they reach middleware on the way out — so middleware sees final responses, not raw exceptions, unless you specifically add exception-handling middleware.

**Example**

```python
import time
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    response = await call_next(request)  # runs the rest of the stack + the route

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-ms"] = f"{duration_ms:.1f}"
    return response
```

**Common pitfall**

Trying to read the request body inside middleware and then having the route handler fail to read it again — the ASGI request body is a stream that's consumed once; reading it in middleware without re-injecting it (or using `request.state` to cache the parsed result) breaks every downstream body-reading code path. This is why body-inspecting logic (e.g. logging payloads) usually belongs in a dependency instead, where FastAPI has already handled body consumption correctly.

**Interview angle**

Q: You need to add an `X-Request-ID` header to every response, including error responses and 404s. Middleware or dependency?
A: Middleware — a dependency only runs for routes that explicitly declare it (and typically not for 404s, since no route matched), whereas middleware wraps every request/response at the ASGI level regardless of whether routing succeeded. Anything that must apply unconditionally to the entire app, including paths that don't exist, belongs in middleware.

## Background tasks

**What it is**

`BackgroundTasks` is FastAPI's built-in mechanism for running a function *after* the HTTP response has already been sent back to the client, within the same request lifecycle. You inject a `BackgroundTasks` object as a route parameter, call `.add_task(func, *args, **kwargs)` on it, and FastAPI executes the queued function(s) once the response has been dispatched. This is ideal for cheap, best-effort, fire-and-forget work — sending a notification email, writing an audit log entry, warming a cache — where the client shouldn't have to wait for it, but it's explicitly not a task queue: it runs in the same process and event loop as the web server, with no persistence, retries, or crash recovery.

**How it works**

- `BackgroundTasks` is a Starlette primitive; FastAPI exposes it as an injectable dependency-like parameter type.
- Tasks are run sequentially after the response is sent, in the same worker process — if the function is `async def`, it's awaited on the event loop; if sync, it's run in the threadpool, following the same rule as endpoints.
- If the process crashes or restarts between "response sent" and "task executed," the task is lost — there's no persistence layer.
- Multiple calls to `add_task` on the same request queue up and run in order.
- Because it shares the worker's resources, a slow or resource-heavy background task can still degrade throughput for *other* concurrent requests on that worker even though the original caller already got their response.

**Example**

```python
from fastapi import BackgroundTasks, FastAPI

app = FastAPI()

def write_audit_log(user_id: str, action: str):
    with open("audit.log", "a") as f:
        f.write(f"{user_id} performed {action}\n")

def send_welcome_email(email: str):
    # pretend this calls an email provider's API
    print(f"Sending welcome email to {email}")

@app.post("/signup")
async def signup(email: str, background_tasks: BackgroundTasks):
    user_id = create_user(email)  # main work, client waits for this

    background_tasks.add_task(write_audit_log, user_id, "signup")
    background_tasks.add_task(send_welcome_email, email)

    return {"user_id": user_id}  # response sent to client now;
    # the two background tasks run after this, before the connection is torn down
```

**Common pitfall**

Reaching for `BackgroundTasks` for work that actually needs durability or retries — e.g. charging a credit card, sending a critical webhook, processing a large file. Since there's no persistence or retry logic, a worker restart, deploy, or unhandled exception inside the task silently drops the work with no automatic recovery. Anything where losing the task is unacceptable belongs in a real task queue (Celery, Arq, SQS + a worker, RQ) instead.

**Interview angle**

Q: What's the ceiling of `BackgroundTasks`, and when would you replace it with a task queue?
A: `BackgroundTasks` runs in-process with no persistence, no retries, and no visibility/monitoring — if the worker dies mid-task or the task raises, the work is simply lost with at most a log line. It's fine for cheap, non-critical, best-effort side effects. The moment you need "this must eventually happen even across restarts," retries with backoff, task-level observability, or heavy/long-running work that shouldn't compete with the web worker's resources, you need a dedicated task queue with its own broker and worker pool.

## Lifespan events

**What it is**

Lifespan events are the mechanism for running setup code once, before the app starts accepting any traffic, and teardown code once, after the app has stopped accepting traffic — the modern replacement for the deprecated `@app.on_event("startup")` / `@app.on_event("shutdown")` decorators. They're implemented as a single `@asynccontextmanager` function: everything before `yield` runs at startup, the value yielded (often a dict) is optionally exposed on `app.state`/`request.state`, and everything after `yield` runs at shutdown. This is the correct place for anything expensive and one-time — opening a DB connection pool, loading an ML model or embedding index into memory, pre-warming an LLM client — because it guarantees the work happens exactly once, before the first request, rather than lazily on some unlucky first caller (or worse, guarded by a `if not initialized` check scattered through request handlers).

**How it works**

- Declared as `async def lifespan(app: FastAPI) -> AsyncIterator[dict]` (or any type), decorated with `@asynccontextmanager`, and passed to `FastAPI(lifespan=lifespan)`.
- Code before `yield` runs once at process startup, blocking the server from accepting connections until it completes — so a slow lifespan means a slow cold start, but a *predictable* one, not a per-request tax.
- Code after `yield` runs once during graceful shutdown (e.g. on SIGTERM), even if the ASGI server is stopped — the right place to close DB pools, flush queues, or release GPU/model resources.
- Values yielded are commonly stored for later access via `request.state` inside route handlers, or attached directly to `app.state`.
- Exceptions raised before `yield` prevent the app from starting at all (fail fast on bad config/unreachable dependencies) — generally desirable over silently starting broken.

**Example**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    print("Loading embedding model and opening DB pool...")
    db_pool = await create_db_pool()
    embedding_model = load_embedding_model()  # expensive, one-time
    yield {"db_pool": db_pool, "embedding_model": embedding_model}
    # --- shutdown ---
    print("Closing DB pool...")
    await db_pool.close()

app = FastAPI(lifespan=lifespan)

@app.get("/search")
async def search(q: str, request: Request):
    model = request.state.embedding_model
    pool = request.state.db_pool
    vector = model.encode(q)
    return await pool.fetch_similar(vector)
```

**Common pitfall**

Loading an expensive resource (a model, a large index) lazily inside a request handler with an `if model is None: model = load_model()` guard, instead of in `lifespan`. This causes the *first* unlucky request to eat the entire load latency (and, without locking, a race where several concurrent early requests all trigger the load simultaneously), whereas lifespan front-loads that cost deterministically before any traffic is accepted.

**Interview angle**

Q: How do you avoid a slow "cold" first request in a FastAPI service that loads a large model?
A: Move the model load into the app's `lifespan` context manager, before the `yield`. The ASGI server won't start accepting connections until lifespan startup completes, so the cost is paid once, deterministically, at deploy/restart time rather than being smeared onto whichever request happens to arrive first — and it avoids race conditions from concurrent lazy-init checks in request handlers.

## `APIRouter` composition

**What it is**

`APIRouter` lets you split route definitions across multiple modules and then compose them into the main `FastAPI` app, instead of declaring every endpoint directly on a single monolithic `app` object. Each router behaves like a mini-app — you decorate functions with `@router.get(...)`, `@router.post(...)`, etc., optionally give the whole router a shared `prefix`, `tags` (for OpenAPI grouping), and `dependencies` (applied to every route in it) — and then wire it into the real app with `app.include_router(router)`. This is the standard way to keep a growing API's codebase organized: a `users` router, a `chat` router, an `admin` router, each ownable and testable somewhat independently, versus one file with hundreds of endpoints.

**How it works**

- `router = APIRouter(prefix="/chat", tags=["chat"])`; routes declared on it inherit that prefix, so `@router.post("/messages")` becomes `POST /chat/messages` once mounted.
- `app.include_router(router)` merges the router's routes into the app's route table; you can also pass additional `prefix`/`dependencies`/`tags` at include-time, layered on top of the router's own.
- Routers can themselves include other routers (nested composition), useful for versioned APIs (`/api/v1`, `/api/v2` each assembled from shared sub-routers).
- `dependencies=[Depends(...)]` at router-construction time (or at `include_router` time) applies that dependency to every route in the router without each one declaring it individually — good for "every route under `/admin` requires an admin user."
- Routers don't run their own separate app/lifespan — lifespan and middleware remain a single top-level concern on the main `FastAPI` instance.

**Example**

```python
# routers/chat.py
from fastapi import APIRouter, Depends
from ..dependencies import get_current_user

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/messages")
async def send_message(text: str, user=Depends(get_current_user)):
    return {"echo": text, "user": user.id}

@router.get("/history")
async def get_history(user=Depends(get_current_user)):
    return {"history": []}

# main.py
from fastapi import FastAPI
from .routers import chat, admin

app = FastAPI()
app.include_router(chat.router)
app.include_router(admin.router, prefix="/api", dependencies=[Depends(require_admin)])
```

**Common pitfall**

Declaring conflicting or overlapping `prefix` values between the router definition and `include_router`, or forgetting that `dependencies` passed at `include_router` time are *additional* to (not a replacement for) any declared on the router itself — leading to a dependency running twice, or assuming a route is protected when the protection was only added at one of the two layers and got missed.

**Interview angle**

Q: How would you structure a FastAPI app that's grown to 50+ endpoints across several domains (users, chat, billing)?
A: Split endpoints into one `APIRouter` per domain, each in its own module, each with a sensible `prefix` and `tags` for OpenAPI grouping, and router-level `dependencies` for cross-cutting auth within that domain. The main `main.py` becomes thin — it just constructs the `FastAPI` app, wires lifespan/middleware once, and calls `include_router` for each domain module. This keeps ownership boundaries clean and makes it easy to test or version one domain's routes independently.

## Dependency overrides for testing

**What it is**

`app.dependency_overrides` is a dict FastAPI consults before resolving any `Depends(...)`, letting you substitute a real dependency (a DB session, an external API client, an auth check) with a fake/stub version purely for tests, without touching the route code at all. You set `app.dependency_overrides[real_dependency] = fake_dependency` (keyed by the *function object*, not a string), and every route that would have used `real_dependency` uses `fake_dependency` instead for the duration of the override — then you clear it (usually via a pytest fixture with teardown) so it doesn't leak into other tests. Combined with `TestClient` (sync, wraps `httpx`) or `httpx.AsyncClient` (for genuinely async test flows), this is the standard, idiomatic way to unit- and integration-test FastAPI apps deterministically, without hitting a real database or a real LLM API.

**How it works**

- The override dict is keyed by the dependency callable itself: `app.dependency_overrides[get_db] = get_test_db`.
- It works because dependencies are DI-resolved at request time by identity, not hardcoded calls — which is exactly why pushing logic behind `Depends` (rather than calling it directly in the route body) is a testability decision, not just a DRY one.
- Overrides apply globally to the `app` instance for as long as they're set — good practice is to set them in a fixture and always clear (`app.dependency_overrides.clear()` or delete the specific key) in teardown to avoid cross-test pollution.
- Overriding a dependency that other dependencies depend on overrides it for the whole subgraph — e.g. overriding `get_db` also affects `get_current_user` if it depends on `get_db`.
- Works seamlessly with `pytest` fixtures scoped per test or per module, and with both `TestClient` (fully synchronous, good for most cases) and `httpx.AsyncClient(app=app, ...)` (needed if you must `await` things inside the test itself, e.g. testing a WebSocket or a truly async fixture).

**Example**

```python
# app.py
from fastapi import Depends, FastAPI

app = FastAPI()

async def get_db():
    return RealDatabaseConnection()

@app.get("/users/{user_id}")
async def get_user(user_id: str, db=Depends(get_db)):
    return db.fetch_user(user_id)

# test_app.py
import pytest
from fastapi.testclient import TestClient
from app import app, get_db

class FakeDB:
    def fetch_user(self, user_id):
        return {"id": user_id, "name": "Test User"}

@pytest.fixture
def client():
    app.dependency_overrides[get_db] = lambda: FakeDB()
    yield TestClient(app)
    app.dependency_overrides.clear()  # always clean up

def test_get_user(client):
    response = client.get("/users/42")
    assert response.status_code == 200
    assert response.json() == {"id": "42", "name": "Test User"}
```

**Common pitfall**

Forgetting to clear `app.dependency_overrides` after a test — since `app` is a module-level singleton shared across the whole test session, an override left in place from one test silently changes behavior in every subsequent test that touches the same dependency, producing confusing failures (or, worse, false passes) far away from the actual cause.

**Interview angle**

Q: How do you test a route that depends on a live external service (e.g. a vector DB or an LLM API) without actually calling it in CI?
A: Make sure the external call is behind a `Depends(...)` (not hardcoded in the route body), then in the test suite set `app.dependency_overrides[real_dep] = fake_dep` to swap in a stub that returns canned data, run requests through `TestClient`, and clear the override afterward. This gives deterministic, fast, offline tests of your route logic (validation, status codes, response shape) independent of the real service's availability or cost.

## OpenAPI / auto-generated docs

**What it is**

FastAPI derives a complete OpenAPI (formerly "Swagger") schema from the type hints and Pydantic models you already wrote for validation — no separate spec file to hand-maintain — and serves interactive documentation from it automatically at `/docs` (Swagger UI) and `/redoc` (ReDoc), plus the raw machine-readable schema at `/openapi.json`. Because the schema is generated directly from your code's types rather than authored separately, it structurally cannot drift out of sync with what the API actually accepts and returns — if you change a Pydantic field, the docs update on the next request. This is a genuinely distinctive part of FastAPI's design (the name literally comes from this): the same annotations serve editor autocompletion, runtime validation, *and* the API contract simultaneously.

**How it works**

- Every route's parameters, request body model, `response_model`, status codes, and docstring feed into the generated `openapi.json` (an OpenAPI 3.x document).
- `/docs` renders that schema via Swagger UI, letting you try requests directly from the browser (including auth flows if you've wired up FastAPI's security schemes); `/redoc` renders the same schema in a different, read-only style.
- You can enrich the generated docs without changing behavior: `summary`, `description`, and `response_description` on the route decorator, `Field(description=...)` on Pydantic fields, docstrings on the handler function, and `responses={...}` for documenting alternate status codes.
- `tags=["chat"]` (on a route or router) groups endpoints in the docs UI; `deprecated=True` marks a route as deprecated in the schema.
- The docs/schema endpoints can be disabled in production (`FastAPI(docs_url=None, redoc_url=None, openapi_url=None)`) if you don't want to expose your API shape publicly.
- Third-party tools (client SDK generators, API gateways, contract-testing tools) consume the raw `openapi.json` directly — it's a standard, not FastAPI-specific.

**Example**

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Chat API", version="1.0.0")

class ChatRequest(BaseModel):
    message: str = Field(description="The user's message to send to the model")

class ChatResponse(BaseModel):
    reply: str = Field(description="The model's generated reply")

@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a chat message",
    tags=["chat"],
    responses={429: {"description": "Rate limit exceeded"}},
)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Send a message to the chat model and get a reply.

    This docstring shows up as the endpoint's extended description in /docs.
    """
    return ChatResponse(reply=f"Echo: {req.message}")

# Visiting /docs shows an interactive UI built entirely from the above;
# /openapi.json returns the raw schema that generated it.
```

**Common pitfall**

Leaving `/docs`, `/redoc`, and `/openapi.json` enabled by default in a production deployment without considering that they expose your full API surface (every route, every field, every status code) to anyone who requests them — fine for an internal or partner API, but often something teams deliberately disable or put behind auth for a public-facing production service.

**Interview angle**

Q: How do API consumers know the exact shape of your endpoints, and how do you keep that documentation from going stale?
A: They read the auto-generated OpenAPI schema at `/openapi.json` (or browse it interactively at `/docs`), which FastAPI builds directly from the same Pydantic models and type hints used for request validation and response filtering. Because there's no separate hand-written spec to maintain, the documentation is structurally incapable of drifting from the actual code — changing a field's type or a route's status codes updates the docs on the next server start, automatically.

## Exception handlers

**What it is**

Exception handlers let you centralize how specific exception types are converted into HTTP responses, instead of scattering `try/except` blocks with manual `JSONResponse` construction across every route. FastAPI already does this internally for `HTTPException` (turns it into the given status code + JSON body) and for Pydantic validation errors (turns them into `422`), and `@app.exception_handler(SomeExceptionType)` lets you register the same behavior for your own domain exceptions — e.g. a custom `LLMProviderError` or `InsufficientCreditsError` — so raising that exception anywhere in your call stack (route, dependency, or deep in a service function) consistently produces the right status code and error shape without every caller needing to know about HTTP at all.

**How it works**

- `@app.exception_handler(ExcType)` registers an async function `(request: Request, exc: ExcType) -> Response` that FastAPI calls whenever `ExcType` (or a subclass) propagates uncaught out of a route/dependency.
- This decouples business/domain logic from HTTP concerns: a service function can `raise InsufficientCreditsError(user_id)` with no knowledge of status codes, and the registered handler is the single place that maps it to `402 Payment Required` with a specific JSON body.
- Handler resolution follows the exception's MRO — a handler registered for a base class also catches subclasses unless a more specific handler is also registered.
- Overriding the *built-in* handlers is done the same way: `@app.exception_handler(RequestValidationError)` lets you customize the shape of validation error responses app-wide (e.g. to match a different API error format).
- A catch-all `@app.exception_handler(Exception)` is possible for last-resort logging/sanitizing of unhandled errors (e.g. to avoid leaking stack traces to clients) but should re-raise or return a generic 500 — swallowing everything silently is a debugging trap.

**Example**

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

class InsufficientCreditsError(Exception):
    def __init__(self, user_id: str, needed: int):
        self.user_id = user_id
        self.needed = needed

@app.exception_handler(InsufficientCreditsError)
async def insufficient_credits_handler(request: Request, exc: InsufficientCreditsError):
    return JSONResponse(
        status_code=402,
        content={
            "error": "insufficient_credits",
            "user_id": exc.user_id,
            "credits_needed": exc.needed,
        },
    )

def charge_for_llm_call(user_id: str, cost: int):
    if get_balance(user_id) < cost:
        raise InsufficientCreditsError(user_id, cost)  # no HTTP knowledge needed here

@app.post("/chat")
async def chat(user_id: str, message: str):
    charge_for_llm_call(user_id, cost=10)  # handler above converts this to a 402 if raised
    return {"reply": "..."}
```

**Common pitfall**

Registering an exception handler and assuming it also catches exceptions raised *inside middleware* or during ASGI-level failures — by default, exception handlers only cover exceptions raised while handling a route (including its dependencies), not exceptions raised in middleware, which propagate differently and typically need to be handled inside the middleware itself with a `try/except` around `call_next`.

**Interview angle**

Q: How do you avoid leaking internal exception details (stack traces, DB error messages) to API clients while still logging them properly server-side?
A: Register a catch-all handler for your base/unexpected exception types (or `Exception` itself) that logs the full exception server-side (with traceback) but returns a generic, sanitized error body and a `500` to the client — combined with specific handlers for known domain exceptions (like a custom `NotFoundError` or `InsufficientCreditsError`) that map to meaningful status codes and safe-to-expose messages. This keeps sensitive internals out of responses without requiring every route to defensively catch and sanitize errors itself.
