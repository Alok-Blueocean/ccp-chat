# Resilience, Logging & Container Networking Concepts

Everything in files 01–05 assumes the app is *running*. This file is about the layer underneath that: what happens when a dependency the app expects isn't there, how the process tells you about it, and why a perfectly healthy server can still be unreachable from your browser.

It's written from a real debugging session on this repo. The presenting symptom was a wall of `connection refused` warnings from Postgres and an app "link" that wouldn't open. Three genuinely separate faults were tangled together:

1. The app treated an unreachable Postgres as a hard dependency, so it retried forever and logged endlessly.
2. `DATABASE_URL` said `localhost`, which inside a container means *the container itself*, not the host.
3. The container never published port 8000 — the reason the link was dead had nothing to do with the database at all.

Each concept below is one of those layers, plus the logging work that made the whole thing observable in the first place.

## Graceful Degradation of Optional Dependencies

**What it is**
Chat memory is a *nice-to-have*: it lets `/chat` recall earlier turns in a session. Retrieval and generation work fine without it. So Postgres being down should cost you conversation history, not the entire service. The pattern is to classify every dependency as **required** (fail loudly at boot — Qdrant, OpenAI) or **optional** (degrade silently to a reduced feature set), and then actually enforce that classification in code. `postgres/client.py` already had half of this — `init_pool()` returned early when `DATABASE_URL` was unset — but "configured yet unreachable" fell through into the hard-failure path.

**How it works**
- `init_pool()` builds the pool with `open=False`, then calls `pool.open(wait=True, timeout=3.0)`. That `wait=True` is the whole trick: it forces the pool to prove it can establish a real connection *now*, converting a lazy, silent misconfiguration into an immediate, catchable exception.
- On failure it calls `pool.close(timeout=1.0)` and leaves the module global `_pool` as `None`. Closing matters for a non-obvious reason covered in the next section — it's what stops the background reconnect thread.
- Every consumer already checked `if pool is None: return []` / `return`, so a `None` pool automatically means "memory disabled" everywhere without touching call sites.
- `ChatMemoryStore`'s two methods additionally wrap their queries in `try/except`, because startup success doesn't guarantee runtime success — Postgres can die *after* boot. A failed read returns `[]` (the LLM just loses history); a failed write logs a warning and moves on.
- `ensure_schema()` is likewise non-fatal: if the DDL can't be applied, it warns and returns rather than killing startup.
- In `main.py`, the whole block is belt-and-braces wrapped so no unexpected pool exception can abort `lifespan`.

**Example**
```python
# File: postgres/client.py
pool = ConnectionPool(
    conninfo=settings.database_url,
    min_size=1, max_size=10,
    open=False,                                   # don't connect in the constructor
    kwargs={"connect_timeout": 3},                # libpq-level per-attempt cap
)
try:
    pool.open(wait=True, timeout=3.0)             # prove connectivity, or raise
except Exception as exc:
    pool.close(timeout=1.0)                       # also stops the retry worker
    logger.warning("Postgres unreachable (%s); continuing without chat memory.", exc)
    return                                        # _pool stays None → memory disabled
_pool = pool
```

```python
# File: app/services/chat_memory.py — degrade, don't raise
try:
    with pool.connection(timeout=_ACQUIRE_TIMEOUT_SECONDS) as conn:
        ...
except Exception as exc:
    logger.warning("Could not load chat history: %s", exc)
    return []          # no memory is better than a failed chat turn
```

**Where in this repo**
`postgres/client.py::init_pool` (the `open=False` → `open(wait=True)` handshake), `::ensure_schema` (non-fatal DDL), `app/services/chat_memory.py::ChatMemoryStore.fetch_recent_messages/append_turn` (runtime try/except), `main.py::lifespan` (outer guard).

**Interview angle**
Q: Isn't swallowing exceptions an anti-pattern? You've hidden a real failure.
A: Swallowing is only wrong when the caller needed the error to make a decision. Here the decision is already made by product requirements: a missing history row must not turn a working answer into a 500. What makes it defensible rather than sloppy is that (a) it's scoped to one optional subsystem, not a blanket `except: pass` around business logic, (b) every swallow emits a `WARNING` with the exception text, so it's visible in logs and alertable, and (c) the degraded behaviour is well-defined — `[]` means "no history," which the prompt builder already handles. The dangerous version is swallowing a *write* to something you claimed was durable; if chat history were a compliance requirement rather than a UX nicety, the correct answer flips to failing the request loudly.

## Fail-Fast vs. Retry-Forever: Connection Pool Semantics

**What it is**
The original log spam was not a bug in the app's logic — it was the connection pool doing exactly what it was told. `ConnectionPool(..., open=True)` returns to the caller *immediately* without a working connection, and spawns a background worker that keeps trying to reach the database for the life of the process, warning on every attempt with a growing backoff. The observed timestamps show the backoff clearly: attempts at `:56.718`, `:57.652`, `:59.537`, `:03.241` — roughly 1s, 2s, 4s apart. Startup "succeeded," the log said `pool opened`, and yet no connection existed. That gap between *reported* and *actual* readiness is the real lesson.

**How it works**
- Three independent timeouts govern this, and conflating them is a common source of confusion:
  - `kwargs={"connect_timeout": 3}` — passed through to libpq, caps a **single TCP/auth attempt**. Without it, an attempt against a black-holed address can hang far longer than you'd expect.
  - `pool.open(wait=True, timeout=3.0)` — caps how long **startup** waits for the pool to reach `min_size`. Raises `PoolTimeout` with the message `pool initialization incomplete after 3.0 sec`.
  - `pool.connection(timeout=3.0)` — caps how long **a request** waits to borrow a connection from the pool. The library default here is 30 seconds, which is disastrous under load: if the DB is gone, every in-flight request parks for 30s, exhausting the worker pool and turning a degraded feature into a site-wide outage.
- `pool.close()` is what terminates the reconnect worker. Setting `_pool = None` alone would leave an orphaned thread happily logging `connection refused` forever, since the pool object stays alive as long as its own thread references it.
- `close()` takes its own timeout (default 5s) for joining worker threads. During testing, closing while a worker was mid-connect produced `couldn't stop thread 'pool-1-worker-0' within 5.0 seconds` and added a 5-second stall to startup. Passing `close(timeout=1.0)` plus a short `connect_timeout` cut the total unreachable-DB startup penalty to roughly 4 seconds.

**Example**
```
# Symptom — pool "opened" but nothing connected; worker retries with backoff
16:29:56,674 | postgres.client | INFO    | Postgres connection pool opened for chat memory.
16:29:56,718 | psycopg.pool    | WARNING | error connecting in 'pool-1': connection refused
16:29:57,652 | psycopg.pool    | WARNING | error connecting in 'pool-1': connection refused
16:29:59,537 | psycopg.pool    | WARNING | error connecting in 'pool-1': connection refused
16:30:03,241 | psycopg.pool    | WARNING | error connecting in 'pool-1': connection refused
       ↑ 1s, 2s, 4s backoff — forever, for the life of the process

# After: one warning, then silence, and the app serves traffic
WARNING | Postgres unreachable (pool initialization incomplete after 3.0 sec); continuing without chat memory.
```

**Where in this repo**
`postgres/client.py` — `_CONNECT_TIMEOUT_SECONDS` (3.0, used for both the libpq attempt and the `open()` wait); `app/services/chat_memory.py::_ACQUIRE_TIMEOUT_SECONDS` (3.0, overriding the 30s borrow default).

**Interview angle**
Q: Why not let it retry forever? The database might come back.
A: Retrying forever is right for a *worker* that has nothing else to do, and wrong for an *HTTP server* that must stay responsive. The killer isn't the retry, it's that retrying while advertising the pool as open means requests queue against a connection that will never arrive — you've converted "one feature is off" into "every request hangs for 30 seconds." Fail fast, degrade explicitly, and let the orchestrator or a health check drive recovery. The trade-off I accepted is that this design won't self-heal: if Postgres comes up later, the pool stays `None` until restart. That's fine for a container you can restart cheaply, and the honest fix if it mattered would be a periodic background re-init rather than a blocking retry on the request path.

## `EXPOSE` vs. Published Ports (Why the Link Was Dead)

**What it is**
The most instructive fault of the session, because the logs were *completely healthy* — `Application startup complete`, `Uvicorn running on http://0.0.0.0:8000` — and the link still refused to open. `EXPOSE 8000` in a Dockerfile is **documentation only**. It records that the image intends to serve on 8000; it creates no route from the host into the container. Only `-p`/`--publish` (or Compose's `ports:`) sets up the actual port forward. Binding uvicorn to `0.0.0.0` is necessary but not sufficient — it makes the server reachable from outside *the container's own network namespace*, and if nothing forwards a host port into that namespace, there is still no path.

**How it works**
- `docker ps` distinguishes the two cases in its `PORTS` column, and this is the fastest way to spot the bug:
  - `8000/tcp` — merely exposed. **Unreachable from the host.**
  - `0.0.0.0:8000->8000/tcp` — published. Reachable.
- `docker inspect` is the authoritative check: `.HostConfig.PortBindings` was `map[]` and `.HostConfig.PublishAllPorts` was `false`, confirming no mapping had ever been requested.
- `-p 8000:8000` reads *host port : container port*. They needn't match — `-p 9000:8000` serves the same container on host port 9000, which is how you run several instances of one image side by side.
- `-P` (capital) publishes every `EXPOSE`d port to random high host ports — convenient, unpredictable, rarely what you want.
- Five containers from earlier attempts were all running in the exposed-but-unpublished state simultaneously, each holding memory and none reachable. Worth internalising: a container that starts successfully is not the same as a service that works.

**Example**
```
# docker ps — the tell is in the PORTS column
ports=[8000/tcp]                   ← EXPOSE only. Nothing forwards here. Dead link.
ports=[0.0.0.0:8000->8000/tcp]     ← published. Works.

# The authoritative check
docker inspect <name> --format "publish={{.HostConfig.PublishAllPorts}} bindings={{.HostConfig.PortBindings}}"
publish=false bindings=map[]        ← no mapping exists

# The fix
docker run -d -p 8000:8000 --env-file .env ccp-chat
#              ^^^^ host:container
```

**Where in this repo**
`Dockerfile` — `EXPOSE 8000` plus `CMD ... uvicorn main:app --host 0.0.0.0 --port ${PORT}` (the `--host 0.0.0.0` half of the requirement); `docker-compose.yml` — `ports: ["8000:8000"]` (the publishing half).

**Interview angle**
Q: If `EXPOSE` doesn't do anything, why write it?
A: It's not useless, it's just metadata rather than mechanism. It documents intent for whoever runs the image, it's what `-P` reads to decide which ports to map, and some tooling and orchestrators use it as a default. The distinction that actually matters in an interview is understanding that container networking has two separate hops — *is the process listening on an address reachable from outside its namespace* (`--host 0.0.0.0`, not `127.0.0.1`) and *is there a forward from the host into that namespace* (`-p`). Both must hold. Binding to `127.0.0.1` inside a container is the mirror-image bug: published port, but the server only accepts connections from within the container, so you still get connection-refused.

## `localhost` Inside a Container, and Host/Service Resolution

**What it is**
`.env` contained `DATABASE_URL=postgresql://operator:CastAIP@localhost:2284/airflow`, which is correct when you run uvicorn directly on Windows and wrong the moment the app is containerised. Every container gets its own network namespace, so `localhost` resolves to *that container*. Nothing listens on 2284 there, hence `connection refused`. Confirming this took one useful diagnostic step: connecting to `localhost:2284` **from the Windows host** succeeded — the error was `relation "chat_messages" does not exist`, which proves a Postgres was listening and reachable, just not from where the app was looking.

**How it works**
The right hostname depends entirely on where Postgres runs:

| Postgres runs on | Host to use | Port | Notes |
|---|---|---|---|
| The Windows/macOS host | `host.docker.internal` | `2284` (published) | Add `--add-host=host.docker.internal:host-gateway` for Linux parity |
| A Compose service named `db` | `db` | `5432` (container's own) | Compose's built-in DNS resolves service names |
| Another container on a user-defined network | that container's `--name` | container port | Requires a shared `--network`; the default bridge has no name resolution |
| Same container (rare) | `localhost` | — | Only correct for a sidecar-in-one-container setup |

- Two port subtleties bite people here. Container-to-container traffic uses the **container's own** port (`5432`), *not* the published one — publishing exists for the host's benefit and is irrelevant on the internal network. Conversely, host-to-container traffic via `host.docker.internal` must use the **published** port (`2284`).
- `host.docker.internal` is provided automatically by Docker Desktop but not by Docker Engine on Linux; `extra_hosts: ["host.docker.internal:host-gateway"]` makes it resolve there too, so the same Compose file works on a developer laptop and a Linux CI box.
- The override lives in `docker-compose.yml`'s `environment:` block rather than in `.env`, deliberately. `.env` keeps the `localhost` value that's correct for running uvicorn natively, while Compose's `environment:` takes precedence over `env_file:` for the containerised case. One repo, two valid run modes, no editing between them. (The same precedence applies to plain `docker run`: an explicit `-e` beats `--env-file`.)

**Example**
```yaml
# File: docker-compose.yml
services:
  api:
    env_file: [.env]                    # localhost:2284 — correct for native runs
    environment:
      # Overrides env_file. Inside a container, localhost is the container itself.
      DATABASE_URL: postgresql://operator:CastAIP@host.docker.internal:2284/airflow
    extra_hosts:
      - "host.docker.internal:host-gateway"   # Linux parity

  # If Postgres moves into Compose, the service name becomes the hostname
  # and the port becomes 5432 (container port), not 2284 (published port):
  #   postgresql://operator:CastAIP@db:5432/airflow
```

**Where in this repo**
`docker-compose.yml` (the `environment:` override, `extra_hosts:`, and the commented-out `db:` service showing the service-name form); `.env` line 26 (the native-run value); consumed by `app/core/configs.py::Settings.database_url` → `postgres/client.py::init_pool`.

**Interview angle**
Q: How would you have found this faster?
A: By testing connectivity *from inside the failing namespace* instead of from where it's convenient. The log said `connection refused` for `127.0.0.1:2284`, and the tempting read is "Postgres is down" — but connecting from the host proved it was up, which immediately reframes the question from "is the DB running" to "who is asking, and from where." `docker exec <container> sh -c "nc -z host.docker.internal 2284"` answers it in one shot. The general principle: when a network error names an address, verify that address from the same vantage point as the code that failed, because `localhost` is the most context-dependent hostname there is.

## Logging: Docker Logs vs. Application Log Files

The easiest way to understand this setup is to separate **Docker logs** from **application log files**. They are two different mechanisms solving two different problems, and most confusion comes from treating them as one thing.

### 1. Docker logs — simplest picture

Your FastAPI application writes something like:

```python
logger.info("User asked a question")
```

If that goes to **stdout/stderr**, Docker captures it:

```text
FastAPI
   │
   │ stdout / stderr
   ▼
Docker
   │
   │ logging driver
   ▼
Docker's log storage
```

Then you can see it with:

```bash
docker logs ccp-chat
```

You **don't need to know where Docker physically stores the file**. `docker logs` is the stable interface.

### 2. What does `docker logs` actually show?

Suppose your application produces:

```text
Application startup complete.
Postgres connection pool opened.
User asked question.
RAG search completed.
```

You run:

```bash
docker logs ccp-chat
```

and Docker shows exactly that:

```text
Application startup complete.
Postgres connection pool opened.
User asked question.
RAG search completed.
```

If you run:

```bash
docker logs -f ccp-chat
```

the `-f` means **follow**, so Docker keeps showing new lines as they arrive:

```text
User asked question
RAG search completed
User asked another question
Calling OpenAI
Response generated
...
```

It's similar to:

```bash
tail -f app.log
```

### 3. Where does Docker actually store these logs?

With the default `json-file` logging driver, Docker stores them internally. Conceptually:

```text
Docker
  │
  └── container
        │
        └── container log
```

On a Docker Desktop / WSL2 setup, the physical storage is inside Docker's Linux environment, **not** simply:

```text
C:\your-project\logs
```

Concretely on this machine: `docker inspect` reports the path as `/var/lib/docker/containers/<id>/<id>-json.log`, but that's the daemon's own namespaced view — the file is actually reachable at `/mnt/docker-desktop-disk/data/docker/containers/<id>/<id>-json.log` inside the `docker-desktop` WSL distro, owned by root.

That mismatch is the point: these physical paths are implementation details that change across Docker versions and platforms, so you generally **shouldn't go looking for the physical Docker log file**. Use:

```bash
docker logs ccp-chat
```

### 4. But there is a problem: container deletion deletes the logs

Imagine:

```text
Container
   │
   ├── Application
   └── Docker logs
```

You do:

```bash
docker rm ccp-chat
```

The container disappears **and its Docker-managed logs disappear with it**.

For example:

```bash
docker logs ccp-chat      # works today
docker rm ccp-chat        # logs are gone with the container
docker run ... ccp-chat   # tomorrow: a new container, none of the old logs
```

That's important — and it's the reason for everything in the next section.

### 5. That's why your application is also writing `app.log`

Python logging is configured with **two handlers**:

```text
                  Python application
                         │
                    logger.info()
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
        StreamHandler         RotatingFileHandler
             │                       │
             ▼                       ▼
        stdout/stderr             app.log
             │                       │
             ▼                       ▼
          Docker                  /ccp/logs
             │                       │
             ▼                       ▼
      docker logs              Host ./logs
```

This is the most important concept in the whole file.

### 6. Why two places?

Because they solve different problems.

**StreamHandler** sends logs to `stdout/stderr`, which is what makes `docker logs ccp-chat` work at all.

**RotatingFileHandler** sends logs to `/ccp/logs/app.log` *inside* the container. But `docker-compose.yml` does:

```yaml
volumes:
  - ./logs:/ccp/logs
```

So:

```text
Container                         Windows
──────────────────               ──────────────

/ccp/logs/app.log   ──────────→  ./logs/app.log
```

Therefore the file actually survives container deletion.

### 7. Why is this called a "bind mount"?

This:

```yaml
volumes:
  - ./logs:/ccp/logs
```

means:

> Take the `logs` directory from my computer and make it available inside the container as `/ccp/logs`.

So:

```text
Windows project
C:\CCP\logs
       │
       │ bind mount
       ▼
Docker container
/ccp/logs
```

If Python writes `/ccp/logs/app.log`, you can open it on your machine at `C:\CCP\logs\app.log`. Unlike Docker's internal `json-file` logs, this file lives **outside the container's filesystem**.

### 8. Why rotate the application log?

Imagine your application writes to `app.log` forever:

```text
After 1 day:      50 MB
After 10 days:   500 MB
After 100 days:    5 GB
```

Eventually your disk fills up. The configuration uses:

```python
RotatingFileHandler(
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
)
```

Meaning:

```text
app.log       ← current
app.log.1
app.log.2
app.log.3
app.log.4
app.log.5
```

Approximately `10 MB × 6 files ≈ 60 MB maximum`. When `app.log` reaches 10 MB:

```text
app.log
   ↓
app.log.1
```

and a fresh `app.log` is created. The oldest (`app.log.5`) is discarded.

Worth noting the container-level equivalent: the `json-file` driver has **no rotation by default**, so Docker's own log grows unbounded until it fills the VM disk. Bound it with `--log-opt max-size=10m --log-opt max-file=3`.

### 9. Why not just use `app.log`?

Because Docker's philosophy is generally:

> **Application writes logs to stdout/stderr; the platform collects them.**

For example, in AWS you could eventually have:

```text
FastAPI
   │
   │ stdout
   ▼
Docker
   │
   ▼
AWS logging
   │
   ▼
CloudWatch
```

Then you don't need `app.log` at all. This is the **twelve-factor** approach — logs are an event *stream*, not a file the app manages. The current file logging is a pragmatic intermediate solution: bind-mounted so it isn't really container state, and rotation-bounded so it can't eat the disk.

### 10. Why does your project currently use both?

For the current development setup:

```text
                     RAG APP
                        │
                  Python logger
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
          Console                File
             │                     │
             ▼                     ▼
      stdout/stderr             app.log
             │                     │
             ▼                     ▼
      Docker logs              ./logs/
             │                     │
             ▼                     ▼
   docker logs ccp-chat       survives container
```

So you have **two ways of looking at logs**:

- **Quick troubleshooting** — `docker logs -f ccp-chat`
- **Persistent logs** — open `logs/app.log`

### 11. What does the Uvicorn part mean?

This is a slightly more advanced Python logging issue. Normally:

```text
Your logger
    ↓
root logger
    ↓
StreamHandler + FileHandler
```

So your application's logs go to both places. But Uvicorn has its own loggers:

```text
uvicorn
uvicorn.error
uvicorn.access
```

and it sets `propagate = False`, which essentially means:

> "Don't send my logs up to the root logger."

Therefore, without `_adopt_uvicorn_loggers()`:

```text
Uvicorn
   │
   └── its own handler
          │
          └── console ✅

Root FileHandler
          ↑
          │
       blocked ❌
```

So `app.log` might contain:

```text
Postgres connection pool opened
RAG request started
RAG request completed
```

but **not**:

```text
Application startup complete
GET /chat 200
```

`_adopt_uvicorn_loggers()` fixes this by saying:

```python
lg.handlers.clear()
lg.propagate = True
```

Now:

```text
Uvicorn
   │
   ▼
Root logger
   │
   ├── Console
   │
   └── app.log
```

Which is why the verified output from the running container has uvicorn's startup *and* access lines in the same file as the app's own:

```
| app.core.logging_config | INFO | Logging to /ccp/logs/app.log (level=INFO, rotate at 10MB x5)
| uvicorn.error           | INFO | Started server process [7]
| postgres.client         | INFO | Postgres connection pool opened for chat memory.
| postgres.client         | INFO | Chat memory schema ensured.
| uvicorn.error           | INFO | Application startup complete.
| uvicorn.access          | INFO | 172.18.0.1:59756 - "GET / HTTP/1.1" 200
```

Ordering is what makes this work: uvicorn configures logging first, *then* imports the app, so `setup_logging()` runs afterwards and its adjustment wins.

### 12. The `basicConfig` part

Another useful interview concept. Suppose you do:

```python
logging.basicConfig(...)
```

Python basically says:

> "If root logging hasn't already been configured, I'll configure it."

So:

```text
First basicConfig()
       ↓
Configures root
       ↓
Second basicConfig()
       ↓
Usually does nothing
```

That's why the existing `app/services/logger.py::get_logger` can keep calling `basicConfig()` without creating duplicate handlers — the new central configuration has already configured root, so `basicConfig` is a convenience mechanism that only acts when root has no handlers (absent `force=True`).

### The whole thing in one picture

```text
                         RAG APPLICATION
                               │
                         Python logger
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
             stdout/stderr              app.log
                  │                         │
                  ▼                         ▼
               Docker                   /ccp/logs
                  │                         │
                  ▼                         ▼
          Docker log driver             bind mount
                  │                         │
                  ▼                         ▼
          docker logs ccp-chat         ./logs/app.log
                  │                         │
                  │                         │
            Temporary-ish             Persistent
            container logs            host file
```

In production you would typically simplify to:

```text
RAG application
      │
      ▼
stdout/stderr
      │
      ▼
Docker
      │
      ▼
AWS logging / CloudWatch
      │
      ▼
centralized observability
```

### Logs are not your whole LLM observability story

Since this project already uses Langfuse (see [03-evaluation-and-observability.md](03-evaluation-and-observability.md)), it's worth separating the purposes explicitly:

| Layer | What it's for |
|---|---|
| **Docker / CloudWatch logs** | Infrastructure and application errors, startup, HTTP requests, exceptions |
| **Langfuse** | LLM traces — prompts, responses, tokens, latency, model calls, RAG spans |
| **Metrics** | CPU, memory, request rate, error rate, latency percentiles |
| **Evaluation / drift** | Answer quality, retrieval quality, data drift |

So **Docker logs are not a complete LLM observability system.** They're your application's operational logs, and confusing the two is how teams end up with no visibility into *why* an answer was bad, only that a request returned 200.

**Where in this repo**
`app/core/logging_config.py::setup_logging/_adopt_uvicorn_loggers`; called at the top of `main.py` (before the router imports, so module-level loggers inherit the configuration); `Dockerfile` (`ENV LOG_DIR=/ccp/logs`, `mkdir -p /ccp/logs`); `docker-compose.yml` (`./logs:/ccp/logs` bind mount, and the default `json-file` driver since there's no explicit `logging:` block); `.gitignore` (`logs/`, `*.log`). Pre-existing per-module helper: `app/services/logger.py::get_logger`. The offline pipelines configure their own `FileHandler` independently (`indexing_pipeline/pipeline_async.py`).

Two implementation details in `setup_logging()` worth knowing for the same reasons: it reads `LOG_DIR`/`LOG_FILE`/`LOG_LEVEL` from `os.environ` rather than `Settings`, because `app/core/configs.py::Settings` has many required fields and could raise `ValidationError` — logging must be upstream of config validation so you can *see* the config error. And handlers are tagged with a marker attribute so a reload can't attach duplicates, the classic cause of every line appearing two or three times.

**Interview angle**
Q: If `docker logs` works, why bother writing a log file at all?
A: Here, so logs outlive the container and can be read without the Docker CLI — but in a mature deployment you'd do neither. Twelve-factor says a container writes to stdout/stderr and the platform collects (`awslogs`, Fluentd, Loki). Writing files inside a container reintroduces exactly what containers were meant to remove: disk growth, rotation, and state in an ephemeral filesystem. The version here is a deliberate middle step — bind-mounted so it isn't really container state, and size-bounded so it can't fill the disk.

## Production Logging: Loggers, Handlers, Levels

**What it is**
Everything above is *this* project's setup. This section is the general model underneath it. Think of Python logging as a **small pipeline**:

```text
Your code
   ↓
Logger
   ↓
LogRecord
   ↓
Filter
   ↓
Handler
   ↓
Formatter
   ↓
Where the log goes
```

The four main logging objects are **Logger, Handler, Formatter, and Filter**. Let's understand each with a simple RAG example.

### 1. Logger — "I want to say something"

In your code:

```python
import logging

logger = logging.getLogger(__name__)

logger.info("RAG search started")
```

The **Logger** is what your application uses to create a log message.

Think:

> "Hey logging system, something happened."

For example:

```python
logger.info("RAG search started")
logger.warning("Qdrant response was slow")
logger.error("OpenAI call failed")
```

The logger itself doesn't decide whether the message goes to a file, CloudWatch, terminal, etc.

It primarily creates a **LogRecord** containing information such as:

```text
message = "RAG search started"
level = INFO
logger = "app.retrieval"
timestamp = ...
module = retriever.py
line = 42
```

### 2. LogRecord — "the actual package of information"

You normally don't create this yourself.

When you do:

```python
logger.info("RAG search started")
```

Python internally creates something like:

```text
LogRecord
────────────────────
message: RAG search started
level: INFO
logger: app.retrieval
timestamp: 10:35:20
file: retriever.py
line: 42
```

Think of it as a **parcel** containing all information about the event.

### 3. Handler — "Where should the log go?"

This is probably the most important concept. A **Handler decides the destination**.

For example:

```text
                    LogRecord
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Console       File       CloudWatch
```

In Python:

```python
console_handler = logging.StreamHandler()
```

means:

> Send logs to stdout/stderr.

And:

```python
file_handler = logging.handlers.RotatingFileHandler(
    "logs/app.log"
)
```

means:

> Send logs to a file.

A logger can have **multiple handlers**.

For your application:

```text
                    logger.info()
                         │
                         ▼
                     LogRecord
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        StreamHandler       FileHandler
              │                     │
              ▼                     ▼
           Console              app.log
              │
              ▼
          Docker logs
              │
              ▼
          CloudWatch
```

This is why the CloudWatch question matters. In production you can have:

```text
Logger
   │
   ▼
StreamHandler
   │
   ▼
stdout
   │
   ▼
Docker
   │
   ▼
CloudWatch
```

and potentially **no `app.log` at all**.

### 4. Formatter — "How should it look?"

Suppose you write:

```python
logger.info("RAG search completed")
```

The Formatter decides how that becomes text.

Without much formatting:

```text
RAG search completed
```

With a formatter:

```python
"%(asctime)s | %(name)s | %(levelname)s | %(message)s"
```

you might get:

```text
2026-08-11 11:30:42 | app.retrieval | INFO | RAG search completed
```

So:

```text
Logger     → What happened?
Handler    → Where does it go?
Formatter  → What does it look like?
```

The Formatter is *how the record is rendered as text* — nothing more.

### 5. Filter — "Should this log be allowed / add extra information?"

Filter is optional. Think of it as a **security guard**.

For example, you might want every log to have:

```text
request_id
```

A filter can add it:

```text
Incoming request
        ↓
Filter
        ↓
request_id = abc123
        ↓
LogRecord
```

Then your CloudWatch log might be:

```json
{
  "level": "INFO",
  "message": "RAG search completed",
  "request_id": "abc123"
}
```

This is exactly the idea behind a `RequestIdFilter` (see §16 below).

### 6. Now put everything together

Suppose a user asks your RAG application:

```text
"What is karma?"
```

Your code:

```python
logger.info("RAG search started")
```

What happens?

**Step 1 — Logger**

```text
logger.info(...)
```

Logger creates a LogRecord.

**Step 2 — LogRecord**

```text
{
    message: "RAG search started",
    level: INFO,
    logger: "app.retrieval",
    timestamp: ...
}
```

**Step 3 — Filter**

Maybe adds:

```text
request_id = abc123
```

**Step 4 — Handler**

The record can go to:

```text
Console Handler
        +
File Handler
```

**Step 5 — Formatter**

Console might produce:

```text
2026-08-11 11:30:42 | INFO | RAG search started
```

**Step 6 — Destination**

```text
stdout
   ↓
Docker
   ↓
CloudWatch
```

That's the complete flow.

### 7. What is Logger hierarchy?

This looks complicated initially:

```text
root
 ├── app
 │    └── app.services
 │         └── app.services.chat
 │
 └── postgres
      └── postgres.client
```

But think of it as **folders**.

If your Python file is:

```text
app/services/chat.py
```

and you do:

```python
logger = logging.getLogger(__name__)
```

then:

```text
__name__
=
app.services.chat
```

Python automatically creates the hierarchy:

```text
root
  ↓
app
  ↓
app.services
  ↓
app.services.chat
```

Dotted logger names form this hierarchy.

### 8. Why is this useful?

Suppose your application has:

```text
app.retrieval
app.llm
app.database
```

You can say:

```python
logging.getLogger("app.retrieval").setLevel(logging.DEBUG)
```

Now you can get detailed retrieval logs:

```text
Retrieved chunk 1
Retrieved chunk 2
Reranking started
Reranking completed
```

while keeping everything else at INFO.

### 9. What is propagation?

This is another important concept.

Suppose:

```text
app.services.chat
        ↓
app.services
        ↓
app
        ↓
root
```

If you do:

```python
logger.info("Chat request")
```

the log can travel **up the hierarchy**.

```text
app.services.chat
        │
        ▼
app.services
        │
        ▼
app
        │
        ▼
root
        │
        ├── Console
        └── File
```

This is called **propagation**.

That's why you can configure the root logger once and capture logs from:

```text
your application
+
psycopg
+
httpx
+
other libraries
```

without modifying those libraries.

### 10. What does `propagate=False` do?

It means:

> Stop here. Don't send this log further up.

For example:

```python
logger.propagate = False
```

Flow becomes:

```text
app.services.chat
        │
        X
        │
       root
```

This is why Uvicorn was special earlier in this file: Uvicorn uses its own handlers and `propagate=False`, so its logs don't automatically reach your root handlers.

### 11. Logging levels — very important

Python has:

```text
DEBUG       10
INFO        20
WARNING     30
ERROR       40
CRITICAL    50
```

Think of them as **importance levels**.

For your RAG application:

**DEBUG** — developer details:

```python
logger.debug("Retrieved %s chunks", len(chunks))
```

Example:

```text
Retrieved 20 chunks
Reranking top 10
Embedding took 250ms
```

Usually disabled in production.

**INFO** — normal operation:

```python
logger.info("RAG search completed")
```

Example:

```text
Application started
Database connected
Request completed
```

**WARNING** — something went wrong, **but your application handled it**:

```python
logger.warning(
    "Postgres unavailable; continuing without chat memory"
)
```

Application continues. The distinction that matters:

> **WARNING = I handled it.**
> **ERROR = someone needs to look.**

**ERROR** — something failed and needs investigation:

```python
logger.error("LLM request failed")
```

**CRITICAL** — application cannot continue:

```python
logger.critical("DATABASE_URL is missing")
```

### 12. The "two gates" concept

This is one of the most important things to understand.

Suppose:

```python
logger.debug("Retrieved 20 chunks")
```

You might think:

> "I set logging to DEBUG, so why don't I see it?"

There are actually **two gates**.

```text
Logger
  │
  │ Gate 1
  ▼
Handler
  │
  │ Gate 2
  ▼
Output
```

Example:

```python
logger.setLevel(logging.DEBUG)

console.setLevel(logging.INFO)
```

The logger says:

> DEBUG is allowed.

But the console handler says:

> I only accept INFO and above.

Therefore:

```text
DEBUG
  ↓
Logger → ✅
  ↓
Handler → ❌
  ↓
Nothing appears
```

This is the **two-gate system**, and it's the single most common cause of "why isn't my log showing up?"

### 13. Very useful production setup

You could configure:

```text
Console
  INFO+

Error file
  ERROR+
```

So:

```text
DEBUG    → nowhere
INFO     → console
WARNING  → console
ERROR    → console + error file
CRITICAL → console + error file
```

This is a very useful pattern — failures land in one small file you can eyeball or alert on.

### 14. Why `dictConfig`?

Instead of configuring logging everywhere:

```python
logger.setLevel(...)
handler.setLevel(...)
formatter...
```

you put configuration in **one place**.

Conceptually:

```python
LOGGING = {
    "handlers": {
        "console": ...,
        "file": ...,
        "errors": ...
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file", "errors"]
    }
}
```

Then:

```python
logging.config.dictConfig(LOGGING)
```

This declarative approach is what production services use — it's the same structure uvicorn and Django use, and it can be loaded from YAML/JSON so ops can change log levels without a code change.

### 15. Why JSON logging matters for CloudWatch

This becomes particularly important for an AWS deployment.

Instead of:

```text
INFO | RAG completed | session=123 | latency=812
```

send:

```json
{
  "level": "INFO",
  "msg": "RAG completed",
  "session_id": "123",
  "latency_ms": 812
}
```

Now CloudWatch or another logging platform can search:

```text
session_id = "123"
```

instead of trying to search text.

Fields get attached with `extra=`:

```python
logger.info("RAG completed", extra={"session_id": "123", "latency_ms": 812})
```

### 16. Request ID — extremely useful for your RAG app

Imagine 3 users make requests at almost the same time:

```text
User A → request A
User B → request B
User C → request C
```

Their logs become mixed:

```text
RAG started
RAG started
Qdrant search
LLM call
Qdrant search
LLM response
LLM response
```

You don't know which belongs to whom.

Add:

```text
request_id
```

Then:

```text
abc123 | RAG started
xyz789 | RAG started
abc123 | Qdrant search
xyz789 | LLM call
abc123 | LLM response
```

Now you can search:

```text
request_id = abc123
```

and see the complete request journey.

Use Python's `ContextVar` to hold it, because it is safe for concurrent async requests:

```python
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True
```

### 17. Exception logging

Don't do this:

```python
try:
    answer = call_llm()
except Exception as e:
    logger.error("LLM failed: %s", e)
```

You get:

```text
LLM failed: timeout
```

But you don't know **where** it happened.

Instead:

```python
try:
    answer = call_llm()
except Exception:
    logger.exception("LLM call failed")
    raise
```

You get:

```text
LLM call failed
Traceback:
  File "chat.py", line 42
  File "llm.py", line 88
  ...
TimeoutError
```

`logger.exception()` is essentially an ERROR log plus the full traceback, and should be used inside `except`.

### 18. One performance trick

Prefer:

```python
logger.debug(
    "Retrieved %s chunks for %s",
    len(chunks),
    query
)
```

instead of:

```python
logger.debug(
    f"Retrieved {len(chunks)} chunks for {query}"
)
```

Why?

If DEBUG is disabled, Python can avoid constructing the first message's final string.

With the f-string, the string is constructed **before** logging decides that DEBUG isn't needed.

### 19. For an LLM/RAG application, don't log everything

This is especially important here.

Don't do:

```python
logger.info(f"User prompt: {prompt}")
logger.info(f"LLM response: {response}")
```

because prompts/responses can contain:

* passwords
* personal information
* confidential data
* sensitive business information

Prefer identifiers, token counts, latency, etc., and use **Langfuse** for full prompt/response data rather than general logs.

So a better log is:

```json
{
  "level": "INFO",
  "msg": "RAG request completed",
  "session_id": "abc123",
  "latency_ms": 812,
  "input_tokens": 450,
  "output_tokens": 230
}
```

And Langfuse handles the detailed LLM trace. Note that logging the *raw* prompt would also bypass this repo's PII pipeline entirely (`app/guardrails/pii_guard.py`, see [02-guardrails-and-safety.md](02-guardrails-and-safety.md)).

### The mental model to remember

You don't need to memorize all the Python logging APIs. Remember these **six questions**:

```text
1. LOGGER
   Who is producing the log?

2. LOG RECORD
   What happened + metadata?

3. LEVEL
   How important is it?

4. FILTER
   Should I modify/allow this record?

5. HANDLER
   Where should it go?

6. FORMATTER
   How should it look?
```

Then your production RAG architecture becomes:

```text
                FastAPI / RAG
                     │
              logger.info()
                     │
                     ▼
                 LogRecord
                     │
             ┌───────┴────────┐
             │                │
           Filter           Level
             │
             ▼
          Handlers
       ┌─────┴──────┐
       │            │
       ▼            ▼
    stdout       file*
       │
       ▼
    Docker
       │
       ▼
  CloudWatch

* file mainly for local/dev;
  in production you can omit it.
```

And separately:

```text
RAG / LLM execution
        │
        ▼
     Langfuse
        │
        ├── traces
        ├── prompts
        ├── model calls
        ├── tokens
        └── latency
```

That distinction is **very important for your LLMOps architecture**: **Python/Docker/CloudWatch logging tells you what the application/infrastructure is doing; Langfuse tells you what happened inside the LLM/RAG execution.**

**Where in this repo**
`app/core/logging_config.py` implements the simple end of this spectrum — root logger, two handlers (§3), one formatter (§4), env-driven level (§11), idempotent setup. It deliberately stops short of `dictConfig` (§14), JSON output (§15), and request IDs (§16), all of which are the natural next steps if this service moves to CloudWatch or another centralized platform. `app/services/logger.py::get_logger` is the pre-existing per-module accessor; `app/guardrails/pii_guard.py` is the reason to be careful about logging prompt text (§19).

**Interview angle**
Q: How would you evolve this project's logging for production?
A: Three changes, roughly in order of payoff. First, JSON to stdout only and drop the file handler — let the platform collect, so logs are queryable by field instead of greppable text (§15). Second, a request-ID `ContextVar` plus a `Filter`, so concurrent requests can be untangled and a user's bug report maps to one trace (§16). Third, move to `dictConfig` loaded from a file so levels can change per subsystem without a deploy (§14). Underneath all of it, keep the discipline that already applies here: `WARNING` means handled, `ERROR` means someone looks (§11), and prompts and responses go to Langfuse rather than general-purpose logs, because they carry PII (§19).

## Debugging Layered Systems: Verify at the Failing Layer

**What it is**
The meta-lesson, and the most transferable thing in this file. One symptom — "the app doesn't work" — hid three faults at three different layers, and the loudest signal (a screenful of Postgres warnings) pointed at the layer that mattered *least* to the actual complaint. The dead link had nothing to do with the database.

**How it works**
- **Separate the symptoms before theorising.** "Logs show DB errors" and "the URL won't open" looked like one problem and were two. Fixing the database would never have fixed the link.
- **Read the logs for what they don't say.** The startup output ended with `Application startup complete` and `Uvicorn running on http://0.0.0.0:8000` — a *healthy* server. That rules out the application layer entirely and points at the hop between host and container.
- **Test from the same vantage point as the failing code.** Connecting to `localhost:2284` from Windows succeeded, which reframed "is Postgres down?" into "who is asking?" — the actual bug.
- **Prefer the authoritative check to the plausible story.** `docker inspect --format "{{.HostConfig.PortBindings}}"` returning `map[]` is proof; "I'm pretty sure I passed `-p`" is not.
- **Verify the fix end to end, not at the layer you touched.** After rebuilding: `HTTP 200` from `http://localhost:8000/`, `Postgres connection pool opened` **and** `Chat memory schema ensured` with no retry warnings, and `logs/app.log` present on the host containing uvicorn's own lines. Three independent confirmations for three independent faults.
- A fourth, self-inflicted fault is worth recording as a Docker-on-Windows gotcha: invoking `docker.exe` by absolute path made the build fail with `docker-credential-desktop: executable file not found in %PATH%`. The credential helper is a *sibling binary* in the same `resources/bin` directory, resolved via `PATH`, so bypassing `PATH` broke authentication to Docker Hub. Prepending that directory to `PATH` fixed it. Tool-invocation errors can masquerade convincingly as project errors.

**Example**
```
Symptom:            "connection refused" spam + link not reachable
                            │
        ┌───────────────────┼───────────────────────┐
        ▼                   ▼                       ▼
Layer 1: app        Layer 2: config          Layer 3: docker networking
DB treated as       localhost inside a       EXPOSE without -p
required            container ≠ host         → no host→container route
        │                   │                       │
        ▼                   ▼                       ▼
open(wait=True)     host.docker.internal     ports: ["8000:8000"]
+ degrade to None   + extra_hosts            → HTTP 200
```

**Where in this repo**
The three fixes landed in `postgres/client.py` + `app/services/chat_memory.py` (layer 1), `docker-compose.yml`'s `environment:` override (layer 2), and `docker-compose.yml`'s `ports:` (layer 3).

**Interview angle**
Q: Walk me through how you'd debug "my app works locally but not in Docker."
A: I'd resist forming a theory until I'd split the symptom, then work outward one hop at a time. Is the process alive and did startup complete (`docker logs`)? Is it bound to `0.0.0.0` rather than `127.0.0.1` (the `CMD`)? Is there a host→container route (`docker ps` PORTS column, `docker inspect .HostConfig.PortBindings`)? Can the container resolve and reach its dependencies *from inside itself* (`docker exec`)? Each check isolates exactly one hop, and each has an authoritative command rather than an inference. Nearly every "works locally, not in Docker" bug is one of two families: an address that means something different inside a namespace (`localhost`), or a boundary that was never actually opened (`EXPOSE` vs `-p`) — and both are invisible if you only read application logs.

## Summary: What Changed and Why

| Change | File | Why |
|---|---|---|
| `open=False` → `open(wait=True, timeout=3)`, degrade to `_pool = None` | `postgres/client.py` | Turn a silent lazy failure into a caught one; DB becomes optional |
| `pool.close(timeout=1.0)` on failure | `postgres/client.py` | Kills the reconnect worker — the source of endless log spam |
| `connect_timeout` in pool `kwargs` | `postgres/client.py` | Caps each libpq attempt so startup penalty stays ~4s |
| `try/except` + 3s acquire timeout on both queries | `app/services/chat_memory.py` | A DB that dies *after* boot must not 500 chat, or park requests for 30s |
| Non-fatal `ensure_schema()` | `postgres/client.py` | Missing DDL permission shouldn't block startup |
| `setup_logging()` with console + rotating file | `app/core/logging_config.py`, `main.py` | Logs that outlive the container, bounded at ~60 MB |
| `_adopt_uvicorn_loggers()` | `app/core/logging_config.py` | Without it, uvicorn's `propagate=False` keeps its lines out of the file |
| `LOG_DIR=/ccp/logs` + `mkdir` | `Dockerfile` | Gives the file handler a writable home in the image |
| `ports: ["8000:8000"]` | `docker-compose.yml` | **The actual fix for the unreachable link** — `EXPOSE` alone routes nothing |
| `DATABASE_URL` → `host.docker.internal`, `extra_hosts` | `docker-compose.yml` | `localhost` in a container is the container; keeps `.env` valid for native runs |
| `./logs:/ccp/logs` bind mount | `docker-compose.yml` | Log file survives `docker compose down` |
| `logs/`, `*.log` | `.gitignore` | Don't commit runtime artifacts |

**Still outstanding (not addressed by this session's work):** all three guardrails fail to load in the container — `llm-guard` isn't installed in the image, and something imports `TFPreTrainedModel`, which recent `transformers` versions no longer export. The container currently serves with input scanning, PII masking, and output scanning **off**. See [02-guardrails-and-safety.md](02-guardrails-and-safety.md) for what that machinery is supposed to do, and note that the pre-warm design in `main.py::_prewarm_guards` degrades rather than crashes here — the same graceful-degradation pattern as the database, which is exactly why the failure is easy to miss.
