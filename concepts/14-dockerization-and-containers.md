# Dockerization & Containers Concepts

General container/Docker knowledge. This repo now *does* have a `Dockerfile` and a `docker-compose.yml`, and `INTERVIEW_QA.md`'s "Deployment & Architecture" section discusses deploying this bot as an AWS Lambda **container image** (to fit native dependencies like Tesseract OCR/OpenCV past Lambda's 250MB zip limit).

For how these concepts played out concretely on this codebase — the `EXPOSE`-without-`-p` bug that made the app unreachable, `localhost` vs `host.docker.internal` for the Postgres connection, and where `json-file` logs actually live on Docker Desktop — see [15-resilience-logging-and-container-networking.md](15-resilience-logging-and-container-networking.md).

## Images vs. Containers

**What it is**

A **Docker image** is a read-only, layered filesystem snapshot plus metadata (entrypoint, default command, exposed ports, env vars) — think of it as a class. A **container** is a running (or stopped) instance created from an image, with its own writable layer on top and its own process namespace — think of it as an object instantiated from that class. The same image can be instantiated into many independent containers simultaneously, each isolated from the others and from the host, sharing the host's kernel (unlike a VM, which virtualizes hardware and runs a full separate kernel).

**How it works**

- `docker build` reads a `Dockerfile` and produces an image, one layer per instruction.
- `docker run <image>` creates a new container from that image: a thin writable layer is added on top of the image's read-only layers (copy-on-write), so multiple containers from the same image share the base layers on disk and only diverge in their own writable layer.
- Containers get their own process ID namespace, network namespace, and filesystem view via Linux namespaces + cgroups — isolation is enforced by the kernel, not by a hypervisor.
- Stopping a container doesn't delete it (`docker ps -a` still shows it); `docker rm` actually removes it. Its writable layer's contents are lost on removal unless you used a volume.
- Because containers share the host kernel, a container starts in milliseconds, not the seconds-to-minutes a VM boot takes — the tradeoff is a weaker isolation boundary than a VM (relevant to the security section below).

**Example**

```bash
docker build -t ccp-chat:latest .          # build image from Dockerfile in cwd
docker run -d --name chat1 -p 8000:8000 ccp-chat:latest   # container #1
docker run -d --name chat2 -p 8001:8000 ccp-chat:latest   # container #2, same image
docker ps                                   # both running, independent, same underlying image layers
docker stop chat1 && docker rm chat1        # container gone; image untouched
```

**Interview angle**

Q: How is a container different from a lightweight VM?
A: A VM virtualizes hardware and boots its own full kernel — strong isolation, slow startup, real per-VM memory overhead. A container shares the host kernel and is isolated via namespaces/cgroups — much faster to start (no kernel boot) and lighter on memory, but the isolation boundary is thinner (a kernel-level exploit can potentially escape a container in a way it can't escape a VM). Containers optimize for density and speed; VMs optimize for isolation strength.

---

## Dockerfile, Layers & Build Cache

**What it is**

A `Dockerfile` is a sequential list of instructions (`FROM`, `COPY`, `RUN`, `ENV`, `CMD`, ...) that Docker executes to build an image, one layer per instruction. Each layer is cached by content hash — if an instruction and its inputs haven't changed since the last build, Docker reuses the cached layer instead of re-running it. This makes instruction *order* a real performance (and correctness) decision: put things that change rarely (installing system/OS packages) before things that change often (copying your application source code), so a source-code edit doesn't invalidate an expensive dependency-install layer.

**How it works**

- Each instruction produces a new layer; layers stack to form the final image filesystem.
- The build cache key for a layer is a hash of the instruction plus its inputs — for `COPY`, that includes the actual file contents being copied, not just the instruction text.
- If layer N's cache is invalidated (its inputs changed), every layer *after* it must be rebuilt too, even if their own inputs didn't change — cache invalidation cascades forward, never backward.
- `.dockerignore` excludes files from the build context sent to the daemon (e.g. `.git`, `node_modules`, local venvs) — smaller context means faster builds and avoids accidentally invalidating cache or baking in unwanted files.
- `RUN` instructions that install packages should pin versions and combine related steps into one layer (`apt-get update && apt-get install -y x y z`) — splitting them across multiple `RUN` lines works, but a stale cached `apt-get update` layer combined with a fresh `install` layer can silently install outdated package lists.

**Example**

```dockerfile
# Bad ordering: any source change invalidates the expensive pip install below it
FROM python:3.12-slim
COPY . /app
RUN pip install -r /app/requirements.txt
CMD ["python", "main.py"]

# Good ordering: dependency layer only rebuilds when requirements.txt itself changes
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

In the "good" version, editing `app/routers/chat.py` and rebuilding only re-runs the final `COPY . .` layer (and anything after it) — the `pip install` layer, which might take 30+ seconds for a stack like this repo's (llama-index, qdrant-client, presidio, spacy models), stays cached.

**Interview angle**

Q: Your Docker builds take 3 minutes even for a one-line code change. What's the fix?
A: Almost always a `Dockerfile` ordering problem — dependency installation is happening *after* the application source is copied, so every code change invalidates and re-runs the (slow) install layer. Reorder so dependency manifests (`requirements.txt`, `package.json`) are copied and installed *before* copying the rest of the source; that layer then only invalidates when dependencies actually change, and a code-only change rebuilds in seconds.

---

## Multi-Stage Builds

**What it is**

A multi-stage build uses multiple `FROM` instructions in one `Dockerfile`, where later stages can selectively copy artifacts from earlier stages via `COPY --from=<stage>`. This lets you use a heavyweight image with full build tooling (compilers, dev headers, package managers) to *produce* an artifact, then copy only that artifact into a minimal final-stage image — so your shipped image doesn't carry the build toolchain's weight or attack surface at all.

**How it works**

- Each `FROM` starts a new, independent build stage; stages can be named (`FROM golang:1.22 AS builder`) for later reference.
- Only the *final* stage's layers end up in the tagged output image — earlier stages exist only during the build and are discarded (though Docker's build cache still keeps them around for reuse on the next build).
- This is most dramatic for compiled languages (a Go/Rust binary built in a full toolchain image, then copied alone into a `scratch` or `alpine` final image — the compiler itself never ships), but it's genuinely useful for Python/Node too: a stage with build-essential/gcc for compiling native extensions (e.g. this repo's numpy/spacy/presidio dependency chain), then a slim final stage with only the installed site-packages copied over.

**Example**

```dockerfile
# Stage 1: build environment with full toolchain
FROM python:3.12 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: minimal runtime image
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

The final image never contains `gcc`, build headers, or pip's build cache — just the installed packages and the app code, meaningfully smaller and with a smaller attack surface than shipping the builder image itself.

**Interview angle**

Q: Why not just install everything in one stage and call it done?
A: A single-stage image ships every tool used to *build* the app, not just the app itself — compilers, dev headers, package manager caches — inflating image size (slower pulls/cold starts) and attack surface (more installed software = more CVEs to track) for no runtime benefit. Multi-stage builds let you pay that cost only during the build and ship a lean final image containing just what's needed to *run*.

---

## Docker Compose (Multi-Container Orchestration)

**What it is**

Docker Compose defines and runs a multi-container application from a single declarative YAML file — instead of a string of manual `docker run` commands (each with its own flags for ports, volumes, env vars, and networks), `docker compose up` brings the whole stack up (and `down` tears it all down) as one unit. It's the natural fit for local development and small deployments where an app is really *several* containers that need to talk to each other — e.g. this repo's actual dependency list (FastAPI app, Postgres, Redis, and in production a remote Qdrant) maps naturally onto one API service plus Postgres and Redis containers.

**How it works**

- Each top-level key under `services:` becomes one container definition (image or build context, ports, environment, volumes, dependencies).
- `depends_on` controls startup *order* but not readiness — a Postgres container reporting "started" is not the same as "accepting connections," so apps still need their own retry/wait logic (or a `healthcheck` + `condition: service_healthy` dependency) rather than assuming a dependency is ready the instant it starts.
- Compose creates a private network by default where services can reach each other **by service name** as a DNS hostname — no manual IP wiring, no `localhost` (that would mean "inside this container," not "the other container").
- `.env` files at the compose-file's location are auto-loaded for variable substitution inside the YAML (`${DATABASE_URL}`), separate from any `.env` the *application itself* might load at runtime.

**Example**

```yaml
# docker-compose.yml — illustrative shape for a service like this repo
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/ccp
      - REDIS_URL=redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started

  postgres:
    image: postgres:16
    environment:
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=ccp
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 5s
      retries: 5
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

volumes:
  pgdata:
```

Note `DATABASE_URL` here uses the hostname `postgres`, not `localhost` or an IP — Compose's built-in DNS resolves that to the Postgres container automatically because they share a network and `postgres` is that service's name.

**Interview angle**

Q: Your API container starts before Postgres is actually ready to accept connections, and the app crashes on boot. What's wrong?
A: `depends_on` alone only sequences container *start* order, not application *readiness* — Postgres's container process starts almost immediately, but the database itself takes a moment longer to accept connections. The fix is a `healthcheck` on the Postgres service (e.g. `pg_isready`) combined with `depends_on: condition: service_healthy` on the dependent service, so Compose actually waits for "ready," not just "started." Belt-and-suspenders: the app itself should still retry its DB connection on boot rather than assuming any orchestration guarantee is airtight.

---

## Volumes & Bind Mounts (Persistence)

**What it is**

Containers are meant to be ephemeral — their writable layer disappears when the container is removed. Anything that needs to *survive* a container being destroyed and recreated (database files, uploaded content) needs to live outside that writable layer, in either a **named volume** (Docker-managed storage, the recommended default) or a **bind mount** (a specific path on the host filesystem mapped directly into the container — useful for local dev, e.g. mounting your source tree so code edits show up without a rebuild).

**How it works**

- A named volume (`docker volume create mydata`, or declared inline in Compose) is managed entirely by Docker — you refer to it by name, not by a host path, and Docker decides where it physically lives.
- A bind mount instead maps an exact host path (`-v /home/user/project:/app`) into the container — changes are visible on both sides instantly, which is exactly why it's the standard trick for live-reloading source code in local dev, but it also ties your setup to that host's specific filesystem layout, which is a poor fit for portable production deployments.
- Volumes (named or bind) persist independently of the container's lifecycle — `docker rm` on a container leaves its attached volumes untouched unless you explicitly pass `-v`/`--volumes` to also remove them.
- Anything written *inside the container but not in a mounted path* is lost the moment the container is removed — this is the single most common "why did my data disappear" surprise for people new to containers.

**Example**

```bash
# Named volume — Docker manages the storage location
docker run -d --name pg -v pgdata:/var/lib/postgresql/data postgres:16

# Bind mount — host path is explicit, great for local dev hot-reload
docker run -it -v $(pwd):/app -p 8000:8000 ccp-chat:latest uvicorn main:app --reload
```

Restarting or even fully removing and recreating the `pg` container preserves everything in `pgdata` — the actual database files live in the named volume, not the container's own writable layer.

**Interview angle**

Q: You redeployed your database container and lost all the data. What went wrong?
A: The database's data directory was never mapped to a volume — it lived only in the container's own writable layer, which is discarded when the container is removed. Any stateful service (a database, anything writing files it needs to keep) must have its data directory mounted to a named volume (production) or bind mount (dev), independent of the container itself, precisely so that removing/recreating the *container* doesn't touch the *data*.

---

## Networking

**What it is**

By default, Docker gives every container its own network namespace and attaches it to a bridge network, where containers on the same network can reach each other by IP — and, in user-defined networks (including the ones Compose creates automatically), by **service/container name** via Docker's built-in embedded DNS. Ports aren't reachable from outside the host unless explicitly published (`-p host_port:container_port`); by default a container's ports are only visible to other containers on the same network.

**How it works**

- The default `bridge` network provides container-to-container connectivity by IP but *not* automatic DNS name resolution — that's specifically a feature of user-defined bridge networks (which is what Compose sets up for you automatically, which is why `postgres` works as a hostname in the earlier example).
- `-p 8000:80` maps host port 8000 to container port 80 — omit it, and the container's port 80 is reachable from other containers on the same network but not from the host machine or the outside world.
- `EXPOSE 80` in a `Dockerfile` is documentation/metadata, not an actual port mapping — it does nothing to make a port reachable by itself; `-p`/`ports:` is what actually does the mapping.
- Multiple containers can share a network namespace or be placed on multiple networks simultaneously (e.g. a container that needs to reach both a "frontend" network and a "backend" network without both networks' members seeing each other).

**Example**

```bash
docker network create backend
docker run -d --name db --network backend postgres:16
docker run -d --name api --network backend -p 8000:8000 ccp-chat:latest
```
Inside `api`'s container, connecting to `db:5432` works — same user-defined network, DNS resolves `db` to the right container IP. From the host machine or the internet, only `api`'s published port 8000 is reachable; `db`'s port 5432 was never published and is invisible outside the `backend` network entirely, which is also a reasonable default security posture (don't expose the database directly).

**Interview angle**

Q: Two containers on the default bridge network can't resolve each other by name, but the same setup works fine under Docker Compose — why?
A: Docker's automatic container-name DNS resolution only works on *user-defined* networks, not the legacy default `bridge` network (which only gives IP-based connectivity, for backward-compatibility reasons). Compose creates a user-defined network for you automatically, which is why hostnames "just work" there — replicating the same behavior with plain `docker run` requires explicitly creating a network (`docker network create`) and attaching both containers to it.

---

## Environment Variables & Secrets

**What it is**

Configuration that varies per environment (dev/staging/prod) — database URLs, API keys, feature flags — is passed into a container at runtime via environment variables, not baked into the image at build time. This matters for more than convenience: an image is meant to be immutable and portable across environments, and baking a specific environment's secrets into it both defeats that portability and — critically — means the secret is now permanently embedded in an image layer, retrievable by anyone who can pull or inspect that image, forever, even after the "secret" is rotated elsewhere.

**How it works**

- `docker run -e KEY=value` or `env_file:`/`environment:` in Compose inject variables into the container's environment at *runtime*, not at build time — they never become part of the image itself.
- A `COPY .env .` or a `RUN export SECRET=...` baked into a `Dockerfile`, by contrast, permanently embeds that value into an image layer — even if you later delete the file in a subsequent layer, the value still exists in the earlier layer's diff and is recoverable (`docker history`, layer inspection).
- For anything more sensitive than "config that's annoying but not catastrophic to leak," dedicated secrets management (Docker Swarm secrets, Kubernetes Secrets, a cloud secrets manager like AWS Secrets Manager/Vault, mounted as files rather than env vars) reduces exposure further — env vars are visible to anything that can inspect the process (`docker inspect`, `/proc/<pid>/environ` inside the container, accidental logging of the environment) in a way that's easy to overlook.
- `.dockerignore` should always exclude `.env` files and credential files from the build context, specifically to prevent an accidental `COPY . .` from baking secrets into the image without anyone noticing.

**Example**

```dockerfile
# BAD — permanently bakes a secret into an image layer
FROM python:3.12-slim
COPY .env /app/.env
CMD ["python", "main.py"]
```
```bash
# GOOD — secret only exists at runtime, never in the image
docker run -e DATABASE_URL="postgresql://..." -e OPENAI_API_KEY="sk-..." ccp-chat:latest
```

This is precisely the mistake flagged elsewhere in this repo's `concepts/` notes about `dags/producer_dags.py` and `dags/consumer_dags.py`: a Redis connection string with live credentials hardcoded directly in committed source. The exact same failure mode applies one layer down if a Dockerfile ever does `COPY .env .` or hardcodes a secret in a `RUN`/`ENV` instruction — the fix in both cases is the same principle: secrets belong in runtime injection (env vars from a secrets manager, or `.env` files that are never committed and never copied into the image), not in anything that gets version-controlled or baked into a build artifact.

**Interview angle**

Q: A teammate suggests just `COPY`ing the `.env` file into the image so it "just works" everywhere. What's wrong with that?
A: It defeats image portability (the image is now tied to one environment's config) and permanently embeds the secret into an image layer — anyone who can pull that image, including via a registry misconfiguration or a leaked image, gets every secret in that file, and deleting the file in a later layer doesn't remove it from the earlier layer's history. Secrets should be injected at container *runtime* (env vars, mounted secret files, a secrets manager), never copied into the image at build time.

---

## Container Registries & Image Tagging

**What it is**

A registry (Docker Hub, AWS ECR, GitHub Container Registry, Google Artifact Registry) stores and serves built images, addressed by `<registry>/<repository>:<tag>`. Tags are how you version and distinguish images — but a tag is just a mutable pointer to a specific image digest, not an immutable identity itself, which has real operational consequences.

**How it works**

- `docker push`/`docker pull` move images to/from a registry; `docker tag` gives a locally-built image an additional name/tag without rebuilding it.
- The special tag `latest` is *not* automatically "the newest image" — it's just a tag like any other, conventionally used for "the most recently pushed untagged build," and can be silently overwritten. Relying on `latest` in production is a common source of "it worked yesterday, now it doesn't" incidents, because you have no record of exactly which build is actually running.
- Best practice: tag every build with something immutable and traceable — a git commit SHA, a semantic version, a build timestamp — and treat `latest` (if used at all) as a convenience alias, never the thing production actually deploys.
- Every image (and every layer) is content-addressed by a SHA256 digest — pulling by digest (`image@sha256:...`) rather than by tag guarantees you get the exact same bytes every time, immune to a tag being retargeted later.

**Example**

```bash
docker build -t myrepo/ccp-chat:$(git rev-parse --short HEAD) .
docker tag myrepo/ccp-chat:$(git rev-parse --short HEAD) myrepo/ccp-chat:latest
docker push myrepo/ccp-chat:$(git rev-parse --short HEAD)
docker push myrepo/ccp-chat:latest
```
Production deploy configs should reference the commit-SHA tag (`myrepo/ccp-chat:a1b2c3d`), not `latest` — that way "what's actually running in prod" is answerable by reading a deploy config, not by guessing which push happened most recently.

**Interview angle**

Q: What can go wrong if your deployment pipeline always pulls `image:latest`?
A: `latest` is a mutable pointer — a new push can retarget it at any time, meaning "redeploying the exact same config" can silently pull a different image than what you tested, and there's no way to tell from the tag alone which build is currently deployed anywhere. Rollbacks become guesswork instead of "redeploy this exact known-good tag." Immutable, traceable tags (commit SHA, semver) fix both problems at once.

---

## Health Checks & Container Lifecycle

**What it is**

A `HEALTHCHECK` instruction (or an orchestrator-level equivalent, like Compose's `healthcheck:` or a Kubernetes liveness/readiness probe) tells the runtime how to actually determine whether a running container is *working*, not just whether its main process happens to still be alive. Without one, "the container is up" and "the container is actually serving traffic correctly" are conflated — a hung process that never crashes but also never responds looks identical to a healthy one from the outside.

**How it works**

- `HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1` runs periodically inside the container; Docker tracks the result and exposes it via `docker ps` (`healthy`/`unhealthy`/`starting`).
- Orchestrators use health status to make real decisions: Compose's `condition: service_healthy` gates dependent services on it; Kubernetes' liveness probe restarts a container that fails repeatedly, while its separate readiness probe controls whether traffic gets routed to it at all (a container can be "alive but not ready," e.g. still loading a model — exactly the pre-warming pattern in this repo's own `main.py::lifespan`).
- Restart policies (`--restart unless-stopped`, `always`, `on-failure`) determine what happens when a container's main process exits — critical for long-running services that should self-heal from transient crashes without manual intervention.
- A container's `CMD`/`ENTRYPOINT` process is PID 1 inside its namespace — if that process exits, the container exits, regardless of any background threads/child processes it may have spawned; this is a frequent source of "why did my container exit even though it looked fine" confusion.

**Example**

```dockerfile
FROM python:3.12-slim
...
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
`--start-period=15s` gives the app a grace window (e.g. for model pre-warming at startup) before failed checks start counting toward `--retries` — without it, a slow-but-healthy startup could get marked unhealthy and restarted before it ever had a chance to finish loading.

**Interview angle**

Q: Your container shows as "running" but the app isn't actually responding to requests. Why doesn't Docker restart it automatically?
A: Without a `HEALTHCHECK`, Docker only knows whether the main process is still alive, not whether it's functioning — a hung event loop, a deadlocked thread, or a crashed-but-not-exited state all look identical to "running" from Docker's point of view. A `HEALTHCHECK` that actually exercises the app (hits a real endpoint) closes that gap, and pairing it with a restart policy turns "detected unhealthy" into "automatically recovered" instead of a silent outage.

---

## Container Image Size, Base Images & Security Basics

**What it is**

Base image choice and final image size affect cold-start latency, attack surface, and storage/transfer cost simultaneously — a `python:3.12` full image is roughly 900MB-1GB+, `python:3.12-slim` cuts that substantially by dropping build tools and docs, and `python:3.12-alpine` (musl libc instead of glibc) can be smaller still but sometimes trades away compatibility with packages that ship prebuilt glibc binary wheels (a real gotcha for a dependency-heavy stack like this repo's numpy/spacy/presidio chain, where Alpine can mean falling back to slow from-source compiles or outright failures).

**How it works**

- Smaller images pull faster (matters a lot for cold starts — e.g. this repo's own discussion in `INTERVIEW_QA.md` of Lambda cold-start latency) and carry fewer installed packages, which directly means fewer potential CVEs to track and patch.
- Running as a non-root user inside the container (`USER appuser` in the Dockerfile, after creating that user) limits the blast radius if the application process itself is compromised — by default, unless told otherwise, a container's main process runs as root *inside the container*, which is a meaningfully larger risk than it sounds given kernel-level container-escape vulnerabilities have historically existed.
- Image scanning tools (Trivy, Grype, Docker Scout, cloud-provider registry scanning) check installed packages against known-CVE databases — cheap to run in CI, and it turns "did that base image update fix a CVE" from a guess into a checked fact.
- Multi-stage builds (covered above) are themselves a security lever, not just a size optimization — every tool you don't ship is a tool an attacker who gets code execution inside the container can't use.

**Example**

```dockerfile
FROM python:3.12-slim
RUN useradd --create-home appuser
WORKDIR /app
COPY --chown=appuser:appuser . .
RUN pip install --no-cache-dir -r requirements.txt
USER appuser
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
Without the `useradd`/`USER appuser` lines, the app would run as root inside the container by default — a small change that meaningfully reduces what an attacker gains from compromising the running process.

**Interview angle**

Q: Your security review flags that your production containers run as root. Why does that matter if they're isolated in containers anyway?
A: Container isolation (namespaces/cgroups) is a real boundary, but it's not as strong as a VM's — kernel-level container-escape vulnerabilities are a known, recurring category of CVE. Running as root inside the container means that if such an escape (or even a less exotic host misconfiguration, like a mounted host path with weak permissions) is ever exploited, the attacker inherits root, not a constrained user. Running as a dedicated non-root user costs almost nothing and removes an entire tier of "what's the worst case if this container is compromised."

---

## Docker Compose vs. Kubernetes — Knowing When You've Outgrown Compose

**What it is**

Compose is a single-host tool: it starts and links containers on one machine, with no built-in story for running across multiple machines, automatically healing a crashed node, or scaling a service to N replicas behind a load balancer. Kubernetes (and managed equivalents like ECS/EKS/GKE) solves the multi-host orchestration problem Compose was never designed for — at the cost of substantially more operational complexity than a single YAML file and a `docker compose up`.

**How it works**

- Compose's unit of deployment is "this host, this set of containers, defined once." There's no native concept of "run 5 replicas of this service across 3 machines and keep exactly 5 running even if a machine dies."
- Kubernetes introduces its own vocabulary for that: Pods (one or more containers scheduled together), Deployments (declarative desired-replica-count management with self-healing), Services (stable networking/load-balancing across replicas), and a scheduler that places Pods across a cluster of nodes.
- The practical trigger for reaching past Compose: you need more than one host, you need automatic recovery when a whole machine (not just a container) goes down, or you need to scale a specific service's replica count independently of the rest of the stack under real load.
- Many real systems don't need this at all — a single well-specced host running Compose (or even this repo's actual AWS Lambda container deployment, which sidesteps orchestration entirely by having the cloud provider manage scaling-to-zero and concurrency) is a completely legitimate, simpler alternative when the traffic pattern fits.

**Example**

This repo's own documented deployment path (per `INTERVIEW_QA.md`) is actually a third option distinct from both: an AWS Lambda **container image** (not a Compose stack, not a Kubernetes cluster) — Lambda uses the Docker image format as a packaging mechanism (to work around the 250MB zip-deploy size limit for native dependencies like Tesseract/OpenCV) while AWS itself handles all scaling/orchestration behind the scenes. It's worth being able to name explicitly that "using Docker" and "needing Kubernetes-style orchestration" are two independent questions — you can use Docker's image format for packaging while letting a serverless platform do the orchestration entirely.

**Interview angle**

Q: If Docker Compose can run multiple containers together, why would you ever need Kubernetes?
A: Compose orchestrates containers on one host — it has no answer for "what happens when that host dies," or "scale this one service to 20 replicas across a cluster while everything else stays at 2." Kubernetes exists specifically for multi-host scheduling, self-healing, and per-service scaling at that scale. The corollary worth stating explicitly: plenty of real systems (including this repo's own Lambda-container deployment) never need either Compose's multi-container orchestration *or* Kubernetes's cluster orchestration — a single container behind a serverless platform, or a single host running Compose, is often the right-sized answer, not a stepping stone you're expected to "graduate" from.
