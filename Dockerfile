# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /ccp

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies separately so application code changes
# don't invalidate the dependency layer.
COPY requirements.txt .

# Persistent BuildKit pip cache.
# Changing requirements.txt causes pip install to run again,
# but already-downloaded packages can be reused.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Install spaCy model
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m spacy download en_core_web_lg

# Application code
COPY . .

ENV PORT=8000

# App logs go to stderr *and* this directory. Bind-mount it to keep logs
# after the container is removed: -v ${PWD}/logs:/ccp/logs
ENV LOG_DIR=/ccp/logs
RUN mkdir -p /ccp/logs

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]