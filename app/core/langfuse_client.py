import os

from langfuse import Langfuse

from app.core.configs import get_settings


def _init() -> Langfuse | None:
    s = get_settings()
    if not s.langfuse_public_key or not s.langfuse_secret_key:
        return None
    # Set env vars so @observe decorators on service methods pick them up automatically
    os.environ["LANGFUSE_PUBLIC_KEY"] = s.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = s.langfuse_secret_key
    os.environ["LANGFUSE_HOST"] = s.langfuse_host
    return Langfuse(
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key,
        host=s.langfuse_host,
    )


_client = _init()


def get_langfuse() -> Langfuse | None:
    return _client
