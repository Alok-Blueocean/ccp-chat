"""
conftest.py — shared pytest fixtures for all test files.

Ensures lru_cache state (get_settings, _load_scanners) is isolated between
tests so that .env values (all guardrails=true) do not bleed into unit tests
that expect guards to be OFF.
"""
from __future__ import annotations

import os
import pytest


# ---------------------------------------------------------------------------
# Override .env guardrail flags for the entire test session via env vars.
# Pydantic Settings reads os.environ before the .env file, so these win.
# Individual tests can still re-patch _enabled() for focused on/off testing.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True, scope="session")
def disable_guardrails_env():
    overrides = {
        "GUARDRAIL_INPUT": "false",
        "GUARDRAIL_OUTPUT": "false",
        "GUARDRAIL_PII": "false",
        "GUARDRAIL_RATE_LIMIT": "false",
    }
    original = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    yield
    for k, v in original.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ---------------------------------------------------------------------------
# Clear all guard lru_caches before every test so that toggling _enabled()
# in one test doesn't pollute the next.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def clear_guard_caches():
    yield
    try:
        from app.core.configs import get_settings
        get_settings.cache_clear()
    except Exception:
        pass

    try:
        from app.guardrails.input_guard import _load_scanners
        _load_scanners.cache_clear()
    except Exception:
        pass

    try:
        from app.guardrails.output_guard import _load_scanners as _out
        _out.cache_clear()
    except Exception:
        pass

    try:
        from app.guardrails.pii_guard import _load_anonymize_engine
        _load_anonymize_engine.cache_clear()
    except Exception:
        pass
