from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    from app.core.configs import get_settings
    return get_settings().guardrail_output


@lru_cache(maxsize=1)
def _load_scanners():
    """Load output scanners once.  Returns an empty list when GUARDRAIL_OUTPUT=false
    so no models are pulled into memory.

    Scanners (when enabled):
    1. Sensitive — strip / block responses that leak credentials or PII patterns.
    2. Toxicity  — block harmful or hateful model outputs.
    """
    if not _enabled():
        logger.info("Output guard is OFF (GUARDRAIL_OUTPUT=false) — no models loaded.")
        return []

    from llm_guard.output_scanners import Sensitive, Toxicity
    logger.info("Loading tokenizer output")
    return [
        Sensitive(),
        Toxicity(threshold=0.7),
    ]


def scan_output(prompt: str, output: str) -> str:
    """Scan LLM output for sensitive data and toxicity.

    Returns the (possibly sanitized) output on success.
    Raises HTTP 502 when a scanner blocks the output.
    Pass-through when GUARDRAIL_OUTPUT=false or llm-guard is not installed.
    """
    if not _enabled():
        return output

    try:
        from llm_guard import scan_output as _llm_scan_output

        scanners = _load_scanners()
        if not scanners:
            return output

        sanitized, results_valid, results_score = _llm_scan_output(scanners, prompt, output)

        blocked = [name for name, ok in results_valid.items() if not ok]
        if blocked:
            logger.warning(
                "Output blocked: scanners=%s scores=%s", blocked, results_score
            )
            raise HTTPException(
                status_code=502,
                detail="LLM response failed safety check.",
            )

        return sanitized

    except HTTPException:
        raise
    except ImportError:
        logger.warning("llm-guard not installed — output scanning skipped.")
        return output
    except Exception as exc:
        logger.error("Output guard unexpected error (allowing): %s", exc)
        return output
