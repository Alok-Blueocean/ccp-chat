from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    from app.core.configs import get_settings
    return get_settings().guardrail_input


@lru_cache(maxsize=1)
def _load_scanners():
    """Load input scanners once.  Returns an empty list when GUARDRAIL_INPUT=false
    so no models are pulled into memory.

    Scanners (when enabled):
    1. TokenLimit       — reject inputs that exceed the token budget.
    2. PromptInjection  — detect attempts to hijack system instructions.
    3. Toxicity         — reject hateful / harmful content.
    """
    if not _enabled():
        logger.info("Input guard is OFF (GUARDRAIL_INPUT=false) — no models loaded.")
        return []

    from llm_guard.input_scanners import PromptInjection, TokenLimit, Toxicity
    logger.info("Loading tokenizer")
    return [
        TokenLimit(limit=512, encoding_name="cl100k_base"),
        PromptInjection(threshold=0.5),
        Toxicity(threshold=0.7),
    ]


def scan_input(prompt: str) -> str:
    """Scan and sanitize a user prompt.

    Returns the (possibly sanitized) prompt on success.
    Raises HTTP 400 when a scanner blocks the input.
    Pass-through when GUARDRAIL_INPUT=false or llm-guard is not installed.
    """
    if not _enabled():
        return prompt

    try:
        from llm_guard import scan_prompt

        scanners = _load_scanners()
        if not scanners:
            return prompt

        sanitized, results_valid, results_score = scan_prompt(scanners, prompt)

        blocked = [name for name, ok in results_valid.items() if not ok]
        if blocked:
            logger.warning(
                "Input blocked: scanners=%s scores=%s prompt_preview=%.80r",
                blocked, results_score, prompt,
            )
            raise HTTPException(
                status_code=400,
                detail=f"Input rejected by safety check: {', '.join(blocked)}.",
            )

        return sanitized

    except HTTPException:
        raise
    except ImportError:
        logger.warning("llm-guard not installed — input scanning skipped.")
        return prompt
    except Exception as exc:
        logger.error("Input guard unexpected error (allowing): %s", exc)
        return prompt
