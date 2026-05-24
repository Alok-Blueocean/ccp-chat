from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import HTTPException

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_scanners():
    """Load input scanners once; models are cached inside llm-guard.

    Scanners applied in order:
    1. TokenLimit   — reject inputs that exceed the token budget before any model call.
    2. PromptInjection — detect attempts to hijack or override system instructions.
    3. Toxicity     — reject hateful / harmful content.
    """
    from llm_guard.input_scanners import PromptInjection, TokenLimit, Toxicity

    return [
        TokenLimit(limit=512, encoding_name="cl100k_base"),
        PromptInjection(threshold=0.5),
        Toxicity(threshold=0.7),
    ]


def scan_input(prompt: str) -> str:
    """Scan and sanitize a user prompt.

    Returns the (possibly sanitized) prompt on success.
    Raises HTTP 400 when any scanner blocks the input.
    Falls through with the original prompt when llm-guard is not installed or
    a scanner raises an unexpected error so the app degrades gracefully.
    """
    try:
        from llm_guard import scan_prompt

        scanners = _load_scanners()
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
