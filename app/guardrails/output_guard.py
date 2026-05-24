from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import HTTPException

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_scanners():
    """Load output scanners once.

    Scanners applied in order:
    1. Sensitive — strip / block responses that leak credentials, keys, or PII patterns.
    2. Toxicity  — block harmful or hateful model outputs.
    """
    from llm_guard.output_scanners import Sensitive, Toxicity

    return [
        Sensitive(),
        Toxicity(threshold=0.7),
    ]


def scan_output(prompt: str, output: str) -> str:
    """Scan LLM output for sensitive data and toxicity.

    Returns the (possibly sanitized) output on success.
    Raises HTTP 502 when any scanner blocks the output.
    Falls through with the original output when llm-guard is not installed or
    an unexpected error occurs.
    """
    try:
        from llm_guard import scan_output as _llm_scan_output

        scanners = _load_scanners()
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
