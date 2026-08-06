from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    from app.core.configs import get_settings
    return get_settings().guardrail_pii


@lru_cache(maxsize=1)
def _load_anonymize_engine() -> bool:
    """Load the heavy presidio AnalyzerEngine + NER model exactly once per process.

    Returns False immediately (without loading anything) when GUARDRAIL_PII=false.
    Returns True after successful load, False on any error.
    """
    if not _enabled():
        logger.info("PII guard is OFF (GUARDRAIL_PII=false) — no models loaded.")
        return False

    try:
        from llm_guard.input_scanners import Anonymize
        from llm_guard.vault import Vault

        # Warm up with a throw-away vault — this pulls the DeBERTa NER model
        # and presidio's recognizer registry into process memory.
        Anonymize(Vault(), preamble="", use_faker=True, language="en")
        logger.info("PII guard: NER model and recognizer registry loaded.")
        return True
    except ImportError:
        logger.warning("llm-guard not installed — PII masking disabled.")
        return False
    except Exception as exc:
        logger.warning("PII guard init failed (%s) — PII masking disabled.", exc)
        return False


def prewarm() -> None:
    """Call during app startup lifespan to load models before any request arrives.
    No-op when GUARDRAIL_PII=false.
    """
    if not _enabled():
        logger.info("PII guard pre-warm skipped (GUARDRAIL_PII=false).")
        return
    logger.info("Pre-warming PII guard (Anonymize NER + presidio recognizers)…")
    logger.info("Loading pii")
    _load_anonymize_engine()


@dataclass
class PiiContext:
    """Holds a per-request Vault so Anonymize→Deanonymize token mappings stay
    strictly scoped to a single request and never bleed across users.

    Returns a no-op pass-through context when GUARDRAIL_PII=false — callers
    need no conditional logic.

    Usage::

        pii = PiiContext.create()
        safe_query = pii.anonymize(user_query)
        answer = llm_service.generate(safe_query, ...)
        final = pii.deanonymize(safe_query, answer)
    """

    _anonymize: object = field(default=None, repr=False)
    _deanonymize: object = field(default=None, repr=False)
    _active: bool = field(default=False, repr=False)

    @classmethod
    def create(cls) -> "PiiContext":
        if not _load_anonymize_engine():
            return cls(_active=False)

        try:
            from llm_guard.input_scanners import Anonymize
            from llm_guard.output_scanners import Deanonymize
            from llm_guard.vault import Vault

            vault = Vault()
            anonymize = Anonymize(vault, preamble="", use_faker=True, language="en")
            deanonymize = Deanonymize(vault)
            return cls(_anonymize=anonymize, _deanonymize=deanonymize, _active=True)

        except Exception as exc:
            logger.error("PiiContext.create error: %s", exc)
            return cls(_active=False)

    def anonymize(self, text: str) -> str:
        if not self._active or self._anonymize is None:
            return text
        try:
            from llm_guard import scan_prompt

            sanitized, _, _ = scan_prompt([self._anonymize], text)
            if sanitized != text:
                logger.info("PII detected and masked in input.")
            return sanitized
        except Exception as exc:
            logger.error("PII anonymize error (using original): %s", exc)
            return text

    def deanonymize(self, prompt: str, output: str) -> str:
        if not self._active or self._deanonymize is None:
            return output
        try:
            from llm_guard import scan_output as _llm_scan

            restored, _, _ = _llm_scan([self._deanonymize], prompt, output)
            return restored
        except Exception as exc:
            logger.error("PII deanonymize error (using original): %s", exc)
            return output
