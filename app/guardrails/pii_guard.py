from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_anonymize_engine():
    """Load the heavy presidio AnalyzerEngine + NER model exactly once per process.

    ``lru_cache`` guarantees a single load regardless of how many requests
    call ``PiiContext.create()`` concurrently.  Returns None when
    llm-guard / presidio / the NER model are unavailable.
    """
    try:
        from llm_guard.input_scanners import Anonymize
        from llm_guard.output_scanners import Deanonymize
        from llm_guard.vault import Vault

        # Warm up with a throw-away vault — this pulls the DeBERTa NER model
        # and presidio's recognizer registry into process memory.
        _warm_vault = Vault()
        _warm = Anonymize(_warm_vault, preamble="", use_faker=True, language="en")

        logger.info("PII guard: NER model and recognizer registry loaded.")
        return True
    except ImportError:
        logger.warning("llm-guard not installed — PII masking disabled.")
        return False
    except Exception as exc:
        logger.warning("PII guard init failed (%s) — PII masking disabled.", exc)
        return False


def prewarm() -> None:
    """Call once during app startup lifespan to load models before any request arrives."""
    logger.info("Pre-warming PII guard (Anonymize NER + presidio recognizers)…")
    _load_anonymize_engine()


@dataclass
class PiiContext:
    """Holds a per-request Vault so Anonymize→Deanonymize token mappings stay
    strictly scoped to a single request and never bleed across users.

    The heavy NER model is loaded by ``_load_anonymize_engine()`` which is
    cached at the module level, so each ``PiiContext.create()`` call only
    creates a fresh Vault (lightweight) and wires up new scanner objects
    that reuse the already-loaded model weights.

    Usage::

        pii = PiiContext.create()
        safe_query = pii.anonymize(user_query)          # mask PII before LLM
        answer = llm_service.generate(safe_query, ...)
        final = pii.deanonymize(safe_query, answer)     # restore original values
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
        """Replace PII entities with faker-generated placeholders."""
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
        """Restore original PII values in the LLM output."""
        if not self._active or self._deanonymize is None:
            return output
        try:
            from llm_guard import scan_output as _llm_scan

            restored, _, _ = _llm_scan([self._deanonymize], prompt, output)
            return restored
        except Exception as exc:
            logger.error("PII deanonymize error (using original): %s", exc)
            return output
