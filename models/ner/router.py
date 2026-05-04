from __future__ import annotations

import os
from typing import Any

from .inference import (
    extract_entities as extract_entities_underthesea,
    _dedup_keep_order,
    _extract_roles,
    _prune_substrings,
)
from .transformers_backend import (
    TransformersNERUnavailableError,
    extract_entities_transformers,
)


def extract_entities(text: str) -> list[dict[str, Any]]:
    """
    Router:
      - if NER_BACKEND=transformers -> transformers only (raise if fails)
      - if NER_BACKEND=auto -> transformers then fallback underthesea
      - if NER_BACKEND=underthesea -> underthesea
    """
    backend = os.getenv("NER_BACKEND", "auto").strip().lower()

    if backend in {"auto", "transformers"}:
        try:
            ents = extract_entities_transformers(text)
            proper = [{"text": e["text"], "type": e["type"]} for e in ents]
            roles = _extract_roles(text)
            merged = _dedup_keep_order(proper + roles)
            return _prune_substrings(merged)
        except TransformersNERUnavailableError:
            if backend == "transformers":
                raise
        except Exception:
            if backend == "transformers":
                raise

    if backend == "underthesea":
        return extract_entities_underthesea(text)

    # fallback
    return extract_entities_underthesea(text)